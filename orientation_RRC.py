"""
orientation_RRC — checkpoint22 segmentation + upright cube frames.

Uses the same clustering as ``checkpoint22.py``. The **center** uses identical-cube
geometry: midpoint of extent along robot-up plus robust in-plane median (with a
small OBB blend). Orientation forces **+Z = robot +Z** in camera frame, **+X**
from the first usable horizontal OBB axis, **+Y = Z × X**.

Run from any working directory: project root is added to ``sys.path`` so
``utils.zed_camera`` and checkpoints resolve.
"""
from __future__ import annotations

import os
import sys
import time
import traceback

# Project root (folder containing this file) — fixes ``utils.*`` / checkpoint imports
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import cv2
import numpy
import open3d as o3d
from xarm.wrapper import XArmAPI

from checkpoint0 import get_transform_camera_robot
from checkpoint1 import grasp_cube, place_cube, GRIPPER_LENGTH, robot_ip
from utils.vis_utils import draw_pose_axes
from utils.zed_camera import ZedCamera

CUBE_SIZE_M = 0.025
STACK_HEIGHT_GOAL = 12
# Blend robust geometric center with OBB (same-size cubes: median + extent beats OBB alone)
OBB_CENTER_BLEND = 0.12


def orthonormalize_rotation(R: numpy.ndarray) -> numpy.ndarray:
    """Project 3x3 matrix onto SO(3)."""
    U, _, Vt = numpy.linalg.svd(R)
    Rn = U @ Vt
    if numpy.linalg.det(Rn) < 0:
        U[:, -1] *= -1.0
        Rn = U @ Vt
    return Rn


def points_to_meters_open3d(xyz):
    """ZED often returns mm; scale to meters for Open3D."""
    if xyz.size == 0:
        return xyz, 1.0
    max_abs = float(numpy.nanmax(numpy.abs(xyz)))
    scale = 0.001 if max_abs > 50.0 else 1.0
    return (xyz * scale).astype(numpy.float64), scale


def isolate_cube_cluster_open3d(pcd: o3d.geometry.PointCloud, num_cubes):
    """Same as checkpoint22: voxel, outliers, plane, DBSCAN, score, fallback."""
    if len(pcd.points) < 150:
        return None, "too few points after NaN filter"

    pcd = pcd.voxel_down_sample(voxel_size=0.003)
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=25, std_ratio=2.0)

    if len(pcd.points) < 100:
        return None, "too few points after voxel/outlier"

    plane_model, inliers = pcd.segment_plane(
        distance_threshold=0.010,
        ransac_n=3,
        num_iterations=1500,
    )
    if len(inliers) > 0 and len(inliers) > 0.12 * len(pcd.points):
        pcd = pcd.select_by_index(inliers, invert=True)

    if len(pcd.points) < 80:
        return None, "nothing left after plane removal"

    labels = numpy.asarray(
        pcd.cluster_dbscan(eps=0.020, min_points=25, print_progress=False)
    )
    max_label = int(labels.max()) if labels.size else -1
    if max_label < 0:
        return None, "DBSCAN found no clusters"

    def score_cluster(cluster, idx):
        obb = cluster.get_oriented_bounding_box()
        ext = numpy.sort(numpy.asarray(obb.extent))
        if ext[2] < 1e-9:
            return -1.0, None
        if obb.center[2] > 50:
            return -1.0, None
        max_dim = float(ext[2])
        min_dim = float(ext[0])
        compact = min_dim / max_dim if max_dim > 0 else 0.0
        size_ok = 0.008 <= max_dim <= 0.090
        if not size_ok:
            return float(idx.size) * 0.01, None
        qual = 1.0 if 0.25 < compact <= 1.0 else 0.3
        return float(idx.size) * compact * qual, cluster

    scored_clusters = []
    fallback_clusters = []

    for cid in range(max_label + 1):
        idx = numpy.where(labels == cid)[0]
        if idx.size < 30:
            continue

        cluster = pcd.select_by_index(idx)
        sc, chosen = score_cluster(cluster, idx)

        fallback_clusters.append((idx.size, cluster))

        if chosen is not None:
            scored_clusters.append((sc, chosen))

    scored_clusters.sort(key=lambda x: x[0], reverse=True)

    if len(scored_clusters) > 0:
        top_clusters = [c for (_, c) in scored_clusters[:num_cubes]]
        return top_clusters, f"{len(top_clusters)} cube-like clusters"

    fallback_clusters.sort(key=lambda x: x[0], reverse=True)

    if len(fallback_clusters) > 0:
        top_clusters = [c for (_, c) in fallback_clusters[:num_cubes] if len(c.points) >= 40]
        if len(top_clusters) > 0:
            return top_clusters, f"fallback: {len(top_clusters)} largest clusters"

    return None, "no cluster passed filters"


def cube_center_same_geometry(pts: numpy.ndarray, z_axis: numpy.ndarray, obb_center: numpy.ndarray) -> numpy.ndarray:
    """
    Estimate cube center using identical cube geometry: symmetric extent along ``z_axis``,
    robust in-plane median, optional light OBB blend.

    ``z_axis`` is unit **up** (robot +Z in camera frame). Points are one cube cluster in meters.
    """
    z_axis = z_axis / (numpy.linalg.norm(z_axis) + 1e-12)
    p0 = numpy.median(pts, axis=0)
    # Signed offsets along up from median reference
    s = numpy.dot(pts - p0[None, :], z_axis)
    s_lo = float(numpy.min(s))
    s_hi = float(numpy.max(s))
    s_mid = 0.5 * (s_lo + s_hi)

    # In-plane residual from p0 (perpendicular to z_axis)
    tang = pts - p0[None, :] - numpy.outer(s, z_axis)
    tang_med = numpy.median(tang, axis=0)

    geom = p0 + s_mid * z_axis + tang_med

    # Optional: if vertical span is wildly wrong, lean more on median (sensor junk)
    span = s_hi - s_lo
    if span > 1.8 * CUBE_SIZE_M or span < 0.4 * CUBE_SIZE_M:
        geom = 0.6 * geom + 0.4 * p0

    out = (1.0 - OBB_CENTER_BLEND) * geom + OBB_CENTER_BLEND * numpy.asarray(obb_center, dtype=numpy.float64)
    return out.astype(numpy.float64)


def get_cube_transform(cube_pcd, camera_pose):
    """
    Robust cube center + fixed-up orientation.

    Center: same-geometry estimate (mid-extent along robot-up + in-plane median), with a
    small OBB blend — better than OBB center alone for identical cubes.

    - Cube +Z is always **robot +Z** in the camera frame.
    - Cube +X from first OBB axis with nonzero horizontal projection.
    - Cube +Y = +Z × +X (right-handed).
    """
    if cube_pcd is None or len(cube_pcd.points) < 30:
        return None

    obb = cube_pcd.get_oriented_bounding_box()
    obb_center = numpy.asarray(obb.center, dtype=numpy.float64)
    pts = numpy.asarray(cube_pcd.points, dtype=numpy.float64)
    R_raw = numpy.asarray(obb.R, dtype=numpy.float64)

    R_cam_robot = camera_pose[:3, :3]
    z_axis = R_cam_robot @ numpy.array([0.0, 0.0, 1.0], dtype=numpy.float64)
    z_axis = z_axis / (numpy.linalg.norm(z_axis) + 1e-12)

    center = cube_center_same_geometry(pts, z_axis, obb_center)

    x_axis = None
    for k in range(3):
        x_raw = numpy.array(R_raw[:, k], dtype=numpy.float64)
        x_proj = x_raw - numpy.dot(x_raw, z_axis) * z_axis
        nrm = numpy.linalg.norm(x_proj)
        if nrm > 1e-6:
            x_axis = x_proj / nrm
            break

    if x_axis is None:
        tmp = numpy.array([1.0, 0.0, 0.0], dtype=numpy.float64)
        if abs(float(numpy.dot(tmp, z_axis))) > 0.9:
            tmp = numpy.array([0.0, 1.0, 0.0], dtype=numpy.float64)
        x_axis = tmp - numpy.dot(tmp, z_axis) * z_axis
        x_axis = x_axis / (numpy.linalg.norm(x_axis) + 1e-12)

    y_axis = numpy.cross(z_axis, x_axis)
    y_axis = y_axis / (numpy.linalg.norm(y_axis) + 1e-12)

    R = orthonormalize_rotation(numpy.column_stack([x_axis, y_axis, z_axis]))

    t_cam_cube = numpy.eye(4, dtype=numpy.float64)
    t_cam_cube[:3, :3] = R
    t_cam_cube[:3, 3] = center

    t_robot_cam = numpy.linalg.inv(camera_pose)
    t_robot_cube = t_robot_cam @ t_cam_cube

    return (t_robot_cube, t_cam_cube)


def get_cube_transforms(observation, camera_intrinsic, camera_pose):
    """
    Same pipeline as checkpoint22: finite points → meters → cluster → pose.

    Returns
    -------
    list or None
        List of ``(t_robot_cube, t_cam_cube)``, or ``None`` if perception fails.
    """
    del camera_intrinsic

    image, point_cloud = observation
    if image is None or point_cloud is None:
        return None

    xyz = point_cloud[..., :3]
    valid_mask = numpy.isfinite(xyz).all(axis=-1)
    valid_points = xyz[valid_mask]

    if valid_points.shape[0] < 100:
        return None

    valid_points_m, _scale = points_to_meters_open3d(valid_points)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(valid_points_m)

    cube_pcds, _msg = isolate_cube_cluster_open3d(pcd, num_cubes=STACK_HEIGHT_GOAL)
    if cube_pcds is None:
        return None

    transforms = []
    for pcd_c in cube_pcds:
        transforms.append(get_cube_transform(pcd_c, camera_pose))

    return transforms


def draw_status_overlay(image_bgra, lines, color=(0, 220, 0)):
    out = image_bgra.copy()
    y = 36
    for line in lines:
        cv2.putText(
            out,
            line[:120],
            (16, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            cv2.LINE_AA,
        )
        y += 28
    return out


def run_pure_vision_perception(cv_image, point_cloud, camera_intrinsic):
    """Calibration + cube transforms + overlay (checkpoint22-style)."""
    if cv_image is None:
        blank = numpy.zeros((720, 1280, 4), dtype=numpy.uint8)
        disp = draw_status_overlay(blank, ["ZED image is None"], (0, 0, 255))
        return None, disp

    t_cam_robot = get_transform_camera_robot(cv_image, camera_intrinsic)
    if t_cam_robot is None:
        disp = draw_status_overlay(
            cv_image,
            ["Calibration FAILED (checkpoint0 tags / PnP)"],
            (0, 0, 255),
        )
        return None, disp

    cube_transforms = get_cube_transforms(
        (cv_image, point_cloud), camera_intrinsic, t_cam_robot
    )
    if cube_transforms is None or len(cube_transforms) == 0:
        disp = draw_status_overlay(
            cv_image,
            ["orientation_RRC: no cube transforms"],
            (0, 0, 255),
        )
        return None, disp

    disp = cv_image.copy()
    for transform in cube_transforms:
        if transform is None:
            continue
        _tr, t_cam_cube = transform
        if t_cam_cube is not None and numpy.isfinite(t_cam_cube).all():
            draw_pose_axes(disp, camera_intrinsic, t_cam_cube)

    lines = [
        "orientation_RRC: same-geometry center + Z-up + OBB face X",
    ]
    disp = draw_status_overlay(disp, lines, (0, 220, 0))
    return cube_transforms, disp


def main():
    zed = ZedCamera()
    camera_intrinsic = zed.camera_intrinsic

    arm = XArmAPI(robot_ip)
    arm.connect()
    arm.motion_enable(enable=True)
    arm.set_tcp_offset([0, 0, GRIPPER_LENGTH, 0, 0, 0])
    arm.set_mode(0)
    arm.set_state(0)
    arm.move_gohome(wait=True)
    time.sleep(0.5)

    try:
        cv_image = zed.image
        point_cloud = zed.point_cloud

        cube_transforms, disp = run_pure_vision_perception(
            cv_image, point_cloud, camera_intrinsic
        )

        if cube_transforms is None:
            print("Perception failed.")
            cv2.namedWindow("orientation_RRC", cv2.WINDOW_NORMAL)
            cv2.imshow("orientation_RRC", disp)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            return

        t_robot_cube, _t_cam_cube = cube_transforms[0]

        cv2.namedWindow("Verifying Cube Pose (orientation_RRC)", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Verifying Cube Pose (orientation_RRC)", 1280, 720)
        cv2.imshow("Verifying Cube Pose (orientation_RRC)", disp)
        key = cv2.waitKey(0)

        if key == ord("k") and t_robot_cube is not None:
            cv2.destroyAllWindows()
            xyz = t_robot_cube[:3, 3]
            print(
                f"Cube in robot frame (m): x={xyz[0]:.3f}, y={xyz[1]:.3f}, z={xyz[2]:.3f}"
            )

            X_FIXED = 0.370
            Y_FIXED = 0.020

            t_robot_stack = t_robot_cube.copy()
            t_robot_stack[0, 3] = X_FIXED
            t_robot_stack[1, 3] = Y_FIXED

            for i in range(STACK_HEIGHT_GOAL):
                if i >= len(cube_transforms):
                    break
                pair = cube_transforms[i]
                if pair is None:
                    continue
                t_rc, _ = pair
                if t_rc is None:
                    continue
                grasp_cube(arm, t_rc)
                place_cube(arm, t_robot_stack)
                t_robot_stack[2, 3] += CUBE_SIZE_M

            arm.stop_lite6_gripper()
        else:
            cv2.destroyAllWindows()

    except Exception:
        traceback.print_exc()
    finally:
        arm.stop_lite6_gripper()
        arm.move_gohome(wait=True)
        time.sleep(0.5)
        arm.disconnect()
        zed.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
