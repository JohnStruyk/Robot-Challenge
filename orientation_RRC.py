"""
orientation_RRC — checkpoint22 clustering + **physical cube** pose on a table.

Assumes each object is a **cube with one face flush on the table** (any yaw).

- Fits the **table plane** once (RANSAC) before removing it; cube **+Z** aligns
  with the table normal (flipped to match robot +Z in the camera frame).
- **Bottom face**: points closest to the table (lowest height percentiles) define
  the **contact footprint** — this reduces depth / “too far forward” bias from
  using the whole cloud or OBB center.
- **Center** = center of bottom face on the plane + ``(CUBE_SIZE_M/2) * n`` (challenge
  1 uses **25 mm** medium cubes; see ``CUBE_SIZE_SMALL_M`` / ``*_MEDIUM`` / ``*_LARGE``),
  then a small **bound-centering** step in the fitted cube frame so the point cloud is
  symmetric in ``[-s/2,s/2]^3``.
- **Yaw** from ``cv2.minAreaRect`` on the bottom-face footprint in the table plane.

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

# Nominal cube edge lengths (meters) — physical inventory
CUBE_SIZE_SMALL_M = 0.0225  # 22.5 mm
CUBE_SIZE_MEDIUM_M = 0.025  # 25 mm
CUBE_SIZE_LARGE_M = 0.030  # 30 mm
# Challenge 1 (standard tower) uses medium (25 mm) cubes
CUBE_SIZE_M = CUBE_SIZE_MEDIUM_M

STACK_HEIGHT_GOAL = 12
# Bottom-face layer: fraction of height range used to approximate the face on the table
BOTTOM_LAYER_FRAC = 0.22
# Refinement: snap translation so AABB in cube frame matches cloud
BOUND_CENTER_ITERS = 4


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
    """
    Same as checkpoint22: voxel, outliers, plane, DBSCAN, score, fallback.

    Also returns ``plane_model`` (4-vector ``ax+by+cz+d=0``) from the **first**
    table RANSAC — needed to place cube centers on the physical table plane.
    """
    if len(pcd.points) < 150:
        return None, "too few points after NaN filter", None

    pcd = pcd.voxel_down_sample(voxel_size=0.003)
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=25, std_ratio=2.0)

    if len(pcd.points) < 100:
        return None, "too few points after voxel/outlier", None

    plane_model, inliers = pcd.segment_plane(
        distance_threshold=0.010,
        ransac_n=3,
        num_iterations=1500,
    )
    plane_np = numpy.asarray(plane_model, dtype=numpy.float64).copy()
    if len(inliers) > 0 and len(inliers) > 0.12 * len(pcd.points):
        pcd = pcd.select_by_index(inliers, invert=True)

    if len(pcd.points) < 80:
        return None, "nothing left after plane removal", plane_np

    labels = numpy.asarray(
        pcd.cluster_dbscan(eps=0.020, min_points=25, print_progress=False)
    )
    max_label = int(labels.max()) if labels.size else -1
    if max_label < 0:
        return None, "DBSCAN found no clusters", plane_np

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
        return top_clusters, f"{len(top_clusters)} cube-like clusters", plane_np

    fallback_clusters.sort(key=lambda x: x[0], reverse=True)

    if len(fallback_clusters) > 0:
        top_clusters = [c for (_, c) in fallback_clusters[:num_cubes] if len(c.points) >= 40]
        if len(top_clusters) > 0:
            return top_clusters, f"fallback: {len(top_clusters)} largest clusters", plane_np

    return None, "no cluster passed filters", plane_np


def _signed_plane_dist(pts: numpy.ndarray, plane_model: numpy.ndarray) -> numpy.ndarray:
    """Signed distance to plane ``ax+by+cz+d=0`` (Open3D convention)."""
    a, b, c, d = plane_model[0], plane_model[1], plane_model[2], plane_model[3]
    return pts[:, 0] * a + pts[:, 1] * b + pts[:, 2] * c + d


def _project_points_to_plane(pts: numpy.ndarray, plane_model: numpy.ndarray) -> numpy.ndarray:
    """Orthogonal projection of points onto the plane."""
    n = numpy.asarray(plane_model[:3], dtype=numpy.float64)
    n = n / (numpy.linalg.norm(n) + 1e-12)
    dist = _signed_plane_dist(pts, plane_model)
    return pts - dist[:, None] * n[None, :]


def _in_plane_basis(n: numpy.ndarray) -> tuple[numpy.ndarray, numpy.ndarray]:
    """Orthonormal e1, e2 spanning the table plane."""
    tmp = numpy.array([1.0, 0.0, 0.0], dtype=numpy.float64)
    if abs(float(numpy.dot(tmp, n))) > 0.9:
        tmp = numpy.array([0.0, 1.0, 0.0], dtype=numpy.float64)
    e1 = numpy.cross(n, tmp)
    e1 = e1 / (numpy.linalg.norm(e1) + 1e-12)
    e2 = numpy.cross(n, e1)
    e2 = e2 / (numpy.linalg.norm(e2) + 1e-12)
    return e1, e2


def physical_cube_pose_from_points(
    pts: numpy.ndarray,
    plane_model: numpy.ndarray,
    camera_pose: numpy.ndarray,
    edge_m: float | None = None,
    *,
    bottom_layer_frac: float | None = None,
    bound_center_iters: int | None = None,
) -> tuple[numpy.ndarray, numpy.ndarray] | None:
    """
    Cube on table: bottom-face footprint + known edge length + min-area yaw + bound centering.

    Parameters
    ----------
    edge_m
        Physical cube edge (m). Pass as 4th positional or ``edge_m=``. If ``None``, uses
        ``CUBE_SIZE_M`` (25 mm challenge default).
    bottom_layer_frac
        Fraction of height range for bottom-face mask; default ``BOTTOM_LAYER_FRAC``.
        Smaller values (e.g. 0.14) tighten the contact slice for large cubes.
    bound_center_iters
        AABB snap iterations in cube frame; default ``BOUND_CENTER_ITERS``.

    Returns
    -------
    (t_robot_cube, t_cam_cube) or None.
    """
    if pts.shape[0] < 30:
        return None

    blf = float(BOTTOM_LAYER_FRAC if bottom_layer_frac is None else bottom_layer_frac)
    blf = float(numpy.clip(blf, 0.06, 0.45))
    bci = int(BOUND_CENTER_ITERS if bound_center_iters is None else bound_center_iters)
    bci = max(1, min(bci, 24))

    pm = numpy.asarray(plane_model, dtype=numpy.float64).copy()
    R_cam_robot = camera_pose[:3, :3]
    z_robot = R_cam_robot @ numpy.array([0.0, 0.0, 1.0], dtype=numpy.float64)
    z_robot = z_robot / (numpy.linalg.norm(z_robot) + 1e-12)
    if float(numpy.dot(pm[:3], z_robot)) < 0.0:
        pm = -pm

    n = pm[:3] / (numpy.linalg.norm(pm[:3]) + 1e-12)
    e1, e2 = _in_plane_basis(n)

    # Signed height above plane (consistent with flipped ``pm``)
    h = _signed_plane_dist(pts, pm)

    h_min = float(numpy.min(h))
    h_max = float(numpy.max(h))
    span = h_max - h_min
    if span < 1e-6:
        return None

    # Bottom face: points closest to the table (reduces forward / perspective bias)
    thresh = h_min + blf * span
    bottom_mask = h <= thresh
    if int(numpy.sum(bottom_mask)) < 8:
        thresh = numpy.percentile(h, 18.0)
        bottom_mask = h <= thresh
    if int(numpy.sum(bottom_mask)) < 5:
        bottom_mask = numpy.ones(pts.shape[0], dtype=bool)

    p_plane_all = _project_points_to_plane(pts, pm)
    p_bot = p_plane_all[bottom_mask]

    c_bottom = numpy.median(p_bot, axis=0)
    size_m = float(CUBE_SIZE_M if edge_m is None else edge_m)
    # Cube center: half edge along table normal from bottom face center
    center = c_bottom + (size_m * 0.5) * n

    # Yaw: min-area rectangle on bottom footprint in (u,v)
    rel = p_bot - c_bottom[None, :]
    u = numpy.dot(rel, e1)
    v = numpy.dot(rel, e2)
    uv = numpy.stack([u, v], axis=1).astype(numpy.float32)
    if uv.shape[0] >= 5:
        rect = cv2.minAreaRect(uv)
        box = cv2.boxPoints(rect).astype(numpy.float64)
        e01 = box[1] - box[0]
        norm_e = float(numpy.linalg.norm(e01))
        if norm_e > 1e-9:
            du, dv = e01[0] / norm_e, e01[1] / norm_e
            x_axis = du * e1 + dv * e2
        else:
            x_axis = e1
    else:
        x_axis = e1

    x_axis = x_axis - numpy.dot(x_axis, n) * n
    x_axis = x_axis / (numpy.linalg.norm(x_axis) + 1e-12)
    y_axis = numpy.cross(n, x_axis)
    y_axis = y_axis / (numpy.linalg.norm(y_axis) + 1e-12)

    R = orthonormalize_rotation(numpy.column_stack([x_axis, y_axis, n]))

    # Snap center so the cloud is centered in the cube frame (fixes systematic offset)
    c = center.copy()
    for _ in range(bci):
        q = R.T @ (pts - c[None, :]).T
        delta = numpy.array(
            [0.5 * (float(numpy.min(q[k])) + float(numpy.max(q[k]))) for k in range(3)],
            dtype=numpy.float64,
        )
        if float(numpy.linalg.norm(delta)) < 1e-7:
            break
        c = c + R @ delta

    t_cam_cube = numpy.eye(4, dtype=numpy.float64)
    t_cam_cube[:3, :3] = R
    t_cam_cube[:3, 3] = c

    t_robot_cam = numpy.linalg.inv(camera_pose)
    t_robot_cube = t_robot_cam @ t_cam_cube
    return (t_robot_cube, t_cam_cube)


def get_cube_transform(cube_pcd, camera_pose, plane_model: numpy.ndarray | None):
    """
    Physical cube on table using ``plane_model`` from scene RANSAC.

    If ``plane_model`` is missing, falls back to OBB-based pose (weaker).
    """
    if cube_pcd is None or len(cube_pcd.points) < 30:
        return None

    pts = numpy.asarray(cube_pcd.points, dtype=numpy.float64)

    if plane_model is not None and numpy.asarray(plane_model).shape[0] >= 4:
        out = physical_cube_pose_from_points(pts, numpy.asarray(plane_model, dtype=numpy.float64), camera_pose)
        if out is not None:
            return out

    # --- Fallback: no valid plane ---
    obb = cube_pcd.get_oriented_bounding_box()
    R_raw = numpy.asarray(obb.R, dtype=numpy.float64)
    R_cam_robot = camera_pose[:3, :3]
    z_axis = R_cam_robot @ numpy.array([0.0, 0.0, 1.0], dtype=numpy.float64)
    z_axis = z_axis / (numpy.linalg.norm(z_axis) + 1e-12)
    center = numpy.asarray(obb.center, dtype=numpy.float64)

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

    cube_pcds, _msg, plane_np = isolate_cube_cluster_open3d(pcd, num_cubes=STACK_HEIGHT_GOAL)
    if cube_pcds is None:
        return None

    transforms = []
    for pcd_c in cube_pcds:
        transforms.append(get_cube_transform(pcd_c, camera_pose, plane_np))

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
        "orientation_RRC: table plane + bottom face + min-rect yaw + bound snap",
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
    time.sleep(0.15)

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
        time.sleep(0.15)
        arm.disconnect()
        zed.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
