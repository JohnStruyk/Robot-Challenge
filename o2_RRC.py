"""
o2_RRC — irregular skyscraper (challenge 2) with **adaptive cube size** and
**largest-first** placement: each cycle picks the **biggest remaining** cube on
the table for the next stack level (big cubes tend to form the base).

Vision reuses ``orientation_RRC`` table plane + physical pose, with **per-cube
edge length** estimated from height span + OBB extents (clamped).

Run: ``python o2_RRC.py`` (working directory arbitrary; project root on ``sys.path``).
"""
from __future__ import annotations

import os
import sys
import time
import traceback

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import cv2
import numpy
import open3d as o3d
from scipy.spatial.transform import Rotation
from xarm.wrapper import XArmAPI

from checkpoint0 import get_transform_camera_robot
from checkpoint1 import GRIPPER_LENGTH, robot_ip
from orientation_RRC import (
    BOUND_CENTER_ITERS,
    BOTTOM_LAYER_FRAC,
    isolate_cube_cluster_open3d,
    orthonormalize_rotation,
    points_to_meters_open3d,
    _in_plane_basis,
    _project_points_to_plane,
    _signed_plane_dist,
)
from utils.vis_utils import draw_pose_axes
from utils.zed_camera import ZedCamera

########################################################
# Motion / clearance (aligned with fast_RRC2)

GRASP_Z_OFFSET = 0.0001
LIFT_Z_DELTA = 0.04
PLACE_Z_OFFSET = 0.002
TOOL_ROLL_DEG = 180.0
TOOL_PITCH_DEG = 0.0

SAFE_CLEARANCE_CUBE_HEIGHTS = 1.5
MIN_GRASP_ABOVE_CUBE_M = 0.038
SAFE_Z_MIN_M = 0.10
SAFE_Z_MAX_M = 0.36

ARM_SPEED_FAST = 2200
ARM_SPEED_PLACE = 180
ARM_SPEED_LIFT = 2000
GRIPPER_SETTLE_GRASP_S = 0.35
GRIPPER_SETTLE_PLACE_S = 0.45

# Adaptive edge bounds (meters) — physical cubes in arena
EDGE_MIN_M = 0.014
EDGE_MAX_M = 0.095
REF_CUBE_HEIGHT_M = 0.03

MAX_CLUSTERS = 14
POSE_SAMPLES = 2
POSE_SAMPLE_DT_S = 0.05


def flip_plane_to_robot_up(plane_model: numpy.ndarray, camera_pose: numpy.ndarray) -> numpy.ndarray:
    """Return plane coefficients with normal aligned to robot +Z in camera frame."""
    pm = numpy.asarray(plane_model, dtype=numpy.float64).copy()
    R_cam_robot = camera_pose[:3, :3]
    z_robot = R_cam_robot @ numpy.array([0.0, 0.0, 1.0], dtype=numpy.float64)
    z_robot /= numpy.linalg.norm(z_robot) + 1e-12
    if float(numpy.dot(pm[:3], z_robot)) < 0.0:
        pm = -pm
    return pm


def estimate_cube_edge_m(
    pts: numpy.ndarray,
    plane_model: numpy.ndarray,
    camera_pose: numpy.ndarray,
) -> float:
    """
    Adaptive edge length: blend vertical span (cube on table) with OBB median edge.
    """
    pm = flip_plane_to_robot_up(plane_model, camera_pose)
    h = _signed_plane_dist(pts, pm)
    span_v = float(numpy.max(h) - numpy.min(h))
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.astype(numpy.float64))
    obb = pcd.get_oriented_bounding_box()
    ext = numpy.sort(numpy.asarray(obb.extent))
    med_edge = float(numpy.median(ext))
    edge = 0.52 * span_v + 0.48 * med_edge
    return float(numpy.clip(edge, EDGE_MIN_M, EDGE_MAX_M))


def physical_cube_pose_with_edge(
    pts: numpy.ndarray,
    plane_model: numpy.ndarray,
    camera_pose: numpy.ndarray,
    edge_m: float,
) -> tuple[numpy.ndarray, numpy.ndarray] | None:
    """Same as orientation_RRC physical pose but uses measured ``edge_m`` for half-height."""
    if pts.shape[0] < 30:
        return None

    pm = flip_plane_to_robot_up(plane_model, camera_pose)
    n = pm[:3] / (numpy.linalg.norm(pm[:3]) + 1e-12)
    e1, e2 = _in_plane_basis(n)

    h = _signed_plane_dist(pts, pm)
    h_min = float(numpy.min(h))
    h_max = float(numpy.max(h))
    span = h_max - h_min
    if span < 1e-6:
        return None

    thresh = h_min + BOTTOM_LAYER_FRAC * span
    bottom_mask = h <= thresh
    if int(numpy.sum(bottom_mask)) < 8:
        thresh = numpy.percentile(h, 18.0)
        bottom_mask = h <= thresh
    if int(numpy.sum(bottom_mask)) < 5:
        bottom_mask = numpy.ones(pts.shape[0], dtype=bool)

    p_plane_all = _project_points_to_plane(pts, pm)
    p_bot = p_plane_all[bottom_mask]

    c_bottom = numpy.median(p_bot, axis=0)
    center = c_bottom + (edge_m * 0.5) * n

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

    c = center.copy()
    for _ in range(BOUND_CENTER_ITERS):
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
    return t_robot_cube, t_cam_cube


def detect_all_cubes_adaptive(
    observation,
    camera_intrinsic,
    camera_pose,
    max_cubes: int = MAX_CLUSTERS,
) -> list[dict]:
    """
    Segment all table cubes; estimate edge per cluster; return list of dicts.

    Each dict: ``t_robot``, ``t_cam``, ``edge_m`` (use as stack height increment).
    """
    image, point_cloud = observation
    if image is None or point_cloud is None:
        return []

    xyz = point_cloud[..., :3]
    valid_mask = numpy.isfinite(xyz).all(axis=-1)
    valid_points = xyz[valid_mask]
    if valid_points.shape[0] < 100:
        return []

    valid_points_m, _ = points_to_meters_open3d(valid_points)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(valid_points_m)

    cube_pcds, _msg, plane_np = isolate_cube_cluster_open3d(pcd, num_cubes=max_cubes)
    if not cube_pcds or plane_np is None:
        return []

    out: list[dict] = []
    for c in cube_pcds:
        pts = numpy.asarray(c.points, dtype=numpy.float64)
        if pts.shape[0] < 30:
            continue
        edge_m = estimate_cube_edge_m(pts, plane_np, camera_pose)
        pose_pair = physical_cube_pose_with_edge(pts, plane_np, camera_pose, edge_m)
        if pose_pair is None:
            continue
        t_robot, t_cam = pose_pair
        out.append(
            {
                "t_robot": t_robot,
                "t_cam": t_cam,
                "edge_m": edge_m,
            }
        )
    return out


def detect_all_cubes_adaptive_fused(
    observation,
    camera_intrinsic,
    camera_pose,
    n_samples: int = POSE_SAMPLES,
    dt_s: float = POSE_SAMPLE_DT_S,
    max_cubes: int = MAX_CLUSTERS,
) -> list[dict]:
    """Fuse a few frames: median edge per cluster index after sorting by edge."""
    obs = observation
    runs: list[list[dict]] = []
    for _ in range(max(1, n_samples)):
        runs.append(detect_all_cubes_adaptive(obs, camera_intrinsic, camera_pose, max_cubes))
        if n_samples > 1:
            time.sleep(dt_s)
        obs = observation

    if not runs or not runs[0]:
        return []

    # Match by greedy nearest-neighbor in XY (robot frame) across frames
    base = sorted(runs[0], key=lambda d: -d["edge_m"])
    fused = []
    for j, det in enumerate(base):
        edges = [det["edge_m"]]
        tr = [det["t_robot"]]
        xy0 = det["t_robot"][:2, 3]
        for run in runs[1:]:
            if not run:
                continue
            best = None
            best_d = 1e9
            for other in run:
                xy = other["t_robot"][:2, 3]
                dist = float(numpy.linalg.norm(xy - xy0))
                if dist < best_d:
                    best_d = dist
                    best = other
            if best is not None and best_d < 0.06:
                edges.append(best["edge_m"])
                tr.append(best["t_robot"])
        edge_m = float(numpy.median(edges))
        pos = numpy.median(numpy.stack([t[:3, 3] for t in tr], axis=0), axis=0)
        t_r = numpy.copy(tr[0])
        t_r[:3, 3] = pos
        t_cam = camera_pose @ t_r
        fused.append(
            {
                "t_robot": t_r,
                "t_cam": t_cam,
                "edge_m": edge_m,
            }
        )
    return fused


def compute_safe_clearance_z_mm(tower_top_z_m, cube_center_z_m, cube_height_m):
    ref_h = max(float(cube_height_m), REF_CUBE_HEIGHT_M)
    clearance = SAFE_CLEARANCE_CUBE_HEIGHTS * ref_h
    z_m = max(tower_top_z_m + clearance, float(cube_center_z_m) + MIN_GRASP_ABOVE_CUBE_M)
    z_m = float(numpy.clip(z_m, SAFE_Z_MIN_M, SAFE_Z_MAX_M))
    return z_m * 1000.0


def set_line(arm, x_mm, y_mm, z_mm, yaw_deg, speed):
    arm.set_position(
        x_mm,
        y_mm,
        z_mm,
        TOOL_ROLL_DEG,
        TOOL_PITCH_DEG,
        yaw_deg,
        speed=speed,
        is_radian=False,
        wait=True,
    )


def grasp_cube(arm, cube_pose, tower_top_z_m, cube_height_m):
    xyz = cube_pose[:3, 3]
    x_mm, y_mm, z_mm = (xyz * 1000.0).tolist()
    z_c_m = float(xyz[2])
    safe_z_mm = compute_safe_clearance_z_mm(tower_top_z_m, z_c_m, cube_height_m)
    grasp_z_mm = z_mm + (GRASP_Z_OFFSET * 1000.0)
    lift_z_mm = max(safe_z_mm, grasp_z_mm + (LIFT_Z_DELTA * 1000.0))

    cube_r = Rotation.from_matrix(cube_pose[:3, :3])
    _, _, cube_yaw_deg = cube_r.as_euler("xyz", degrees=True)

    arm.open_lite6_gripper()
    time.sleep(GRIPPER_SETTLE_GRASP_S)
    set_line(arm, x_mm, y_mm, safe_z_mm, cube_yaw_deg, ARM_SPEED_FAST)
    set_line(arm, x_mm, y_mm, grasp_z_mm, cube_yaw_deg, ARM_SPEED_FAST)
    arm.close_lite6_gripper()
    time.sleep(GRIPPER_SETTLE_GRASP_S)
    set_line(arm, x_mm, y_mm, lift_z_mm, cube_yaw_deg, ARM_SPEED_LIFT)


def place_cube(arm, cube_pose, tower_top_z_m, cube_height_m):
    xyz = cube_pose[:3, 3]
    x_mm, y_mm, z_mm = (xyz * 1000.0).tolist()
    z_c_m = float(xyz[2])
    safe_z_mm = compute_safe_clearance_z_mm(tower_top_z_m, z_c_m, cube_height_m)
    place_z_mm = z_mm + (PLACE_Z_OFFSET * 1000.0)
    lift_z_mm = max(safe_z_mm, place_z_mm + (LIFT_Z_DELTA * 1000.0))

    cube_r = Rotation.from_matrix(cube_pose[:3, :3])
    _, _, cube_yaw_deg = cube_r.as_euler("xyz", degrees=True)

    set_line(arm, x_mm, y_mm, safe_z_mm, cube_yaw_deg, ARM_SPEED_FAST)
    set_line(arm, x_mm, y_mm, place_z_mm, cube_yaw_deg, ARM_SPEED_PLACE)
    arm.open_lite6_gripper()
    time.sleep(GRIPPER_SETTLE_PLACE_S)
    set_line(arm, x_mm, y_mm, lift_z_mm, cube_yaw_deg, ARM_SPEED_LIFT)


def apply_delta_z_to_pose(t_pose, delta_z_m):
    out = numpy.copy(t_pose)
    out[2, 3] += delta_z_m
    return out


def draw_status_overlay(image_bgra, lines, color=(0, 220, 0)):
    out = image_bgra.copy()
    y = 36
    for line in lines:
        cv2.putText(
            out,
            line[:120],
            (16, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2,
            cv2.LINE_AA,
        )
        y += 26
    return out


def run_challenge_o2_irregular(
    arm,
    zed,
    *,
    max_cubes: int = 10,
    time_limit_s: float = 120.0,
    dry_run_preview: bool = True,
) -> int:
    """
    Irregular tower: each iteration grabs the **largest remaining** cube (by
    estimated edge). Stack top advances by that cube's ``edge_m``.
    """
    camera_intrinsic = zed.camera_intrinsic
    cv_image = zed.image
    point_cloud = zed.point_cloud

    if dry_run_preview:
        if cv_image is None:
            print("No image from ZED.")
            return 0
        t_cam_robot = get_transform_camera_robot(cv_image, camera_intrinsic)
        if t_cam_robot is None:
            print("Calibration failed.")
            return 0
        obs = (cv_image, point_cloud)
        dets = detect_all_cubes_adaptive_fused(obs, camera_intrinsic, t_cam_robot)
        if not dets:
            print("Preview: no cubes detected.")
            return 0
        dets.sort(key=lambda d: -d["edge_m"])
        disp = cv_image.copy()
        for det in dets:
            draw_pose_axes(disp, camera_intrinsic, det["t_cam"])
        lines = [
            f"o2_RRC: {len(dets)} cubes (largest edge ≈ {dets[0]['edge_m']:.3f} m)",
            "Press k to run largest-first stack, any other key to abort",
        ]
        disp = draw_status_overlay(disp, lines, (0, 220, 0))
        cv2.namedWindow("o2_RRC", cv2.WINDOW_NORMAL)
        cv2.imshow("o2_RRC", disp)
        key = cv2.waitKey(0)
        cv2.destroyAllWindows()
        if key != ord("k"):
            print("Aborted.")
            return 0

    placed = 0
    start = time.time()
    current_top_z: float | None = None

    for _ in range(max_cubes):
        if time.time() - start > time_limit_s:
            break

        t_cam_robot = get_transform_camera_robot(zed.image, camera_intrinsic)
        if t_cam_robot is None:
            print("Calibration failed.")
            break

        dets = detect_all_cubes_adaptive_fused(
            (zed.image, zed.point_cloud),
            camera_intrinsic,
            t_cam_robot,
        )
        if not dets:
            print("No cubes left or detect failed.")
            break

        # Largest estimated cube first (bottom of tower first in time)
        dets.sort(key=lambda d: -d["edge_m"])
        pick = dets[0]
        t_src = pick["t_robot"]
        h_m = float(pick["edge_m"])
        z_c = float(t_src[2, 3])

        if current_top_z is None:
            current_top_z = z_c + h_m / 2.0

        desired_center_z = current_top_z + h_m / 2.0
        delta_z = desired_center_z - z_c
        t_tgt = apply_delta_z_to_pose(t_src, delta_z)

        tower_top_z_m = float(current_top_z)
        h_grasp = max(h_m, REF_CUBE_HEIGHT_M)

        try:
            grasp_cube(arm, t_src, tower_top_z_m, h_grasp)
            place_cube(arm, t_tgt, tower_top_z_m, h_m)
            arm.stop_lite6_gripper()
            current_top_z += h_m
            placed += 1
        except Exception as exc:
            traceback.print_exc()
            print("Motion error:", exc)
            break

    time.sleep(0.5)
    return placed


def main():
    zed = ZedCamera()
    arm = XArmAPI(robot_ip)
    arm.connect()
    arm.motion_enable(enable=True)
    arm.set_tcp_offset([0, 0, GRIPPER_LENGTH, 0, 0, 0])
    arm.set_mode(0)
    arm.set_state(0)
    arm.move_gohome(wait=True)
    time.sleep(0.5)

    try:
        n = run_challenge_o2_irregular(
            arm,
            zed,
            max_cubes=10,
            time_limit_s=120.0,
            dry_run_preview=True,
        )
        print(f"o2_RRC: placed {n} cube(s).")
    finally:
        arm.stop_lite6_gripper()
        arm.move_gohome(wait=True)
        time.sleep(0.5)
        arm.disconnect()
        zed.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
