"""
o2_RRC — challenge 2: **9 cubes** (one per size × color), **AprilTag** board calibration
(``checkpoint0``), nominal edges 22.5 / 25 / 30 mm, **stack plan: largest → smallest**
in three tiers (3× large, 3× medium, 3× small), with **left→right board color order**
from mean robot XY per hue class.

Vision: GPU depth mask, dense crops, precise nominal pose, HSV color id on image ROI,
multi-frame fusion. Tune ``POSE_SAMPLES``, ``REFINE_AABB_ITERS``, HSV bands in
``color_id_from_image_patch``.
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
    CUBE_SIZE_LARGE_M,
    CUBE_SIZE_MEDIUM_M,
    CUBE_SIZE_SMALL_M,
    isolate_cube_cluster_open3d,
    orthonormalize_rotation,
    points_to_meters_open3d,
    _in_plane_basis,
    _project_points_to_plane,
    _signed_plane_dist,
)
from utils.vis_utils import draw_pose_axes
from utils.zed_camera import ZedCamera

try:
    import torch
except Exception:
    torch = None

########################################################
# Fixed tower location in robot frame (meters) — tune for your arena / stage
TOWER_X_M = 0.370
TOWER_Y_M = 0.020

########################################################
# Motion / clearance (aligned with fast_RRC2)

GRASP_Z_OFFSET = 0.0001
LIFT_Z_DELTA = 0.04
# Slightly thicker offset so the TCP does not crash through the top cube
PLACE_Z_OFFSET = 0.003
TOOL_ROLL_DEG = 180.0
TOOL_PITCH_DEG = 0.0

SAFE_CLEARANCE_CUBE_HEIGHTS = 1.5
MIN_GRASP_ABOVE_CUBE_M = 0.038
SAFE_Z_MIN_M = 0.10
SAFE_Z_MAX_M = 0.36

ARM_SPEED_FAST = 2200
ARM_SPEED_APPROACH = 450
ARM_SPEED_PLACE_FINAL = 85
ARM_SPEED_LIFT = 1800
GRIPPER_SETTLE_GRASP_S = 0.35
GRIPPER_SETTLE_PLACE_S = 0.55
RELEASE_WAIT_S = 0.55
# Last millimeters of descent at ARM_SPEED_PLACE_FINAL (mm above nominal place height)
PLACE_SOFT_LAST_MM = 7.0

# Nominal cube edges (same as orientation_RRC); stack / clearance use largest for margins
CUBE_SIZES_M = (CUBE_SIZE_SMALL_M, CUBE_SIZE_MEDIUM_M, CUBE_SIZE_LARGE_M)
# Clamp raw estimate before snapping (meters), just outside 22.5–30 mm
EDGE_MIN_M = 0.021
EDGE_MAX_M = 0.032
REF_CUBE_HEIGHT_M = CUBE_SIZE_LARGE_M  # 30 mm — worst-case safe clearance

MAX_CLUSTERS = 14
# Multi-frame fusion — more frames = stabler pose (slower)
POSE_SAMPLES = 6
POSE_SAMPLE_DT_S = 0.08

# Dense geometry (accuracy over speed)
GPU_Z_MIN_M = 0.30
GPU_Z_MAX_M = 2.00
DENSE_CROP_MARGIN_M = 0.006
BOTTOM_LAYER_FRAC = 0.20
REFINE_AABB_ITERS = 18
REFINE_AABB_ITERS_POST_VERTICAL = 8

# Arena: 9 cubes = 3 sizes × 3 colors; stack bottom = largest tier, top = smallest
TOTAL_CUBES = 9
# Hue classification (OpenCV H 0–180); tune for your lighting / cube paints
HSV_RED_ORANGE = (0, 18)
HSV_RED_HIGH = (165, 180)
HSV_GREEN = (38, 95)
HSV_BLUE = (95, 135)


def snap_edge_to_nominal(edge_m: float) -> float:
    """Map a noisy edge estimate to the closest of 22.5 / 25 / 30 mm."""
    return min(CUBE_SIZES_M, key=lambda s: abs(float(s) - float(edge_m)))


# Stack plan: three tiers (large → medium → small); within each tier, left → right on the table
STACK_SIZE_ORDER = (CUBE_SIZE_LARGE_M, CUBE_SIZE_MEDIUM_M, CUBE_SIZE_SMALL_M)
# Match fused edge to plan step (nominal values; float tolerance)
EDGE_MATCH_TOL_M = 0.0008


def cam_center_to_pixel(t_cam: numpy.ndarray, K: numpy.ndarray) -> tuple[float, float] | None:
    """Project cube center (camera frame, column of ``t_cam``) to pixel coordinates."""
    c = numpy.asarray(t_cam[:3, 3], dtype=numpy.float64)
    z = float(c[2])
    if z <= 1e-7:
        return None
    K = numpy.asarray(K, dtype=numpy.float64)
    u = float(K[0, 0] * c[0] / z + K[0, 2])
    v = float(K[1, 1] * c[1] / z + K[1, 2])
    return u, v


def color_id_from_image_patch(
    image_bgra: numpy.ndarray,
    u: float,
    v: float,
    radius: int = 14,
) -> int:
    """
    Classify cube face color from BGRA image at projected center (HSV hue bands).
    Returns 0 = red/orange, 1 = green, 2 = blue; tune bands for your lighting.
    """
    if image_bgra is None or image_bgra.size == 0:
        return 0
    h, w = image_bgra.shape[:2]
    u0, v0 = int(round(u)), int(round(v))
    x0 = max(0, u0 - radius)
    x1 = min(w, u0 + radius + 1)
    y0 = max(0, v0 - radius)
    y1 = min(h, v0 + radius + 1)
    if y1 <= y0 or x1 <= x0:
        return 0
    patch = image_bgra[y0:y1, x0:x1, :3]
    if patch.size == 0:
        return 0
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    h_vals = hsv[:, :, 0].astype(numpy.float64).ravel()
    h_med = float(numpy.median(h_vals))
    lo_r, hi_r = HSV_RED_ORANGE
    lo_r2, hi_r2 = HSV_RED_HIGH
    lo_g, hi_g = HSV_GREEN
    lo_b, hi_b = HSV_BLUE
    if h_med <= hi_r or h_med >= lo_r2:
        return 0
    if lo_g <= h_med <= hi_g:
        return 1
    if lo_b <= h_med <= hi_b:
        return 2
    # Ambiguous: partition 0–180 into three bins
    return int(numpy.clip(round(h_med / 60.0), 0, 2))


def board_color_order_left_to_right(dets: list[dict]) -> list[int]:
    """
    Order the three hue classes by mean robot **X** (left → right). Each color id
    should appear equally often (3×) when all 9 cubes are visible.
    """
    sums = {0: numpy.zeros(3), 1: numpy.zeros(3), 2: numpy.zeros(3)}
    counts = {0: 0, 1: 0, 2: 0}
    for d in dets:
        cid = int(d.get("color_id", 0))
        cid = max(0, min(2, cid))
        sums[cid] += d["t_robot"][:3, 3]
        counts[cid] += 1
    present = [c for c in (0, 1, 2) if counts[c] > 0]
    if not present:
        return [0, 1, 2]
    means = {c: sums[c] / max(1, counts[c]) for c in present}
    ordered = sorted(present, key=lambda c: float(means[c][0]))
    # Ensure length 3 for plan: missing colors append in id order
    for c in (0, 1, 2):
        if c not in ordered:
            ordered.append(c)
    return ordered[:3]


def build_stack_plan(color_ids_left_to_right: list[int]) -> list[tuple[float, int]]:
    """Nine placements: for each size tier (large→small), left→right colors on the board."""
    out: list[tuple[float, int]] = []
    cols = color_ids_left_to_right[:3]
    for edge_m in STACK_SIZE_ORDER:
        for cid in cols:
            out.append((float(edge_m), int(cid)))
    return out


def pick_matching_cube(
    dets: list[dict],
    target_edge_m: float,
    target_color_id: int,
) -> dict | None:
    """Pick one detection matching nominal edge and color; prefer cube farther from tower XY (still on table)."""
    best: dict | None = None
    best_key: tuple[float, float] | None = None
    tower_xy = numpy.array([TOWER_X_M, TOWER_Y_M], dtype=numpy.float64)
    for d in dets:
        if int(d.get("color_id", -1)) != int(target_color_id):
            continue
        if abs(float(d["edge_m"]) - float(target_edge_m)) > EDGE_MATCH_TOL_M:
            continue
        xy = d["t_robot"][:2, 3]
        dist_tower = float(numpy.linalg.norm(xy - tower_xy))
        edge_err = abs(float(d["edge_m"]) - float(target_edge_m))
        # Prefer farther from tower (avoid already stacked cube); then tighter edge match
        key = (-dist_tower, edge_err)
        if best is None or key < best_key:
            best = d
            best_key = key
    return best


def pick_largest_fallback(dets: list[dict]) -> dict | None:
    if not dets:
        return None
    dets = sorted(dets, key=lambda d: -float(d["edge_m"]))
    return dets[0]


def flip_plane_to_robot_up(plane_model: numpy.ndarray, camera_pose: numpy.ndarray) -> numpy.ndarray:
    """Return plane coefficients with normal aligned to robot +Z in camera frame."""
    pm = numpy.asarray(plane_model, dtype=numpy.float64).copy()
    R_cam_robot = camera_pose[:3, :3]
    z_robot = R_cam_robot @ numpy.array([0.0, 0.0, 1.0], dtype=numpy.float64)
    z_robot /= numpy.linalg.norm(z_robot) + 1e-12
    if float(numpy.dot(pm[:3], z_robot)) < 0.0:
        pm = -pm
    return pm


def gpu_depth_mask(point_cloud) -> numpy.ndarray | None:
    """CUDA morphological depth mask when torch is available; else CPU threshold."""
    if point_cloud is None:
        return None
    z = point_cloud[..., 2]
    if torch is None:
        m = numpy.isfinite(z) & (z > GPU_Z_MIN_M) & (z < GPU_Z_MAX_M)
        return (m.astype(numpy.uint8) * 255)

    use_cuda = torch.cuda.is_available()
    device = "cuda" if use_cuda else "cpu"
    z_t = torch.as_tensor(z, device=device, dtype=torch.float32)
    finite = torch.isfinite(z_t)
    mask = finite & (z_t > GPU_Z_MIN_M) & (z_t < GPU_Z_MAX_M)
    x = mask.float().unsqueeze(0).unsqueeze(0)
    for _ in range(2):
        x = torch.nn.functional.max_pool2d(x, kernel_size=5, stride=1, padding=2)
        x = 1.0 - torch.nn.functional.max_pool2d(1.0 - x, kernel_size=5, stride=1, padding=2)
    for _ in range(1):
        x = 1.0 - torch.nn.functional.max_pool2d(1.0 - x, kernel_size=3, stride=1, padding=1)
        x = torch.nn.functional.max_pool2d(x, kernel_size=3, stride=1, padding=1)
    m = (x[0, 0] > 0.5).detach().cpu().numpy()
    return (m.astype(numpy.uint8) * 255)


def points_to_scene_meters(point_cloud) -> numpy.ndarray:
    """Finite points with optional GPU mask; convert to meters."""
    xyz = point_cloud[..., :3]
    valid = numpy.isfinite(xyz).all(axis=-1)
    mask = gpu_depth_mask(point_cloud)
    if mask is not None:
        valid = valid & (mask > 0)
    pts = xyz[valid]
    if pts.shape[0] < 150:
        pts = xyz[numpy.isfinite(xyz).all(axis=-1)]
    if pts.shape[0] < 100:
        return numpy.zeros((0, 3), dtype=numpy.float64)
    pts_m, _ = points_to_meters_open3d(pts)
    return pts_m.astype(numpy.float64)


def extract_dense_cluster_points(
    full_pts_m: numpy.ndarray,
    cluster: o3d.geometry.PointCloud,
    margin_m: float = DENSE_CROP_MARGIN_M,
) -> numpy.ndarray:
    """Crop full-resolution cloud to a loose AABB around the segmented cluster."""
    pts = numpy.asarray(cluster.points, dtype=numpy.float64)
    if pts.shape[0] == 0:
        return pts
    lo = numpy.min(pts, axis=0) - margin_m
    hi = numpy.max(pts, axis=0) + margin_m
    m = numpy.all((full_pts_m >= lo) & (full_pts_m <= hi), axis=1)
    dense = full_pts_m[m]
    if dense.shape[0] < 45:
        return pts
    return dense


def classify_nominal_edge(
    pts: numpy.ndarray,
    plane_model: numpy.ndarray,
    camera_pose: numpy.ndarray,
) -> float:
    """
    Combine multiple geometric cues, then snap to exactly 22.5 / 25 / 30 mm.
    """
    pm = flip_plane_to_robot_up(plane_model, camera_pose)
    h = _signed_plane_dist(pts, pm)
    span_v = float(numpy.max(h) - numpy.min(h))
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.astype(numpy.float64))
    obb = pcd.get_oriented_bounding_box()
    ext = numpy.sort(numpy.asarray(obb.extent))
    med_edge = float(numpy.median(ext))

    n = pm[:3] / (numpy.linalg.norm(pm[:3]) + 1e-12)
    e1, e2 = _in_plane_basis(n)
    p_plane = _project_points_to_plane(pts, pm)
    pref = numpy.median(p_plane, axis=0)
    rel = p_plane - pref[None, :]
    u = numpy.dot(rel, e1)
    v = numpy.dot(rel, e2)
    uv = numpy.stack([u, v], axis=1).astype(numpy.float32)
    rect_w, rect_h = 0.05, 0.05
    if uv.shape[0] >= 5:
        _rect = cv2.minAreaRect(uv)
        rect_w, rect_h = float(_rect[1][0]), float(_rect[1][1])
        if rect_w < rect_h:
            rect_w, rect_h = rect_h, rect_w
    foot = 0.5 * (rect_w + rect_h) if rect_w > 1e-6 else med_edge

    pooled = float(numpy.median(numpy.array([span_v, med_edge, foot], dtype=numpy.float64)))
    pooled = float(numpy.clip(pooled, EDGE_MIN_M, EDGE_MAX_M))
    return snap_edge_to_nominal(pooled)


def _refine_center_aabb(pts: numpy.ndarray, R: numpy.ndarray, c0: numpy.ndarray, iters: int) -> numpy.ndarray:
    """Center the point cloud in the cube frame (symmetric AABB)."""
    c = c0.copy()
    for _ in range(iters):
        q = R.T @ (pts - c[None, :]).T
        delta = numpy.array(
            [0.5 * (float(numpy.min(q[k])) + float(numpy.max(q[k]))) for k in range(3)],
            dtype=numpy.float64,
        )
        if float(numpy.linalg.norm(delta)) < 1e-8:
            break
        c = c + R @ delta
    return c


def _signed_height_point(c: numpy.ndarray, pm: numpy.ndarray) -> float:
    """Signed distance of point ``c`` to plane ax+by+cz+d=0."""
    return float(c[0] * pm[0] + c[1] * pm[1] + c[2] * pm[2] + pm[3])


def precise_cube_pose_nominal(
    pts: numpy.ndarray,
    plane_model: numpy.ndarray,
    camera_pose: numpy.ndarray,
    edge_m: float,
) -> tuple[numpy.ndarray, numpy.ndarray] | None:
    """
    Full geometry with **fixed** nominal edge ``edge_m``: table bottom + yaw from
    full footprint + AABB refinement + vertical anchor to ``h_bottom + edge/2``.
    """
    if pts.shape[0] < 35:
        return None

    pm = flip_plane_to_robot_up(plane_model, camera_pose)
    n = pm[:3] / (numpy.linalg.norm(pm[:3]) + 1e-12)
    e1, e2 = _in_plane_basis(n)

    h = _signed_plane_dist(pts, pm)
    h_min = float(numpy.min(h))
    h_max = float(numpy.max(h))
    if h_max - h_min < 1e-6:
        return None

    thresh = h_min + BOTTOM_LAYER_FRAC * (h_max - h_min)
    bottom_mask = h <= thresh
    if int(numpy.sum(bottom_mask)) < 10:
        bottom_mask = h <= numpy.percentile(h, 20.0)
    if int(numpy.sum(bottom_mask)) < 8:
        bottom_mask = numpy.ones(pts.shape[0], dtype=bool)

    p_plane_all = _project_points_to_plane(pts, pm)
    p_bot = p_plane_all[bottom_mask]
    c_bottom = numpy.median(p_bot, axis=0)
    center = c_bottom + (edge_m * 0.5) * n

    rel = p_plane_all - numpy.median(p_plane_all, axis=0)[None, :]
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

    c = _refine_center_aabb(pts, R, center, REFINE_AABB_ITERS)

    h_bottom = float(numpy.percentile(h, 6.0))
    h_target_center = h_bottom + edge_m * 0.5
    h_c = _signed_height_point(c, pm)
    c = c + n * (h_target_center - h_c)

    c = _refine_center_aabb(pts, R, c, REFINE_AABB_ITERS_POST_VERTICAL)

    t_cam_cube = numpy.eye(4, dtype=numpy.float64)
    t_cam_cube[:3, :3] = R
    t_cam_cube[:3, 3] = c

    t_robot_cam = numpy.linalg.inv(camera_pose)
    t_robot_cube = t_robot_cam @ t_cam_cube
    return t_robot_cube, t_cam_cube


def average_rotation_matrices(Rs: list[numpy.ndarray]) -> numpy.ndarray:
    """Mean rotation on SO(3) via quaternion average (sign-aligned)."""
    if not Rs:
        return numpy.eye(3)
    if len(Rs) == 1:
        return numpy.asarray(Rs[0], dtype=numpy.float64)
    mats = numpy.stack([orthonormalize_rotation(numpy.asarray(R, dtype=numpy.float64)) for R in Rs], axis=0)
    quats = Rotation.from_matrix(mats).as_quat()
    ref = quats[0].copy()
    aligned = numpy.empty_like(quats)
    aligned[0] = ref
    for i in range(1, quats.shape[0]):
        q = quats[i].copy()
        if numpy.dot(ref, q) < 0.0:
            q = -q
        aligned[i] = q
    q_mean = numpy.mean(aligned, axis=0)
    nrm = numpy.linalg.norm(q_mean)
    if nrm < 1e-12:
        return mats[0]
    q_mean /= nrm
    return Rotation.from_quat(q_mean).as_matrix()


def detect_all_cubes_adaptive(
    observation,
    camera_intrinsic,
    camera_pose,
    max_cubes: int = MAX_CLUSTERS,
) -> list[dict]:
    """
    Segment cubes; **dense** points per cluster; classify nominal edge; precise pose;
    HSV ``color_id`` (0/1/2) from image patch at projected cube center.
    """
    image, point_cloud = observation
    if image is None or point_cloud is None:
        return []

    K = numpy.asarray(camera_intrinsic, dtype=numpy.float64)

    full_pts_m = points_to_scene_meters(point_cloud)
    if full_pts_m.shape[0] < 120:
        return []

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(full_pts_m)

    cube_pcds, _msg, plane_np = isolate_cube_cluster_open3d(pcd, num_cubes=max_cubes)
    if not cube_pcds or plane_np is None:
        return []

    out: list[dict] = []
    for c in cube_pcds:
        pts = extract_dense_cluster_points(full_pts_m, c)
        if pts.shape[0] < 35:
            pts = numpy.asarray(c.points, dtype=numpy.float64)
        if pts.shape[0] < 30:
            continue
        edge_m = classify_nominal_edge(pts, plane_np, camera_pose)
        pose_pair = precise_cube_pose_nominal(pts, plane_np, camera_pose, edge_m)
        if pose_pair is None:
            continue
        t_robot, t_cam = pose_pair
        uv = cam_center_to_pixel(t_cam, K)
        if uv is not None:
            color_id = color_id_from_image_patch(image, uv[0], uv[1])
        else:
            color_id = 0
        out.append({"t_robot": t_robot, "t_cam": t_cam, "edge_m": edge_m, "color_id": int(color_id)})
    return out


def detect_all_cubes_adaptive_fused(
    observation,
    camera_intrinsic,
    camera_pose,
    n_samples: int = POSE_SAMPLES,
    dt_s: float = POSE_SAMPLE_DT_S,
    max_cubes: int = MAX_CLUSTERS,
) -> list[dict]:
    """Fuse frames: quaternion-mean rotation, median translation, snapped edge."""
    obs = observation
    runs: list[list[dict]] = []
    for _ in range(max(1, n_samples)):
        runs.append(detect_all_cubes_adaptive(obs, camera_intrinsic, camera_pose, max_cubes))
        if n_samples > 1:
            time.sleep(dt_s)
        obs = observation

    if not runs or not runs[0]:
        return []

    base = sorted(runs[0], key=lambda d: -d["edge_m"])
    fused: list[dict] = []
    for det in base:
        edges = [det["edge_m"]]
        colors = [int(det.get("color_id", 0))]
        Rs = [det["t_robot"][:3, :3]]
        positions = [det["t_robot"][:3, 3]]
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
                colors.append(int(best.get("color_id", 0)))
                Rs.append(best["t_robot"][:3, :3])
                positions.append(best["t_robot"][:3, 3])
        edge_m = snap_edge_to_nominal(float(numpy.median(edges)))
        uq, cnt = numpy.unique(colors, return_counts=True)
        color_id = int(uq[int(numpy.argmax(cnt))])
        Rm = average_rotation_matrices(Rs)
        pos = numpy.median(numpy.stack(positions, axis=0), axis=0)
        t_r = numpy.eye(4, dtype=numpy.float64)
        t_r[:3, :3] = Rm
        t_r[:3, 3] = pos
        t_cam = camera_pose @ t_r
        fused.append({"t_robot": t_r, "t_cam": t_cam, "edge_m": edge_m, "color_id": color_id})
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
    """
    Place at ``cube_pose`` XY/Z with a softer landing: approach fast, slow last
    millimeters, then dwell after opening the gripper.
    """
    xyz = cube_pose[:3, 3]
    x_mm, y_mm, z_mm = (xyz * 1000.0).tolist()
    z_c_m = float(xyz[2])
    safe_z_mm = compute_safe_clearance_z_mm(tower_top_z_m, z_c_m, cube_height_m)
    place_z_mm = z_mm + (PLACE_Z_OFFSET * 1000.0)
    lift_z_mm = max(safe_z_mm, place_z_mm + (LIFT_Z_DELTA * 1000.0))

    cube_r = Rotation.from_matrix(cube_pose[:3, :3])
    _, _, cube_yaw_deg = cube_r.as_euler("xyz", degrees=True)

    set_line(arm, x_mm, y_mm, safe_z_mm, cube_yaw_deg, ARM_SPEED_FAST)
    # Descend most of the way at moderate speed
    pre_touch_mm = place_z_mm - PLACE_SOFT_LAST_MM
    if pre_touch_mm > safe_z_mm + 0.5:
        set_line(arm, x_mm, y_mm, pre_touch_mm, cube_yaw_deg, ARM_SPEED_APPROACH)
    set_line(arm, x_mm, y_mm, place_z_mm, cube_yaw_deg, ARM_SPEED_PLACE_FINAL)
    arm.open_lite6_gripper()
    time.sleep(GRIPPER_SETTLE_PLACE_S)
    time.sleep(RELEASE_WAIT_S)
    set_line(arm, x_mm, y_mm, lift_z_mm, cube_yaw_deg, ARM_SPEED_LIFT)


def stack_target_pose(source_pose: numpy.ndarray, tower_x_m: float, tower_y_m: float, z_center_m: float) -> numpy.ndarray:
    """Build place pose: fixed tower XY, given stack center height; keep orientation from grasp."""
    out = numpy.copy(source_pose)
    out[0, 3] = float(tower_x_m)
    out[1, 3] = float(tower_y_m)
    out[2, 3] = float(z_center_m)
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
    max_cubes: int = TOTAL_CUBES,
    time_limit_s: float = 120.0,
    dry_run_preview: bool = True,
) -> int:
    """
    **Nine-cube** arena: AprilTag calibration, fused pose + color. Stack order is
    **large tier (3 colors, left→right on table)**, then **medium**, then **small**
    — biggest physical tier at the bottom of the tower, smallest on top.
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
        col_order = board_color_order_left_to_right(dets)
        plan = build_stack_plan(col_order)
        dets.sort(key=lambda d: -d["edge_m"])
        disp = cv_image.copy()
        for det in dets:
            draw_pose_axes(disp, camera_intrinsic, det["t_cam"])
        plan_str = " | ".join([f"{e * 1000:.0f}mm c{c}" for e, c in plan[:6]])
        plan_str2 = " | ".join([f"{e * 1000:.0f}mm c{c}" for e, c in plan[6:9]])
        lines = [
            f"o2_RRC: {len(dets)} cubes  colors L→R (id): {col_order}",
            f"Plan (9): {plan_str}",
            plan_str2,
            "Press k to run planned stack, any other key to abort",
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
    # z of table surface near the work area (initialized from first grasp)
    table_z_m: float | None = None
    # z of top face of the stack at TOWER_X_M, TOWER_Y_M (starts at table)
    tower_top_z_m: float | None = None
    stack_plan: list[tuple[float, int]] | None = None

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

        if stack_plan is None:
            stack_plan = build_stack_plan(board_color_order_left_to_right(dets))

        step = int(placed)
        if step >= len(stack_plan):
            break
        target_edge, target_color = stack_plan[step]
        pick = pick_matching_cube(dets, target_edge, target_color)
        if pick is None:
            print(
                f"Plan step {step + 1}/9: no match for edge={target_edge:.4f} color={target_color}; "
                "fallback: largest remaining."
            )
            pick = pick_largest_fallback(dets)
        if pick is None:
            break

        t_src = pick["t_robot"]
        h_m = float(pick["edge_m"])
        z_c = float(t_src[2, 3])

        if table_z_m is None:
            # Cube sitting on table: center ≈ table + h/2
            table_z_m = z_c - h_m / 2.0
            tower_top_z_m = float(table_z_m)

        # Next cube center sits half an edge above the current stack top surface
        desired_center_z = float(tower_top_z_m) + h_m / 2.0
        t_tgt = stack_target_pose(t_src, TOWER_X_M, TOWER_Y_M, desired_center_z)

        # Clearance uses stack top *before* this block is added
        tower_top_for_motion = float(tower_top_z_m)
        h_grasp = max(h_m, REF_CUBE_HEIGHT_M)

        try:
            grasp_cube(arm, t_src, tower_top_for_motion, h_grasp)
            place_cube(arm, t_tgt, tower_top_for_motion, h_m)
            arm.stop_lite6_gripper()
            tower_top_z_m = float(tower_top_z_m) + h_m
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
            max_cubes=TOTAL_CUBES,
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
