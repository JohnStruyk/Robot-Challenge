"""
o2_RRC — challenge 2 (minimal): AprilTag pose (checkpoint0), segment cubes, snap edge
to 22.5 / 25 / 30 mm, stack at fixed tower XY. Cube orientation uses a **black border**
hint (dark pixels on the top face) plus PCA / min-area rect fallbacks on bottom-layer points.

**Route**: After the first good detection pass, cubes are sorted **largest → smallest**
(nominal edge, then X). That order is fixed for the run. Each pick matches the **next**
planned center to the **nearest** current detection (no color, no re-planning tiers).

**Grasp**: Yaw snapped to 90° for parallel jaws; simple approach (safe Z → grasp Z).
"""
from __future__ import annotations

import os
import sys
import time
import traceback

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

os.environ.setdefault("QT_LOGGING_RULES", "*.debug=false;qt.qpa.*=false")

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

########################################################
TOWER_X_M = 0.370
TOWER_Y_M = 0.020

GRASP_Z_OFFSET = 0.0001
LIFT_Z_DELTA = 0.04
PLACE_Z_OFFSET = 0.003
TOOL_ROLL_DEG = 180.0
TOOL_PITCH_DEG = 0.0

SAFE_CLEARANCE_CUBE_HEIGHTS = 1.5
MIN_GRASP_ABOVE_CUBE_M = 0.045
SAFE_Z_MIN_M = 0.10
SAFE_Z_MAX_M = 0.36

ARM_SPEED_TRAVEL = 1200
ARM_SPEED_APPROACH = 320
ARM_SPEED_PLACE_FINAL = 70
ARM_SPEED_LIFT = 1000
ARM_SPEED_GRASP_DESCEND = 250
GRIPPER_SETTLE_GRASP_S = 0.25
GRIPPER_SETTLE_PLACE_S = 0.45
RELEASE_WAIT_S = 0.45
PLACE_SOFT_LAST_MM = 7.0
GRASP_YAW_OFFSET_DEG = 0.0

CUBE_SIZES_M = (CUBE_SIZE_SMALL_M, CUBE_SIZE_MEDIUM_M, CUBE_SIZE_LARGE_M)
EDGE_MIN_M = 0.021
EDGE_MAX_M = 0.032
REF_CUBE_HEIGHT_M = CUBE_SIZE_LARGE_M

MAX_CLUSTERS = 9
DEDUPE_MIN_SEP_M = 0.018
TOTAL_CUBES = 9

GPU_Z_MIN_M = 0.28
GPU_Z_MAX_M = 1.4
DENSE_CROP_MARGIN_M = 0.006
BOTTOM_LAYER_FRAC = 0.18
REFINE_AABB_ITERS = 18
REFINE_AABB_ITERS_POST_VERTICAL = 8

# Black cube rim in the left camera image (BGR): use dark samples to pick top-face edge direction
BLACK_BORDER_GRAY_MAX = 100
BLACK_BORDER_MIN_POINTS = 14

# Match live cube to planned step (meters)
MATCH_MAX_DIST_M = 0.055
MATCH_EDGE_TOL_M = 0.002


def snap_edge_to_nominal(edge_m: float) -> float:
    return min(CUBE_SIZES_M, key=lambda s: abs(float(s) - float(edge_m)))


def flip_plane_to_robot_up(plane_model: numpy.ndarray, camera_pose: numpy.ndarray) -> numpy.ndarray:
    pm = numpy.asarray(plane_model, dtype=numpy.float64).copy()
    R_cam_robot = camera_pose[:3, :3]
    z_robot = R_cam_robot @ numpy.array([0.0, 0.0, 1.0], dtype=numpy.float64)
    z_robot /= numpy.linalg.norm(z_robot) + 1e-12
    if float(numpy.dot(pm[:3], z_robot)) < 0.0:
        pm = -pm
    return pm


def gpu_depth_mask(point_cloud) -> numpy.ndarray | None:
    if point_cloud is None:
        return None
    z = point_cloud[..., 2]
    m = numpy.isfinite(z) & (z > GPU_Z_MIN_M) & (z < GPU_Z_MAX_M)
    return (m.astype(numpy.uint8) * 255)


def points_to_scene_meters(point_cloud) -> numpy.ndarray:
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
    return float(c[0] * pm[0] + c[1] * pm[1] + c[2] * pm[2] + pm[3])


def _bgr_gray_at_cam_points(
    image_bgra: numpy.ndarray,
    pts_cam: numpy.ndarray,
    K: numpy.ndarray,
) -> numpy.ndarray:
    """Approx luminance (0–255) at ZED pixels for 3D points in the **camera** frame (meters)."""
    h, w = image_bgra.shape[:2]
    K = numpy.asarray(K, dtype=numpy.float64)
    n = pts_cam.shape[0]
    g = numpy.full(n, 255.0, dtype=numpy.float64)
    z = numpy.maximum(pts_cam[:, 2], 1e-9)
    u = numpy.round(K[0, 0] * pts_cam[:, 0] / z + K[0, 2]).astype(numpy.int32)
    v = numpy.round(K[1, 1] * pts_cam[:, 1] / z + K[1, 2]).astype(numpy.int32)
    m = (pts_cam[:, 2] > 1e-7) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
    if numpy.any(m):
        bgr = image_bgra[v[m], u[m], :3].astype(numpy.float64)
        g[m] = 0.114 * bgr[:, 0] + 0.587 * bgr[:, 1] + 0.299 * bgr[:, 2]
    return g


def _x_axis_from_plane_uv(uv: numpy.ndarray) -> numpy.ndarray | None:
    """Unit direction in plane (coefficients on e1,e2) from min-area rect of 2D points."""
    if uv.shape[0] < 5:
        return None
    rect = cv2.minAreaRect(uv.astype(numpy.float32))
    box = cv2.boxPoints(rect).astype(numpy.float64)
    e01 = box[1] - box[0]
    e12 = box[2] - box[1]
    n1 = float(numpy.linalg.norm(e01))
    n2 = float(numpy.linalg.norm(e12))
    if n1 < 1e-9 and n2 < 1e-9:
        return None
    if n1 >= n2:
        du, dv = e01[0] / n1, e01[1] / n1
    else:
        du, dv = e12[0] / n2, e12[1] / n2
    d = numpy.array([du, dv], dtype=numpy.float64)
    d /= numpy.linalg.norm(d) + 1e-12
    return d


def _x_axis_from_bottom_pca(u: numpy.ndarray, v: numpy.ndarray) -> numpy.ndarray | None:
    """Major axis of bottom-layer footprint in plane coordinates (u·e1, u·e2)."""
    n = u.size
    if n < 8:
        return None
    uv = numpy.stack([u, v], axis=1).astype(numpy.float64)
    uv -= numpy.mean(uv, axis=0, keepdims=True)
    c = (uv.T @ uv) / max(1, n - 1)
    _, vecs = numpy.linalg.eigh(c)
    major = vecs[:, -1]
    major /= numpy.linalg.norm(major) + 1e-12
    return major


def _choose_x_axis_bottom_face(
    pts_cam: numpy.ndarray,
    bottom_mask: numpy.ndarray,
    p_plane_all: numpy.ndarray,
    e1: numpy.ndarray,
    e2: numpy.ndarray,
    n: numpy.ndarray,
    image_bgra: numpy.ndarray | None,
    K: numpy.ndarray | None,
) -> numpy.ndarray:
    """
    In-plane x axis (3D, unit, orthogonal to n). Prefers **black border** samples
    in the image; else PCA on full bottom layer; else min-area rect on full bottom UV.
    """
    pref = numpy.median(p_plane_all[bottom_mask], axis=0)
    rel_bot = p_plane_all[bottom_mask] - pref[None, :]
    u = numpy.dot(rel_bot, e1)
    v = numpy.dot(rel_bot, e2)
    uv_bot = numpy.stack([u, v], axis=1)

    du, dv = 1.0, 0.0
    got = False

    if image_bgra is not None and K is not None:
        gray = _bgr_gray_at_cam_points(image_bgra, pts_cam, K)
        dark = bottom_mask & (gray <= float(BLACK_BORDER_GRAY_MAX))
        if int(numpy.sum(dark)) >= BLACK_BORDER_MIN_POINTS:
            rel_d = p_plane_all[dark] - numpy.median(p_plane_all[dark], axis=0)[None, :]
            ud = numpy.dot(rel_d, e1)
            vd = numpy.dot(rel_d, e2)
            uvd = numpy.stack([ud, vd], axis=1)
            xd = _x_axis_from_plane_uv(uvd)
            if xd is not None:
                du, dv = float(xd[0]), float(xd[1])
                got = True

    if not got:
        x2 = _x_axis_from_bottom_pca(u, v)
        if x2 is not None:
            du, dv = float(x2[0]), float(x2[1])
            got = True
    if not got:
        xf = _x_axis_from_plane_uv(uv_bot)
        if xf is not None:
            du, dv = float(xf[0]), float(xf[1])

    x_axis = du * e1 + dv * e2
    x_axis = x_axis - numpy.dot(x_axis, n) * n
    x_axis = x_axis / (numpy.linalg.norm(x_axis) + 1e-12)
    return x_axis


def precise_cube_pose_nominal(
    pts: numpy.ndarray,
    plane_model: numpy.ndarray,
    camera_pose: numpy.ndarray,
    edge_m: float,
    image_bgra: numpy.ndarray | None = None,
    K: numpy.ndarray | None = None,
) -> tuple[numpy.ndarray, numpy.ndarray] | None:
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

    x_axis = _choose_x_axis_bottom_face(
        pts, bottom_mask, p_plane_all, e1, e2, n, image_bgra, K
    )
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


def dedupe_detections(dets: list[dict]) -> list[dict]:
    if len(dets) <= 1:
        return dets
    ordered = sorted(dets, key=lambda d: (-float(d["edge_m"]), -float(d["t_robot"][0, 3])))
    kept: list[dict] = []
    for d in ordered:
        p = numpy.asarray(d["t_robot"][:3, 3], dtype=numpy.float64)
        if any(float(numpy.linalg.norm(p - numpy.asarray(k["t_robot"][:3, 3]))) < DEDUPE_MIN_SEP_M for k in kept):
            continue
        kept.append(d)
    return kept


def detect_cubes_once(
    observation,
    camera_pose,
    camera_intrinsic: numpy.ndarray | None = None,
    max_cubes: int = MAX_CLUSTERS,
) -> list[dict]:
    image, point_cloud = observation
    if image is None or point_cloud is None:
        return []

    K = numpy.asarray(camera_intrinsic, dtype=numpy.float64) if camera_intrinsic is not None else None

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
        pose_pair = precise_cube_pose_nominal(
            pts,
            plane_np,
            camera_pose,
            edge_m,
            image_bgra=image,
            K=K,
        )
        if pose_pair is None:
            continue
        t_robot, t_cam = pose_pair
        out.append({"t_robot": t_robot, "t_cam": t_cam, "edge_m": float(edge_m)})
    return dedupe_detections(out)


def preplan_route(dets: list[dict]) -> list[tuple[numpy.ndarray, float]]:
    """
    Fixed pick order: **largest nominal edge first**, then more -X (table left) as tie-break.
    Returns list of (center_xyz, edge_m) length <= 9.
    """
    if not dets:
        return []
    ranked = sorted(
        dets,
        key=lambda d: (-float(d["edge_m"]), -float(d["t_robot"][0, 3])),
    )
    ranked = ranked[:TOTAL_CUBES]
    return [(numpy.asarray(d["t_robot"][:3, 3], dtype=numpy.float64).copy(), float(d["edge_m"])) for d in ranked]


def match_step_to_detection(
    dets: list[dict],
    planned_center: numpy.ndarray,
    planned_edge: float,
) -> dict | None:
    """Nearest current cube to this plan step; edge must be roughly consistent."""
    best: dict | None = None
    best_d = 1e9
    for d in dets:
        if abs(float(d["edge_m"]) - float(planned_edge)) > MATCH_EDGE_TOL_M:
            continue
        p = numpy.asarray(d["t_robot"][:3, 3], dtype=numpy.float64)
        dist = float(numpy.linalg.norm(p - planned_center))
        if dist < best_d:
            best_d = dist
            best = d
    if best is None:
        return None
    if best_d > MATCH_MAX_DIST_M:
        return None
    return best


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


def snap_yaw_to_face_deg(yaw_deg: float) -> float:
    y = float(yaw_deg)
    y = ((y + 180.0) % 360.0) - 180.0
    return round(y / 90.0) * 90.0


def grasp_cube(arm, cube_pose, tower_top_z_m, cube_height_m):
    xyz = cube_pose[:3, 3]
    x_mm, y_mm, z_mm = (xyz * 1000.0).tolist()
    z_c_m = float(xyz[2])
    safe_z_mm = compute_safe_clearance_z_mm(tower_top_z_m, z_c_m, cube_height_m)
    grasp_z_mm = z_mm + (GRASP_Z_OFFSET * 1000.0)
    lift_z_mm = max(safe_z_mm, grasp_z_mm + (LIFT_Z_DELTA * 1000.0))

    cube_r = Rotation.from_matrix(cube_pose[:3, :3])
    _, _, cube_yaw_deg = cube_r.as_euler("xyz", degrees=True)
    yaw_use = snap_yaw_to_face_deg(cube_yaw_deg) + GRASP_YAW_OFFSET_DEG

    arm.open_lite6_gripper()
    time.sleep(GRIPPER_SETTLE_GRASP_S)
    set_line(arm, x_mm, y_mm, safe_z_mm, yaw_use, ARM_SPEED_TRAVEL)
    set_line(arm, x_mm, y_mm, grasp_z_mm, yaw_use, ARM_SPEED_GRASP_DESCEND)
    arm.close_lite6_gripper()
    time.sleep(GRIPPER_SETTLE_GRASP_S)
    set_line(arm, x_mm, y_mm, lift_z_mm, yaw_use, ARM_SPEED_LIFT)


def place_cube(arm, cube_pose, tower_top_z_m, cube_height_m):
    xyz = cube_pose[:3, 3]
    x_mm, y_mm, z_mm = (xyz * 1000.0).tolist()
    z_c_m = float(xyz[2])
    safe_z_mm = compute_safe_clearance_z_mm(tower_top_z_m, z_c_m, cube_height_m)
    place_z_mm = z_mm + (PLACE_Z_OFFSET * 1000.0)
    lift_z_mm = max(safe_z_mm, place_z_mm + (LIFT_Z_DELTA * 1000.0))

    cube_r = Rotation.from_matrix(cube_pose[:3, :3])
    _, _, cube_yaw_deg = cube_r.as_euler("xyz", degrees=True)
    yaw_place = snap_yaw_to_face_deg(cube_yaw_deg) + GRASP_YAW_OFFSET_DEG

    set_line(arm, x_mm, y_mm, safe_z_mm, yaw_place, ARM_SPEED_TRAVEL)
    pre_touch_mm = place_z_mm - PLACE_SOFT_LAST_MM
    if pre_touch_mm > safe_z_mm + 0.5:
        set_line(arm, x_mm, y_mm, pre_touch_mm, yaw_place, ARM_SPEED_APPROACH)
    set_line(arm, x_mm, y_mm, place_z_mm, yaw_place, ARM_SPEED_PLACE_FINAL)
    arm.open_lite6_gripper()
    time.sleep(GRIPPER_SETTLE_PLACE_S)
    time.sleep(RELEASE_WAIT_S)
    set_line(arm, x_mm, y_mm, lift_z_mm, yaw_place, ARM_SPEED_LIFT)


def stack_target_pose(source_pose: numpy.ndarray, tower_x_m: float, tower_y_m: float, z_center_m: float) -> numpy.ndarray:
    out = numpy.copy(source_pose)
    out[0, 3] = float(tower_x_m)
    out[1, 3] = float(tower_y_m)
    out[2, 3] = float(z_center_m)
    return out


def draw_status_overlay(image_bgra, lines, color=(0, 220, 0)):
    out = image_bgra.copy()
    y = 36
    for line in lines:
        safe = "".join(c if 32 <= ord(c) < 127 else "?" for c in str(line)[:120])
        try:
            cv2.putText(out, safe, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_8)
        except Exception:
            pass
        y += 26
    return out


def run_challenge_o2(
    arm,
    zed,
    *,
    max_cubes: int = TOTAL_CUBES,
    time_limit_s: float = 120.0,
    dry_run_preview: bool = True,
) -> int:
    camera_intrinsic = zed.camera_intrinsic
    route: list[tuple[numpy.ndarray, float]] | None = None

    if dry_run_preview:
        img = zed.image
        if img is None:
            print("No image.")
            return 0
        T = get_transform_camera_robot(img, camera_intrinsic)
        if T is None:
            print("AprilTag calibration failed.")
            return 0
        d0 = detect_cubes_once((img, zed.point_cloud), T, camera_intrinsic)
        if len(d0) < 1:
            print("No cubes detected.")
            return 0
        route = preplan_route(d0)
        disp = img.copy()
        for d in d0:
            draw_pose_axes(disp, camera_intrinsic, d["t_cam"])
        lines = [
            f"cubes={len(d0)}  route order: largest->smallest ({len(route)} steps)",
            "press k to run, other key abort",
        ]
        cv2.namedWindow("o2_RRC", cv2.WINDOW_NORMAL)
        cv2.imshow("o2_RRC", draw_status_overlay(disp, lines))
        if cv2.waitKey(0) != ord("k"):
            cv2.destroyAllWindows()
            print("Aborted.")
            return 0
        cv2.destroyAllWindows()

    placed = 0
    t0 = time.time()
    table_z_m: float | None = None
    tower_top_z_m: float | None = None

    for _ in range(max_cubes):
        if time.time() - t0 > time_limit_s:
            break
        T = get_transform_camera_robot(zed.image, camera_intrinsic)
        if T is None:
            print("Calibration failed.")
            break

        dets = detect_cubes_once((zed.image, zed.point_cloud), T, camera_intrinsic)
        if route is None:
            route = preplan_route(dets)
            if not route:
                print("No cubes to plan.")
                break
            print(f"Planned route: {len(route)} picks (largest -> smallest).")

        if placed >= len(route):
            break
        if not dets:
            print("Lost all detections.")
            break

        pc, pe = route[placed]
        pick = match_step_to_detection(dets, pc, pe)
        if pick is None:
            pick = sorted(dets, key=lambda d: -float(d["edge_m"]))[0]
            print(f"Step {placed + 1}: loose match; taking largest visible cube.")

        h_m = float(pick["edge_m"])
        t_src = pick["t_robot"]
        z_c = float(t_src[2, 3])

        if table_z_m is None:
            table_z_m = z_c - h_m / 2.0
            tower_top_z_m = float(table_z_m)

        desired_center_z = float(tower_top_z_m) + h_m / 2.0
        t_tgt = stack_target_pose(t_src, TOWER_X_M, TOWER_Y_M, desired_center_z)
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

    time.sleep(0.3)
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
    time.sleep(0.4)

    try:
        n = run_challenge_o2(arm, zed, max_cubes=TOTAL_CUBES, dry_run_preview=True)
        print(f"o2_RRC: placed {n} cube(s).")
    finally:
        arm.stop_lite6_gripper()
        arm.move_gohome(wait=True)
        time.sleep(0.4)
        arm.disconnect()
        zed.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
