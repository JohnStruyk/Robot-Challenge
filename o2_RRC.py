"""
o2_RRC — challenge 2 (minimal): AprilTag (checkpoint0), cluster cubes, snap edge to
22.5 / 25 / 30 mm.

**Cube pose** — ``orientation_RRC.physical_cube_pose_from_points`` (RANSAC table plane,
bottom footprint, min-area yaw, bound centering), then extra AABB passes and **height snap**
along the table normal. **30 mm (large) cubes**: thinner bottom-layer mask, more bound
iterations, **4th-percentile** contact height (vs raw min), wider dense crop, and **two**
refine↔snap cycles. Same AprilTag / world frame as PnP.

**Play area**: Dense points and detections are cropped to the **white mat** in world
frame — the axis-aligned rectangle spanned by the **four** AprilTag centers in
``checkpoint0.TAG_CENTER_COORDINATES`` (PnP can use 3+ tags; the spatial gate always
uses this full rectangle so nothing off-mat is clustered). Small XY margin inset
avoids the tag markers themselves.

**Route**: Largest → smallest (nominal edge, then X); each step matches the planned
center. **Grasp**: yaw snapped to 90°.
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

from checkpoint0 import TAG_CENTER_COORDINATES, get_transform_camera_robot
from checkpoint1 import GRIPPER_LENGTH, robot_ip
from orientation_RRC import (
    CUBE_SIZE_LARGE_M,
    CUBE_SIZE_MEDIUM_M,
    CUBE_SIZE_SMALL_M,
    isolate_cube_cluster_open3d,
    physical_cube_pose_from_points,
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
DENSE_CROP_MARGIN_M = 0.008
# Larger clusters (~30 mm cubes) need a wider dense crop so sides/top contribute to pose.
LARGE_DENSE_CROP_MARGIN_M = 0.012
LARGE_CLUSTER_MIN_EXTENT_M = 0.027
# Extra AABB-in-cube-frame passes after physical_cube_pose_from_points (orientation already uses BOUND_CENTER_ITERS).
POSE_BOUND_CENTER_EXTRA_ITERS = 4
POSE_BOUND_CENTER_EXTRA_ITERS_LARGE = 8
# Large cubes: thinner bottom slice + more bound iterations inside physical_cube_pose_from_points.
LARGE_BOTTOM_LAYER_FRAC = 0.14
LARGE_PHYSICAL_BOUND_ITERS = 6
# Signed-height contact: use low percentile instead of min to ignore depth speckle (large cubes).
LARGE_H_CONTACT_PERCENTILE = 4.0
# Alternate refine ↔ height snap for large cubes (tighter convergence).
LARGE_POSE_REFINE_SNAP_CYCLES = 2

# White play mat = axis-aligned rectangle through the four AprilTag centers (world XY, meters).
_TAG_CENTERS_XY = numpy.asarray(TAG_CENTER_COORDINATES, dtype=numpy.float64)
# Inset from tag layout so tag plastic / border is not clustered as cubes.
PLAY_AREA_XY_MARGIN_M = 0.015
PLAY_AREA_X_MIN_M = float(_TAG_CENTERS_XY[:, 0].min() + PLAY_AREA_XY_MARGIN_M)
PLAY_AREA_X_MAX_M = float(_TAG_CENTERS_XY[:, 0].max() - PLAY_AREA_XY_MARGIN_M)
PLAY_AREA_Y_MIN_M = float(_TAG_CENTERS_XY[:, 1].min() + PLAY_AREA_XY_MARGIN_M)
PLAY_AREA_Y_MAX_M = float(_TAG_CENTERS_XY[:, 1].max() - PLAY_AREA_XY_MARGIN_M)
# Vertical band (robot Z): table surface + cubes; independent of tag XY layout.
WORKSPACE_Z_ROBOT_M = (-0.12, 0.48)

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


def points_to_scene_meters(point_cloud, camera_pose: numpy.ndarray | None = None) -> numpy.ndarray:
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
    pts_m = pts_m.astype(numpy.float64)
    if camera_pose is not None:
        pts_m = filter_points_playmat_cam_frame(pts_m, camera_pose)
    return pts_m


def filter_points_playmat_cam_frame(pts_m: numpy.ndarray, T_cam_robot: numpy.ndarray) -> numpy.ndarray:
    """Keep camera-frame points whose world/robot position lies inside the tag-bounded play mat."""
    if pts_m.shape[0] == 0:
        return pts_m
    T_robot_cam = numpy.linalg.inv(numpy.asarray(T_cam_robot, dtype=numpy.float64))
    hom = numpy.ones((pts_m.shape[0], 4), dtype=numpy.float64)
    hom[:, :3] = pts_m
    pr = (T_robot_cam @ hom.T).T[:, :3]
    z0, z1 = WORKSPACE_Z_ROBOT_M
    m = (
        (pr[:, 0] >= PLAY_AREA_X_MIN_M)
        & (pr[:, 0] <= PLAY_AREA_X_MAX_M)
        & (pr[:, 1] >= PLAY_AREA_Y_MIN_M)
        & (pr[:, 1] <= PLAY_AREA_Y_MAX_M)
        & (pr[:, 2] >= z0)
        & (pr[:, 2] <= z1)
    )
    return pts_m[m]


def is_center_on_playmat_robot(p_robot_xyz: numpy.ndarray) -> bool:
    x, y, z = float(p_robot_xyz[0]), float(p_robot_xyz[1]), float(p_robot_xyz[2])
    z0, z1 = WORKSPACE_Z_ROBOT_M
    return (
        (PLAY_AREA_X_MIN_M <= x <= PLAY_AREA_X_MAX_M)
        and (PLAY_AREA_Y_MIN_M <= y <= PLAY_AREA_Y_MAX_M)
        and (z0 <= z <= z1)
    )


def cluster_median_robot(
    pts_cam_m: numpy.ndarray,
    T_cam_robot: numpy.ndarray,
) -> numpy.ndarray:
    T_robot_cam = numpy.linalg.inv(numpy.asarray(T_cam_robot, dtype=numpy.float64))
    hom = numpy.ones((pts_cam_m.shape[0], 4), dtype=numpy.float64)
    hom[:, :3] = pts_cam_m
    pr = (T_robot_cam @ hom.T).T[:, :3]
    return numpy.median(pr, axis=0)


def dense_margin_for_cluster(cluster: o3d.geometry.PointCloud) -> float:
    """Wider crop for physically large clusters (≈30 mm) so more surface points are used."""
    try:
        obb = cluster.get_oriented_bounding_box()
        ext = numpy.sort(numpy.asarray(obb.extent))
        if ext.size >= 3 and float(ext[2]) >= LARGE_CLUSTER_MIN_EXTENT_M:
            return LARGE_DENSE_CROP_MARGIN_M
    except Exception:
        pass
    return DENSE_CROP_MARGIN_M


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


def refine_cube_pose_bound_center_cam(
    pts_cam: numpy.ndarray,
    t_cam_cube: numpy.ndarray,
    iters: int,
) -> numpy.ndarray:
    """More AABB-in-cube-frame centering (same idea as orientation_RRC BOUND_CENTER_ITERS)."""
    R = t_cam_cube[:3, :3].copy()
    c = t_cam_cube[:3, 3].copy()
    for _ in range(max(0, int(iters))):
        q = R.T @ (pts_cam - c[None, :]).T
        delta = numpy.array(
            [0.5 * (float(numpy.min(q[k])) + float(numpy.max(q[k]))) for k in range(3)],
            dtype=numpy.float64,
        )
        if float(numpy.linalg.norm(delta)) < 1e-9:
            break
        c = c + R @ delta
    out = numpy.copy(t_cam_cube)
    out[:3, 3] = c
    return out


def snap_cube_center_along_plane_normal(
    center_cam: numpy.ndarray,
    pts_cam: numpy.ndarray,
    plane_model: numpy.ndarray,
    camera_pose: numpy.ndarray,
    edge_m: float,
    *,
    h_contact_percentile: float = 0.0,
) -> numpy.ndarray:
    """
    Move center along the table normal so signed plane distance matches ``h_contact + edge/2``.
    ``h_contact`` is the reference height of the contact patch: ``min(h)`` by default, or
    ``percentile(h, h_contact_percentile)`` when ``h_contact_percentile > 0`` (large cubes).
    """
    pm = flip_plane_to_robot_up(plane_model, camera_pose)
    h_pts = _signed_plane_dist(pts_cam, pm)
    if h_contact_percentile > 0.0:
        h_contact = float(numpy.percentile(h_pts, float(h_contact_percentile)))
    else:
        h_contact = float(numpy.min(h_pts))
    n = pm[:3] / (numpy.linalg.norm(pm[:3]) + 1e-12)
    h_c = float(_signed_plane_dist(center_cam.reshape(1, 3), pm)[0])
    target_h = h_contact + float(edge_m) * 0.5
    return numpy.asarray(center_cam, dtype=numpy.float64) + n * (target_h - h_c)


def _is_nominal_large_cube(edge_m: float) -> bool:
    """30 mm inventory — tightest grasp margin; extra pose refinement."""
    return abs(float(edge_m) - float(CUBE_SIZE_LARGE_M)) < 1.5e-3


def compute_cube_pose_o2(
    pts: numpy.ndarray,
    plane_model: numpy.ndarray,
    camera_pose: numpy.ndarray,
    edge_m: float,
) -> tuple[numpy.ndarray, numpy.ndarray] | None:
    """
    Table-plane physical pose, extra bound-centering, height snap along normal, robot transform.
    Large (30 mm) cubes use a thinner bottom layer, more AABB iterations, robust contact
    height (percentile), and multiple refine↔snap cycles.
    """
    if pts.shape[0] < 30:
        return None
    large = _is_nominal_large_cube(edge_m)
    pair = physical_cube_pose_from_points(
        pts,
        plane_model,
        camera_pose,
        edge_m,
        bottom_layer_frac=(LARGE_BOTTOM_LAYER_FRAC if large else None),
        bound_center_iters=(LARGE_PHYSICAL_BOUND_ITERS if large else None),
    )
    if pair is None:
        return None
    _, t_cam = pair
    extra = POSE_BOUND_CENTER_EXTRA_ITERS_LARGE if large else POSE_BOUND_CENTER_EXTRA_ITERS
    snap_pct = LARGE_H_CONTACT_PERCENTILE if large else 0.0
    cycles = LARGE_POSE_REFINE_SNAP_CYCLES if large else 1
    t_cam = numpy.copy(t_cam)
    for _ in range(cycles):
        t_cam = refine_cube_pose_bound_center_cam(pts, t_cam, extra)
        c = snap_cube_center_along_plane_normal(
            t_cam[:3, 3],
            pts,
            plane_model,
            camera_pose,
            edge_m,
            h_contact_percentile=snap_pct,
        )
        t_cam[:3, 3] = c
    T_cam_robot = numpy.asarray(camera_pose, dtype=numpy.float64)
    t_robot = numpy.linalg.inv(T_cam_robot) @ t_cam
    return t_robot, t_cam


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

    full_pts_m = points_to_scene_meters(point_cloud, camera_pose)
    if full_pts_m.shape[0] < 120:
        return []

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(full_pts_m)

    cube_pcds, _msg, plane_np = isolate_cube_cluster_open3d(pcd, num_cubes=max_cubes)
    if not cube_pcds or plane_np is None:
        return []

    out: list[dict] = []
    for c in cube_pcds:
        pts_sparse = numpy.asarray(c.points, dtype=numpy.float64)
        if pts_sparse.shape[0] < 15:
            continue
        if not is_center_on_playmat_robot(cluster_median_robot(pts_sparse, camera_pose)):
            continue

        pts = extract_dense_cluster_points(full_pts_m, c, margin_m=dense_margin_for_cluster(c))
        if pts.shape[0] < 35:
            pts = pts_sparse
        if pts.shape[0] < 30:
            continue
        edge_m = classify_nominal_edge(pts, plane_np, camera_pose)
        pose_pair = compute_cube_pose_o2(pts, plane_np, camera_pose, edge_m)
        if pose_pair is None:
            continue
        t_robot, t_cam = pose_pair
        ctr = numpy.asarray(t_robot[:3, 3], dtype=numpy.float64)
        if not is_center_on_playmat_robot(ctr):
            continue
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
