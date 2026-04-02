"""
o2_RRC — challenge 2 (minimal): AprilTag (checkpoint0), cluster cubes, nominal 22.5 / 25 / 30 mm.

**Inventory (9 cubes)**: three nominal sizes, three cubes each — large (30 mm, e.g. red/green/blue),
medium (25 mm), small (22.5 mm). Route picks **all three large** (order by −X), then **three medium**,
then **three small** (same tie-break).

**Arm pose math** (must match ``orientation_RRC.physical_cube_pose_from_points``):
``get_transform_camera_robot`` → ``T_cam_robot`` with **P_cam = T_cam_robot @ P_robot** (tag/world = robot
frame). Cube pose ``T_cam_cube`` maps cube frame to camera. Then **T_robot_cube = inv(T_cam_robot) @ T_cam_cube**.

**Perception**: Points cropped to the play mat; pose = physical init + ICP + footprint blend. Detections
whose cube center **does not project into the current image** are rejected (avoids grasping to empty space
when pose is inconsistent with the RGB view).

**Route**: Sort by **nominal** edge (not raw float), then −X. Match steps by nominal edge + nearest XY.
**Grasp**: yaw snapped to 90°; no wiggle.
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
    orthonormalize_rotation,
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
# Grasp: slow at cube level (~35 mm gripper vs 30 mm cubes → need precise XY; keep ≥50).
ARM_SPEED_GRASP_CUBE_LEVEL = 50
ARM_SPEED_GRASP_DESCEND_FINAL = 50
GRASP_FINAL_APPROACH_MM = 5.0
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
# Expected counts per nominal size (challenge layout: 3 large, 3 medium, 3 small).
CUBES_PER_NOMINAL_SIZE = 3

GPU_Z_MIN_M = 0.28
GPU_Z_MAX_M = 1.4
DENSE_CROP_MARGIN_M = 0.008
# Per-size dense re-crop after nominal edge is known (meters).
SMALL_DENSE_CROP_MARGIN_M = 0.006
MEDIUM_DENSE_CROP_MARGIN_M = 0.008
# Larger clusters (~30 mm) need a wider crop so sides/top contribute to pose.
LARGE_DENSE_CROP_MARGIN_M = 0.012
LARGE_CLUSTER_MIN_EXTENT_M = 0.027

# Open3D ICP: synthetic cube mesh → measured cluster (camera frame, meters).
ICP_MESH_SAMPLES = 2800
ICP_VOXEL_M = 0.0025
ICP_MAX_CORR_COARSE_M = 0.020
ICP_MAX_CORR_FINE_M = 0.007
ICP_COARSE_ITERS = 55
ICP_FINE_ITERS = 45
ICP_MIN_FITNESS = 0.12

# Pre-ICP physical init: single pipeline; ICP does the fine alignment.
PHYSICAL_INIT_BOUND_EXTRA = 4
PHYSICAL_INIT_YAW_SOURCE = "bottom"
# Extra AABB centering in cube frame after ICP + footprint blend (large cubes need tight XY).
BOUND_CENTER_ITERS_AFTER_ICP_SMALL = 4
BOUND_CENTER_ITERS_AFTER_ICP_MEDIUM = 6
BOUND_CENTER_ITERS_AFTER_ICP_LARGE = 10


def _nominal_edge_bucket(edge_m: float) -> str:
    e = float(edge_m)
    if abs(e - float(CUBE_SIZE_LARGE_M)) < 1.5e-3:
        return "large"
    if abs(e - float(CUBE_SIZE_SMALL_M)) < 1.5e-3:
        return "small"
    return "medium"


def bottom_slice_height_fraction_for_edge(edge_m: float) -> float:
    """
    Fraction of cluster height range used to mask bottom-face points for footprint centroid.
    Large cubes: narrow band (true contact only). Small: wider (sparse bottom in depth).
    """
    b = _nominal_edge_bucket(edge_m)
    if b == "large":
        return 0.10
    if b == "small":
        return 0.24
    return 0.18


def footprint_translation_blend_weight(edge_m: float) -> float:
    """
    Weight on table-plane footprint centroid vs ICP translation. ~35 mm gripper on 30 mm
    cubes needs nearly full footprint trust for large; small clouds blend more with ICP.
    """
    b = _nominal_edge_bucket(edge_m)
    if b == "large":
        return 0.92
    if b == "small":
        return 0.38
    return 0.48


def cube_center_from_bottom_footprint(
    pts_cam: numpy.ndarray,
    plane_model: numpy.ndarray,
    camera_pose: numpy.ndarray,
    edge_m: float,
) -> numpy.ndarray:
    """
    Geometric cube center from bottom-face footprint: centroid of points projected onto the
    table plane, then offset along the plane normal by h_contact + edge/2 (same h_contact
    rule as ``snap_cube_center_along_plane_normal``).
    """
    pm = flip_plane_to_robot_up(plane_model, camera_pose)
    n = pm[:3] / (numpy.linalg.norm(pm[:3]) + 1e-12)
    h_pts = _signed_plane_dist(pts_cam, pm)
    h_min = float(numpy.min(h_pts))
    h_max = float(numpy.max(h_pts))
    span = max(h_max - h_min, 1e-9)
    frac = bottom_slice_height_fraction_for_edge(edge_m)
    mask = h_pts <= h_min + span * frac
    if int(numpy.sum(mask)) < 8:
        mask = h_pts <= h_min + span * max(frac, 0.28)
    p = pts_cam[mask]
    sd = _signed_plane_dist(p, pm)
    p_proj = p - sd[:, numpy.newaxis] * n[numpy.newaxis, :]
    c_flat = numpy.mean(p_proj, axis=0)
    hp = h_contact_percentile_for_edge(edge_m)
    if hp > 0.0:
        h_contact = float(numpy.percentile(h_pts, float(hp)))
    else:
        h_contact = float(numpy.min(h_pts))
    return c_flat + n * (h_contact + float(edge_m) * 0.5)


def refine_translation_footprint_icp_blend(
    t_cam_cube: numpy.ndarray,
    pts_cam: numpy.ndarray,
    plane_model: numpy.ndarray,
    camera_pose: numpy.ndarray,
    edge_m: float,
) -> numpy.ndarray:
    """Blend footprint-based center with ICP translation, then re-snap height along plane normal."""
    w = footprint_translation_blend_weight(edge_m)
    c_foot = cube_center_from_bottom_footprint(pts_cam, plane_model, camera_pose, edge_m)
    c_icp = numpy.asarray(t_cam_cube[:3, 3], dtype=numpy.float64)
    c_blend = w * c_foot + (1.0 - w) * c_icp
    c_snap = snap_cube_center_along_plane_normal(
        c_blend,
        pts_cam,
        plane_model,
        camera_pose,
        edge_m,
        h_contact_percentile=h_contact_percentile_for_edge(edge_m),
    )
    out = numpy.copy(t_cam_cube)
    out[:3, 3] = c_snap
    return out


def _icp_tuning_for_edge(edge_m: float) -> tuple[float, float, int, int]:
    """(voxel_m, max_corr_fine_m, coarse_iters, fine_iters) — tighter for 30 mm (gripper margin)."""
    b = _nominal_edge_bucket(edge_m)
    if b == "large":
        return 0.0018, 0.0045, 68, 58
    if b == "small":
        return 0.0030, 0.0080, 48, 38
    return ICP_VOXEL_M, ICP_MAX_CORR_FINE_M, ICP_COARSE_ITERS, ICP_FINE_ITERS


def bound_center_iters_after_icp(edge_m: float) -> int:
    b = _nominal_edge_bucket(edge_m)
    if b == "large":
        return BOUND_CENTER_ITERS_AFTER_ICP_LARGE
    if b == "small":
        return BOUND_CENTER_ITERS_AFTER_ICP_SMALL
    return BOUND_CENTER_ITERS_AFTER_ICP_MEDIUM


def dense_margin_for_nominal_edge(edge_m: float) -> float:
    """Second-pass crop margin after ``classify_nominal_edge`` (size-aware)."""
    e = float(edge_m)
    if abs(e - float(CUBE_SIZE_SMALL_M)) < 1.5e-3:
        return SMALL_DENSE_CROP_MARGIN_M
    if abs(e - float(CUBE_SIZE_LARGE_M)) < 1.5e-3:
        return LARGE_DENSE_CROP_MARGIN_M
    return MEDIUM_DENSE_CROP_MARGIN_M

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

# Match live cube to planned step (meters). Compare **nominal** edges via ``snap_edge_to_nominal``.
MATCH_MAX_DIST_M = 0.042
# Soft fallback only if still within this XY distance (meters) from planned center.
FALLBACK_MAX_DIST_M = 0.052
# Soft fallback when no exact nominal match (raw edge tolerance, meters).
MATCH_EDGE_SOFT_TOL_M = 0.008
# Cube center must project into image with this margin (pixels) or detection is rejected.
IMAGE_PROJECTION_MARGIN_PX = 12.0


def snap_edge_to_nominal(edge_m: float) -> float:
    return min(CUBE_SIZES_M, key=lambda s: abs(float(s) - float(edge_m)))


def cube_center_visible_in_image(
    t_cam_cube: numpy.ndarray,
    camera_intrinsic: numpy.ndarray,
    image_shape_hw,
    *,
    margin_px: float = IMAGE_PROJECTION_MARGIN_PX,
) -> bool:
    """
    Cube origin in camera frame must lie in front of the camera and project inside the
    image (OpenCV camera: +Z forward, X right, Y down). Rejects bogus poses that would
    drive the arm to empty space while nothing appears at that pixel.
    """
    K = numpy.asarray(camera_intrinsic, dtype=numpy.float64)
    p = numpy.asarray(t_cam_cube[:3, 3], dtype=numpy.float64).reshape(3)
    z = float(p[2])
    if z <= 0.05:
        return False
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    u = fx * float(p[0]) / z + cx
    v = fy * float(p[1]) / z + cy
    h = int(image_shape_hw[0])
    w = int(image_shape_hw[1])
    m = float(margin_px)
    return m <= u < w - m and m <= v < h - m


def count_detections_by_nominal(dets: list[dict]) -> dict[float, int]:
    """How many detections per nominal edge (for inventory sanity)."""
    out: dict[float, int] = {}
    for d in dets:
        k = float(snap_edge_to_nominal(float(d["edge_m"])))
        out[k] = out.get(k, 0) + 1
    return out


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
    """
    Nominal edge (22.5 / 25 / 30 mm). **Primary** cue: vertical span along the table
    normal in **tag/world meters** (PnP-scaled) — that matches physical cube height.
    Secondary: OBB + footprint (legacy) when ambiguous.
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

    edge_from_height = snap_edge_to_nominal(span_v)
    edge_from_pool = snap_edge_to_nominal(pooled)

    # Height span is the most direct metric for stock size in world units.
    if edge_from_height != edge_from_pool:
        err_h = abs(span_v - float(edge_from_height))
        err_p = abs(span_v - float(edge_from_pool))
        if err_h + 1e-9 < err_p:
            return float(edge_from_height)
        if err_p + 1e-9 < err_h:
            return float(edge_from_pool)
        # Tie-break: favour height for large physical cubes (common mislabel: blue 30 mm → 25 mm).
        if span_v >= 0.0275:
            return float(CUBE_SIZE_LARGE_M)
        if span_v <= 0.0235:
            return float(edge_from_height)
        return float(edge_from_pool)

    return float(edge_from_height)


def h_contact_percentile_for_edge(edge_m: float) -> float:
    """Percentile for contact height on large cubes (noisy bottom band); 0 = min(h)."""
    if abs(float(edge_m) - float(CUBE_SIZE_LARGE_M)) < 1.5e-3:
        return 3.5
    return 0.0


def icp_refine_cube_pose_cam(
    pts_cam: numpy.ndarray,
    t_cam_init: numpy.ndarray,
    edge_m: float,
) -> numpy.ndarray:
    """
    Refine cube pose in camera frame: register a synthetic ``edge_m`` cube mesh (centered
    at origin in cube frame) to the measured cluster via Open3D ICP. Handles oblique views
    without snapping to mat cardinals. Voxel / fine correspondence tightened for 30 mm cubes.
    """
    em = float(edge_m)
    T0 = numpy.asarray(t_cam_init, dtype=numpy.float64)
    if em < 1e-4 or pts_cam.shape[0] < 25:
        return T0
    voxel_m, corr_fine_m, coarse_iters, fine_iters = _icp_tuning_for_edge(edge_m)
    mesh = o3d.geometry.TriangleMesh.create_box(em, em, em)
    mesh.translate(-numpy.array([em * 0.5, em * 0.5, em * 0.5], dtype=numpy.float64))
    n_src = min(ICP_MESH_SAMPLES, max(500, int(pts_cam.shape[0] * 30)))
    source = mesh.sample_points_uniformly(number_of_points=n_src)
    target = o3d.geometry.PointCloud()
    target.points = o3d.utility.Vector3dVector(pts_cam.astype(numpy.float64))
    target_ds = target.voxel_down_sample(voxel_size=float(voxel_m))
    if len(target_ds.points) < 20:
        target_ds = target
    try:
        target_ds.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.012, max_nn=30)
        )
    except Exception:
        return T0
    criteria_coarse = o3d.pipelines.registration.ICPConvergenceCriteria(
        relative_fitness=1e-6,
        relative_rmse=1e-6,
        max_iteration=int(coarse_iters),
    )
    try:
        reg1 = o3d.pipelines.registration.registration_icp(
            source,
            target_ds,
            ICP_MAX_CORR_COARSE_M,
            T0,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            criteria_coarse,
        )
        T1 = numpy.asarray(reg1.transformation, dtype=numpy.float64)
        if float(reg1.fitness) < ICP_MIN_FITNESS:
            return T0
        try:
            target_ds.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.008, max_nn=25)
            )
        except Exception:
            T = T1
            T[:3, :3] = orthonormalize_rotation(T[:3, :3])
            return T
        criteria_fine = o3d.pipelines.registration.ICPConvergenceCriteria(
            max_iteration=int(fine_iters),
        )
        reg2 = o3d.pipelines.registration.registration_icp(
            source,
            target_ds,
            float(corr_fine_m),
            T1,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            criteria_fine,
        )
        T = numpy.asarray(reg2.transformation, dtype=numpy.float64)
        if float(reg2.fitness) < ICP_MIN_FITNESS * 0.85:
            T = T1
        T[:3, :3] = orthonormalize_rotation(T[:3, :3])
        return T
    except Exception:
        return T0


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
    ``percentile(h, h_contact_percentile)`` when ``h_contact_percentile > 0``.
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


def compute_cube_pose_o2(
    pts: numpy.ndarray,
    plane_model: numpy.ndarray,
    camera_pose: numpy.ndarray,
    edge_m: float,
) -> tuple[numpy.ndarray, numpy.ndarray] | None:
    """
    Seed pose from ``physical_cube_pose_from_points`` (table plane + footprint), one
    bound-center + height snap along the plane normal, then **Open3D ICP** (synthetic cube
    mesh → cloud). **Translation** is then blended with a **table-plane footprint centroid**
    (size-dependent weight; large cubes → ~92% footprint for ~35 mm gripper vs 30 mm stock),
    extra AABB centering in cube frame, then robot pose.
    """
    if pts.shape[0] < 30:
        return None
    pair = physical_cube_pose_from_points(
        pts,
        plane_model,
        camera_pose,
        edge_m,
        yaw_source=PHYSICAL_INIT_YAW_SOURCE,
    )
    if pair is None:
        return None
    _, t_cam = pair
    t_cam = numpy.copy(t_cam)
    t_cam = refine_cube_pose_bound_center_cam(pts, t_cam, PHYSICAL_INIT_BOUND_EXTRA)
    c = snap_cube_center_along_plane_normal(
        t_cam[:3, 3],
        pts,
        plane_model,
        camera_pose,
        edge_m,
        h_contact_percentile=h_contact_percentile_for_edge(edge_m),
    )
    t_cam[:3, 3] = c
    t_cam = icp_refine_cube_pose_cam(pts, t_cam, edge_m)
    t_cam = refine_translation_footprint_icp_blend(
        t_cam, pts, plane_model, camera_pose, edge_m
    )
    t_cam = refine_cube_pose_bound_center_cam(pts, t_cam, bound_center_iters_after_icp(edge_m))
    # P_cam = T_cam_robot @ P_robot and P_cam = T_cam_cube @ P_cube  →  T_robot_cube = T_robot_cam @ T_cam_cube
    T_cam_robot = numpy.asarray(camera_pose, dtype=numpy.float64)
    t_robot = numpy.linalg.inv(T_cam_robot) @ t_cam
    return t_robot, t_cam


def dedupe_detections(dets: list[dict]) -> list[dict]:
    if len(dets) <= 1:
        return dets
    ordered = sorted(
        dets,
        key=lambda d: (-float(snap_edge_to_nominal(float(d["edge_m"]))), -float(d["t_robot"][0, 3])),
    )
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
        pts_sz = extract_dense_cluster_points(
            full_pts_m, c, margin_m=dense_margin_for_nominal_edge(edge_m)
        )
        if pts_sz.shape[0] >= 40:
            pts = pts_sz
        pose_pair = compute_cube_pose_o2(pts, plane_np, camera_pose, edge_m)
        if pose_pair is None:
            continue
        t_robot, t_cam = pose_pair
        ctr = numpy.asarray(t_robot[:3, 3], dtype=numpy.float64)
        if not is_center_on_playmat_robot(ctr):
            continue
        if camera_intrinsic is not None and image is not None:
            if not cube_center_visible_in_image(t_cam, camera_intrinsic, image.shape):
                continue
        out.append({"t_robot": t_robot, "t_cam": t_cam, "edge_m": float(edge_m)})
    return dedupe_detections(out)


def preplan_route(dets: list[dict]) -> list[tuple[numpy.ndarray, float]]:
    """
    Pick order: **nominal** size large → medium → small (three of each in the challenge),
    then tie-break **−X** (more negative / table left first) within each nominal size.
    Returns list of (center_xyz, edge_m) length <= 9.
    """
    if not dets:
        return []
    ranked = sorted(
        dets,
        key=lambda d: (-float(snap_edge_to_nominal(float(d["edge_m"]))), -float(d["t_robot"][0, 3])),
    )
    ranked = ranked[:TOTAL_CUBES]
    return [(numpy.asarray(d["t_robot"][:3, 3], dtype=numpy.float64).copy(), float(d["edge_m"])) for d in ranked]


def match_step_to_detection(
    dets: list[dict],
    planned_center: numpy.ndarray,
    planned_edge: float,
) -> dict | None:
    """Nearest cube whose **nominal** edge matches the planned step (not raw float ±tol)."""
    pe = snap_edge_to_nominal(float(planned_edge))
    best: dict | None = None
    best_d = 1e9
    pc = numpy.asarray(planned_center, dtype=numpy.float64).reshape(3)
    for d in dets:
        if snap_edge_to_nominal(float(d["edge_m"])) != pe:
            continue
        p = numpy.asarray(d["t_robot"][:3, 3], dtype=numpy.float64)
        dist = float(numpy.linalg.norm(p - pc))
        if dist < best_d:
            best_d = dist
            best = d
    if best is None:
        return None
    if best_d > MATCH_MAX_DIST_M:
        return None
    return best


def fallback_pick_for_step(
    dets: list[dict],
    planned_center: numpy.ndarray,
    planned_edge: float,
) -> dict | None:
    """
    If strict nominal match failed: prefer same nominal edge anyway (nearest XY) within
    ``FALLBACK_MAX_DIST_M``, then soft raw-edge tolerance — never “just take largest edge”.
    """
    if not dets:
        return None
    pe = snap_edge_to_nominal(float(planned_edge))
    pc = numpy.asarray(planned_center, dtype=numpy.float64).reshape(3)

    same_nominal = [d for d in dets if snap_edge_to_nominal(float(d["edge_m"])) == pe]
    if same_nominal:
        best = min(
            same_nominal,
            key=lambda d: float(numpy.linalg.norm(numpy.asarray(d["t_robot"][:3, 3]) - pc)),
        )
        d_xy = float(numpy.linalg.norm(numpy.asarray(best["t_robot"][:3, 3]) - pc))
        if d_xy <= FALLBACK_MAX_DIST_M:
            return best

    soft = [
        d
        for d in dets
        if abs(float(d["edge_m"]) - float(planned_edge)) < MATCH_EDGE_SOFT_TOL_M
    ]
    if soft:
        best = min(
            soft,
            key=lambda d: float(numpy.linalg.norm(numpy.asarray(d["t_robot"][:3, 3]) - pc)),
        )
        d_xy = float(numpy.linalg.norm(numpy.asarray(best["t_robot"][:3, 3]) - pc))
        if d_xy <= FALLBACK_MAX_DIST_M:
            return best
    return None


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
    # Pre-touch height: stay below safe clearance; at least ~0.5 mm above grasp, then final plunge.
    pre_touch_z = min(
        grasp_z_mm + float(GRASP_FINAL_APPROACH_MM),
        float(safe_z_mm) - 1.0,
    )
    pre_touch_z = max(pre_touch_z, grasp_z_mm + 0.5)
    if pre_touch_z > grasp_z_mm + 0.2:
        set_line(arm, x_mm, y_mm, pre_touch_z, yaw_use, ARM_SPEED_GRASP_CUBE_LEVEL)
        set_line(arm, x_mm, y_mm, grasp_z_mm, yaw_use, ARM_SPEED_GRASP_DESCEND_FINAL)
    else:
        set_line(arm, x_mm, y_mm, grasp_z_mm, yaw_use, ARM_SPEED_GRASP_DESCEND_FINAL)
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
        cnt = count_detections_by_nominal(d0)
        disp = img.copy()
        for d in d0:
            draw_pose_axes(disp, camera_intrinsic, d["t_cam"])
        lines = [
            f"cubes={len(d0)}  route: 3x30mm -> 3x25mm -> 3x22.5mm (nominal), steps={len(route)}",
            f"per-size counts {cnt} (expect 3 each)",
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
            cnt = count_detections_by_nominal(dets)
            print(f"Planned route: {len(route)} picks (nominal large x3, medium x3, small x3). Counts: {cnt}")
            for em, n in cnt.items():
                if n != CUBES_PER_NOMINAL_SIZE:
                    print(f"  Warning: expected {CUBES_PER_NOMINAL_SIZE} cubes at {em:.4f} m, saw {n}.")

        if placed >= len(route):
            break
        if not dets:
            print("Lost all detections.")
            break

        pc, pe = route[placed]
        pick = match_step_to_detection(dets, pc, pe)
        if pick is None:
            pick = fallback_pick_for_step(dets, pc, pe)
            if pick is not None:
                print(f"Step {placed + 1}: using soft edge / nearest-XY fallback to planned nominal.")
        if pick is None:
            print(f"Step {placed + 1}: no detection matches planned edge {pe:.4f} m — stopping.")
            break

        if camera_intrinsic is not None and zed.image is not None:
            if not cube_center_visible_in_image(pick["t_cam"], camera_intrinsic, zed.image.shape):
                print(f"Step {placed + 1}: cube pose does not project into image — skip grasp (bad pose).")
                break

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
