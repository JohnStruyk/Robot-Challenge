"""
orientation_RRC — high-precision cube center + orientation for known cubes.

Uses CUDA (when available) for depth masking and robust statistics; geometry uses
table-plane RANSAC, 2D min-area rectangle in the table plane for yaw, and
vertical centering from min/max extent along the table normal (cube geometry).

Drop-in style API for checkpoint22: ``get_cube_transforms(...)`` returns the same
shape as there (list of ``(t_robot_cube, t_cam_cube)`` tuples).
"""
from __future__ import annotations

import time
import traceback

import cv2
import numpy
import open3d as o3d
from xarm.wrapper import XArmAPI

from checkpoint0 import get_transform_camera_robot
from checkpoint1 import grasp_cube, place_cube, GRIPPER_LENGTH, robot_ip
from utils.vis_utils import draw_pose_axes
from utils.zed_camera import ZedCamera

try:
    import torch
except Exception:
    torch = None

# --- Scene / cube (match checkpoint22) ---
CUBE_SIZE_M = 0.025
STACK_HEIGHT_GOAL = 12

# GPU depth band (ZED clouds often in meters after heuristic; mask also works in raw mm band)
GPU_Z_MIN_M = 0.30
GPU_Z_MAX_M = 2.00

# Segmentation (slightly finer voxel + multi-eps DBSCAN for stable clusters)
VOXEL_M = 0.0025
PLANE_DIST_M = 0.008
DBSCAN_EPS_CANDIDATES_M = (0.016, 0.018, 0.020, 0.022)
DBSCAN_MIN_PTS = 25


def orthonormalize_rotation(R: numpy.ndarray) -> numpy.ndarray:
    """Project 3x3 matrix onto SO(3)."""
    U, _, Vt = numpy.linalg.svd(R)
    Rn = U @ Vt
    if numpy.linalg.det(Rn) < 0:
        U[:, -1] *= -1.0
        Rn = U @ Vt
    return Rn


def gpu_depth_mask(point_cloud, z_min_m=GPU_Z_MIN_M, z_max_m=GPU_Z_MAX_M):
    """Depth validity mask with CUDA morphological cleanup when torch+GPU exist."""
    if point_cloud is None:
        return None
    z = point_cloud[..., 2]
    if torch is None:
        m = numpy.isfinite(z) & (z > z_min_m) & (z < z_max_m)
        return (m.astype(numpy.uint8) * 255)

    use_cuda = torch.cuda.is_available()
    device = "cuda" if use_cuda else "cpu"
    z_t = torch.as_tensor(z, device=device, dtype=torch.float32)
    finite = torch.isfinite(z_t)
    mask = finite & (z_t > z_min_m) & (z_t < z_max_m)
    x = mask.float().unsqueeze(0).unsqueeze(0)
    for _ in range(2):
        x = torch.nn.functional.max_pool2d(x, kernel_size=5, stride=1, padding=2)
        x = 1.0 - torch.nn.functional.max_pool2d(1.0 - x, kernel_size=5, stride=1, padding=2)
    for _ in range(1):
        x = 1.0 - torch.nn.functional.max_pool2d(1.0 - x, kernel_size=3, stride=1, padding=1)
        x = torch.nn.functional.max_pool2d(x, kernel_size=3, stride=1, padding=1)
    m = (x[0, 0] > 0.5).detach().cpu().numpy()
    return (m.astype(numpy.uint8) * 255)


def points_to_meters_open3d(xyz):
    """Scale XYZ to meters if values look like millimeters."""
    if xyz.size == 0:
        return xyz, 1.0
    max_abs = float(numpy.nanmax(numpy.abs(xyz)))
    scale = 0.001 if max_abs > 50.0 else 1.0
    return (xyz * scale).astype(numpy.float64), scale


def plane_point_from_model(plane_model: numpy.ndarray) -> numpy.ndarray:
    """One point on plane ax+by+cz+d=0 (numpy array length 4)."""
    a, b, c, d = float(plane_model[0]), float(plane_model[1]), float(plane_model[2]), float(plane_model[3])
    nrm = numpy.sqrt(a * a + b * b + c * c) + 1e-12
    a, b, c, d = a / nrm, b / nrm, c / nrm, d / nrm
    if abs(c) > 0.15:
        return numpy.array([0.0, 0.0, -d / c], dtype=numpy.float64)
    if abs(a) > 0.15:
        return numpy.array([-d / a, 0.0, 0.0], dtype=numpy.float64)
    return numpy.array([0.0, -d / b, 0.0], dtype=numpy.float64)


def isolate_cube_cluster_open3d(pcd: o3d.geometry.PointCloud, num_cubes: int):
    """
    Segment cubes; return clusters, table plane normal, plane model, status.

    Returns
    -------
    tuple
        (list of cluster clouds or None, plane_normal or None, plane_model or None, message).
    """
    if len(pcd.points) < 150:
        return None, None, None, "too few points after NaN filter"

    pcd = pcd.voxel_down_sample(voxel_size=VOXEL_M)
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=25, std_ratio=2.0)

    if len(pcd.points) < 100:
        return None, None, None, "too few points after voxel/outlier"

    plane_model, inliers = pcd.segment_plane(
        distance_threshold=PLANE_DIST_M,
        ransac_n=3,
        num_iterations=2000,
    )
    plane_np = numpy.asarray(plane_model, dtype=numpy.float64)
    plane_n = numpy.asarray(plane_np[:3], dtype=numpy.float64)
    pn = numpy.linalg.norm(plane_n)
    if pn > 1e-9:
        plane_n /= pn
    else:
        plane_n = None

    if len(inliers) > 0 and len(inliers) > 0.12 * len(pcd.points):
        pcd = pcd.select_by_index(inliers, invert=True)

    if len(pcd.points) < 80:
        return None, plane_n, plane_np, "nothing left after plane removal"

    labels = None
    for eps in DBSCAN_EPS_CANDIDATES_M:
        candidate = numpy.asarray(
            pcd.cluster_dbscan(eps=eps, min_points=DBSCAN_MIN_PTS, print_progress=False)
        )
        if candidate.size > 0 and candidate.max() >= 0:
            labels = candidate
            break
    if labels is None:
        return None, plane_n, plane_np, "DBSCAN found no clusters"

    max_label = int(labels.max())

    def score_cluster(cluster, idx):
        obb = cluster.get_oriented_bounding_box()
        ext = numpy.sort(numpy.asarray(obb.extent))
        if ext[2] < 1e-9:
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
        return top_clusters, plane_n, plane_np, f"{len(top_clusters)} cube-like clusters"

    fallback_clusters.sort(key=lambda x: x[0], reverse=True)
    if len(fallback_clusters) > 0:
        top_clusters = [c for (_, c) in fallback_clusters[:num_cubes] if len(c.points) >= 40]
        if len(top_clusters) > 0:
            return top_clusters, plane_n, plane_np, f"fallback: {len(top_clusters)} largest clusters"

    return None, plane_n, plane_np, "no cluster passed filters"


def gpu_robust_center(pts: numpy.ndarray) -> numpy.ndarray:
    """Median center with optional CUDA; falls back to numpy."""
    if pts.shape[0] == 0:
        return numpy.zeros(3, dtype=numpy.float64)
    if torch is not None and torch.cuda.is_available():
        t = torch.as_tensor(pts, device="cuda", dtype=torch.float32)
        med, _ = torch.median(t, dim=0)
        return med.detach().cpu().numpy().astype(numpy.float64)
    return numpy.median(pts, axis=0).astype(numpy.float64)


def gpu_extent_along_axis(
    pts: numpy.ndarray, axis: numpy.ndarray, origin: numpy.ndarray | None = None
) -> tuple[float, float]:
    """Min and max signed projection of (pts - origin) onto unit axis (GPU if available)."""
    axis = axis / (numpy.linalg.norm(axis) + 1e-12)
    if origin is None:
        origin = numpy.zeros(3, dtype=numpy.float64)
    proj = (pts - origin[None, :]) @ axis
    if torch is not None and torch.cuda.is_available():
        pt = torch.as_tensor(proj, device="cuda", dtype=torch.float32)
        lo = float(torch.nanmin(pt).item())
        hi = float(torch.nanmax(pt).item())
        return lo, hi
    return float(numpy.nanmin(proj)), float(numpy.nanmax(proj))


def table_plane_basis(
    plane_normal_cam: numpy.ndarray | None,
    plane_model: numpy.ndarray | None,
    pts: numpy.ndarray,
    camera_pose: numpy.ndarray,
) -> tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]:
    """
    z = table normal (points toward camera / upward in view); e1,e2 span the table plane.
    Returns (z_axis, e1, e2, plane_point).
    """
    if plane_normal_cam is not None and numpy.linalg.norm(plane_normal_cam) > 1e-9:
        z_axis = numpy.asarray(plane_normal_cam, dtype=numpy.float64).copy()
        z_axis /= numpy.linalg.norm(z_axis) + 1e-12
    else:
        z_axis = numpy.array([0.0, 0.0, 1.0], dtype=numpy.float64)

    # Prefer normal from plane equation if present
    if plane_model is not None and len(plane_model) >= 4:
        n = numpy.asarray(plane_model[:3], dtype=numpy.float64)
        nn = numpy.linalg.norm(n)
        if nn > 1e-9:
            z_axis = n / nn

    # Point on physical table plane
    if plane_model is not None and len(plane_model) >= 4:
        plane_point = plane_point_from_model(numpy.asarray(plane_model, dtype=numpy.float64))
    else:
        plane_point = numpy.median(pts, axis=0)

    # Flip so table normal matches +robot Z in camera frame (same idea as checkpoint22)
    R_cam_robot = camera_pose[:3, :3]
    z_robot_in_cam = R_cam_robot @ numpy.array([0.0, 0.0, 1.0], dtype=numpy.float64)
    z_robot_in_cam /= numpy.linalg.norm(z_robot_in_cam) + 1e-12
    if float(numpy.dot(z_axis, z_robot_in_cam)) < 0:
        z_axis = -z_axis

    tmp = numpy.array([1.0, 0.0, 0.0], dtype=numpy.float64)
    if abs(float(numpy.dot(tmp, z_axis))) > 0.9:
        tmp = numpy.array([0.0, 1.0, 0.0], dtype=numpy.float64)
    e1 = numpy.cross(z_axis, tmp)
    e1 /= numpy.linalg.norm(e1) + 1e-12
    e2 = numpy.cross(z_axis, e1)
    e2 /= numpy.linalg.norm(e2) + 1e-12
    return z_axis, e1, e2, plane_point


def get_cube_transform_refined(
    cube_pcd: o3d.geometry.PointCloud,
    camera_pose: numpy.ndarray,
    plane_normal_cam: numpy.ndarray | None,
    plane_model: numpy.ndarray | None,
) -> tuple[numpy.ndarray, numpy.ndarray] | None:
    """
    Cube pose: table-plane min-area rectangle for yaw + vertical extent for center.

    Inputs
    ------
    cube_pcd : open3d point cloud for one cube.
    camera_pose : 4x4 T_cam_robot.
    plane_normal_cam, plane_model : from RANSAC on full scene (optional).

    Outputs
    -------
    (t_robot_cube, t_cam_cube) or None.
    """
    if cube_pcd is None or len(cube_pcd.points) < 30:
        return None

    pts = numpy.asarray(cube_pcd.points, dtype=numpy.float64)
    z_axis, e1, e2, plane_point = table_plane_basis(plane_normal_cam, plane_model, pts, camera_pose)

    # Project onto table plane (foot of perpendicular from each point)
    n = z_axis
    dist_signed = (pts - plane_point[None, :]) @ n
    pts_on_plane = pts - dist_signed[:, None] * n[None, :]

    u = (pts_on_plane - plane_point[None, :]) @ e1
    v = (pts_on_plane - plane_point[None, :]) @ e2
    uv = numpy.stack([u, v], axis=1).astype(numpy.float32)
    if uv.shape[0] < 5:
        return None

    rect = cv2.minAreaRect(uv)
    (cu, cv), (_w, _h), _angle = rect
    box = cv2.boxPoints(rect).astype(numpy.float64)

    # In-plane center from rectangle (tight footprint)
    foot = plane_point + cu * e1 + cv * e2

    # Vertical center: midpoint of signed extent along table normal from plane_point
    lo, hi = gpu_extent_along_axis(pts, z_axis, plane_point)
    mid = 0.5 * (lo + hi)

    # Optional: blend with robust 3D median for slight denoising (in-plane already fixed)
    med3 = gpu_robust_center(pts)
    foot_blend = 0.85 * foot + 0.15 * (med3 - float((med3 - plane_point) @ n) * n)

    center_cam = foot_blend + mid * z_axis

    # x-axis along first edge of min-area box in 3D
    e01 = box[1] - box[0]
    norm2 = float(numpy.linalg.norm(e01))
    if norm2 < 1e-9:
        e01 = box[2] - box[1]
        norm2 = float(numpy.linalg.norm(e01))
    if norm2 < 1e-9:
        x_axis = e1
    else:
        du, dv = e01[0] / norm2, e01[1] / norm2
        x_axis = du * e1 + dv * e2
    x_axis = x_axis - numpy.dot(x_axis, z_axis) * z_axis
    x_axis /= numpy.linalg.norm(x_axis) + 1e-12
    y_axis = numpy.cross(z_axis, x_axis)
    y_axis /= numpy.linalg.norm(y_axis) + 1e-12

    R = orthonormalize_rotation(numpy.column_stack([x_axis, y_axis, z_axis]))

    t_cam_cube = numpy.eye(4, dtype=numpy.float64)
    t_cam_cube[:3, :3] = R
    t_cam_cube[:3, 3] = center_cam

    t_robot_cam = numpy.linalg.inv(camera_pose)
    t_robot_cube = t_robot_cam @ t_cam_cube
    return t_robot_cube, t_cam_cube


def get_cube_transforms(observation, camera_intrinsic, camera_pose):
    """
    Multi-cube poses with GPU mask + refined orientation/center.

    Parameters
    ----------
    observation : tuple
        (image, point_cloud) from ZED.
    camera_intrinsic : ndarray
        3x3 (passed for API compatibility; mask uses depth only).
    camera_pose : ndarray
        4x4 T_cam_robot.

    Returns
    -------
    list
        List of ``(t_robot_cube, t_cam_cube)`` (may contain None entries on failure).
    """
    del camera_intrinsic  # depth pipeline; intrinsics not required here

    image, point_cloud = observation
    if image is None or point_cloud is None:
        return []

    xyz = point_cloud[..., :3]
    mask = gpu_depth_mask(point_cloud)
    valid = numpy.isfinite(xyz).all(axis=-1)
    if mask is not None:
        keep = (mask > 0) & valid
    else:
        keep = valid

    pts = xyz[keep]
    if pts.shape[0] < 150:
        pts = xyz[valid]
    if pts.shape[0] < 100:
        return []

    valid_points_m, _scale = points_to_meters_open3d(pts)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(valid_points_m)

    cube_pcds, plane_n, plane_model, _msg = isolate_cube_cluster_open3d(pcd, num_cubes=STACK_HEIGHT_GOAL)
    if not cube_pcds:
        return []

    transforms = []
    for pcd_c in cube_pcds:
        transforms.append(get_cube_transform_refined(pcd_c, camera_pose, plane_n, plane_model))
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
    """Same role as checkpoint22: calibration + cube transforms + overlay."""
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

    cube_transforms = get_cube_transforms((cv_image, point_cloud), camera_intrinsic, t_cam_robot)
    if not cube_transforms:
        disp = draw_status_overlay(cv_image, ["orientation_RRC: no cubes"], (0, 0, 255))
        return None, disp

    disp = cv_image.copy()
    for transform in cube_transforms:
        if transform is None:
            continue
        _t_robot, t_cam_cube = transform
        if t_cam_cube is not None and numpy.isfinite(t_cam_cube).all():
            draw_pose_axes(disp, camera_intrinsic, t_cam_cube)

    lines = ["orientation_RRC: refined min-rect + GPU stats"]
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

            for i in range(min(STACK_HEIGHT_GOAL, len(cube_transforms))):
                pair = cube_transforms[i]
                if pair is None:
                    break
                t_robot_cube_i, _ = pair
                if t_robot_cube_i is None:
                    continue
                grasp_cube(arm, t_robot_cube_i)
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
