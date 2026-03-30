import os
import time

import cv2
import numpy
import open3d as o3d
from scipy.spatial.transform import Rotation
from xarm.wrapper import XArmAPI

from checkpoint1 import GRIPPER_LENGTH, robot_ip
from checkpoint4 import STACK_HEIGHT
from utils.vis_utils import draw_pose_axes
from utils.zed_camera import ZedCamera

try:
    import torch
except Exception:
    torch = None


# If you have a known camera->robot transform, save it as 4x4 float matrix here.
CAMERA_TO_ROBOT_PATH = "camera_to_robot.npy"
GPU_Z_MIN_M = 0.05
GPU_Z_MAX_M = 1.20
MAX_CUBES = 8
VOXEL_M = 0.0025
PLANE_DIST_M = 0.008
DBSCAN_EPS_M = 0.018
DBSCAN_MIN_PTS = 40
CUBE_MIN_M = 0.018
CUBE_MAX_M = 0.080
SAFE_Z_M = 0.22
GRASP_Z_OFFSET_M = 0.0001
PLACE_Z_OFFSET_M = 0.002
LIFT_Z_DELTA_M = 0.06
TOOL_ROLL_DEG = 180.0
TOOL_PITCH_DEG = 0.0
# Fast everywhere except final place descent.
ARM_SPEED_FAST = 3000
ARM_SPEED_PLACE = 140
GRIPPER_SETTLE_S = 0.30


def get_camera_to_robot_transform():
    """Load a fixed camera->robot transform; use identity if no file exists.

    Inputs: none.
    Outputs: 4x4 numpy array.
    """
    if os.path.exists(CAMERA_TO_ROBOT_PATH):
        T = numpy.load(CAMERA_TO_ROBOT_PATH)
        if T.shape == (4, 4):
            return T.astype(numpy.float64)
    return numpy.eye(4, dtype=numpy.float64)


def orthonormalize_rotation(R):
    """Project a 3x3 matrix onto SO(3).

    Inputs: R — 3x3 matrix.
    Outputs: valid 3x3 rotation matrix.
    """
    U, _, Vt = numpy.linalg.svd(R)
    Rn = U @ Vt
    if numpy.linalg.det(Rn) < 0:
        U[:, -1] *= -1.0
        Rn = U @ Vt
    return Rn


def gpu_depth_mask(point_cloud, z_min_m=GPU_Z_MIN_M, z_max_m=GPU_Z_MAX_M):
    """Build a depth mask with CUDA when available, CPU otherwise.

    Inputs: point_cloud — HxWx4 or HxWx3 cloud; z_min_m/z_max_m — valid depth band.
    Outputs: uint8 mask HxW where 255 is valid foreground candidate.
    """
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

    # Morphological close/open using pooling to denoise masks.
    x = mask.float().unsqueeze(0).unsqueeze(0)
    for _ in range(2):
        x = torch.nn.functional.max_pool2d(x, kernel_size=5, stride=1, padding=2)  # dilate
        x = 1.0 - torch.nn.functional.max_pool2d(1.0 - x, kernel_size=5, stride=1, padding=2)  # erode
    for _ in range(1):
        x = 1.0 - torch.nn.functional.max_pool2d(1.0 - x, kernel_size=3, stride=1, padding=1)  # erode
        x = torch.nn.functional.max_pool2d(x, kernel_size=3, stride=1, padding=1)  # dilate

    m = (x[0, 0] > 0.5).detach().cpu().numpy()
    return (m.astype(numpy.uint8) * 255)


def isolate_cube_clusters(masked_points_m):
    """Remove plane, cluster points, and keep cube-like clusters.

    Inputs: masked_points_m — Nx3 float points in meters.
    Outputs: (clusters, plane_normal_cam, msg).
    """
    if masked_points_m.shape[0] < 200:
        return [], None, "too few masked points"

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(masked_points_m.astype(numpy.float64))
    pcd = pcd.voxel_down_sample(VOXEL_M)
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=30, std_ratio=2.0)
    if len(pcd.points) < 150:
        return [], None, "too few points after filtering"

    plane_model, inliers = pcd.segment_plane(
        distance_threshold=PLANE_DIST_M,
        ransac_n=3,
        num_iterations=2500,
    )
    plane_n = numpy.asarray(plane_model[:3], dtype=numpy.float64)
    nrm = numpy.linalg.norm(plane_n)
    if nrm > 1e-9:
        plane_n /= nrm
    pcd_np = pcd.select_by_index(inliers, invert=True) if len(inliers) > 0 else pcd
    if len(pcd_np.points) < 120:
        return [], plane_n, "no objects above table"

    labels = numpy.asarray(
        pcd_np.cluster_dbscan(eps=DBSCAN_EPS_M, min_points=DBSCAN_MIN_PTS, print_progress=False)
    )
    if labels.size == 0 or labels.max() < 0:
        return [], plane_n, "dbscan found no clusters"

    scored = []
    for cid in range(int(labels.max()) + 1):
        idx = numpy.where(labels == cid)[0]
        if idx.size < DBSCAN_MIN_PTS:
            continue
        c = pcd_np.select_by_index(idx)
        obb = c.get_oriented_bounding_box()
        ext = numpy.sort(numpy.asarray(obb.extent))
        dmin = float(ext[0])
        dmax = float(ext[2])
        if not (CUBE_MIN_M <= dmax <= CUBE_MAX_M):
            continue
        compact = dmin / dmax if dmax > 1e-9 else 0.0
        if compact < 0.45:
            continue
        score = float(idx.size) * compact
        scored.append((score, c))

    if not scored:
        return [], plane_n, "no cube-like clusters"
    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in scored[:MAX_CUBES]], plane_n, "ok"


def stabilized_pose_from_cluster(cluster, plane_normal_cam):
    """Estimate stable pose from cluster with robust center and orientation.

    Inputs: cluster — open3d point cloud cluster; plane_normal_cam — 3-vector or None.
    Outputs: (T_cam_cube, cube_height_m).
    """
    pts = numpy.asarray(cluster.points)
    obb = cluster.get_oriented_bounding_box()
    center_med = numpy.median(pts, axis=0)
    center_obb = numpy.asarray(obb.center)
    center = 0.65 * center_med + 0.35 * center_obb

    R = orthonormalize_rotation(numpy.asarray(obb.R))
    if plane_normal_cam is not None and numpy.linalg.norm(plane_normal_cam) > 1e-9:
        n = plane_normal_cam / numpy.linalg.norm(plane_normal_cam)
        # Pick OBB axis most aligned with table normal as local z-axis for consistent yaw.
        axes = [R[:, 0], R[:, 1], R[:, 2]]
        best = int(numpy.argmax([abs(float(numpy.dot(a, n))) for a in axes]))
        z_axis = axes[best]
        if numpy.dot(z_axis, n) < 0:
            z_axis = -z_axis
        x_guess = axes[(best + 1) % 3]
        x_axis = x_guess - numpy.dot(x_guess, z_axis) * z_axis
        if numpy.linalg.norm(x_axis) < 1e-9:
            x_guess = axes[(best + 2) % 3]
            x_axis = x_guess - numpy.dot(x_guess, z_axis) * z_axis
        x_axis /= (numpy.linalg.norm(x_axis) + 1e-12)
        y_axis = numpy.cross(z_axis, x_axis)
        y_axis /= (numpy.linalg.norm(y_axis) + 1e-12)
        R = numpy.column_stack([x_axis, y_axis, z_axis])
        R = orthonormalize_rotation(R)

    ext = numpy.sort(numpy.asarray(obb.extent))
    h_m = float(ext[2])
    T = numpy.eye(4, dtype=numpy.float64)
    T[:3, :3] = R
    T[:3, 3] = center
    return T, h_m


class MaskCubeDetector:
    """High-accuracy non-AprilTag multi-cube detector using GPU mask + 3D fitting."""

    def __init__(self, camera_intrinsic, t_cam_robot):
        self.K = camera_intrinsic
        self.t_cam_robot = t_cam_robot

    def detect_all(self, observation, max_cubes=MAX_CUBES):
        """Detect all cube poses in one frame.

        Inputs: observation — (image, point_cloud); max_cubes — cap on returned cubes.
        Outputs: (list_of_dicts, debug_mask, message).
        """
        image, cloud = observation
        if image is None or cloud is None:
            return [], None, "missing image/cloud"

        mask = gpu_depth_mask(cloud)
        valid = mask > 0
        xyz = cloud[..., :3]
        finite = numpy.isfinite(xyz).all(axis=-1)
        keep = valid & finite
        pts = xyz[keep]
        if pts.shape[0] < 200:
            return [], mask, "too few valid masked points"

        # Convert to meters if needed.
        max_abs = float(numpy.nanmax(numpy.abs(pts)))
        scale = 0.001 if max_abs > 50.0 else 1.0
        pts_m = (pts * scale).astype(numpy.float64)

        clusters, plane_n, msg = isolate_cube_clusters(pts_m)
        if not clusters:
            return [], mask, msg

        t_robot_cam = numpy.linalg.inv(self.t_cam_robot)
        out = []
        for c in clusters[:max_cubes]:
            T_cam, h_m = stabilized_pose_from_cluster(c, plane_n)
            T_robot = t_robot_cam @ T_cam
            out.append(
                {
                    "t_robot": T_robot,
                    "t_cam": T_cam,
                    "height_m": h_m,
                    "source": "mask-geometry",
                }
            )
        return out, mask, "ok"


def make_stack_pose(base_pose, stack_index):
    """Build placement pose by stacking upward from base pose.

    Inputs: base_pose — 4x4 pose; stack_index — 0,1,2...
    Outputs: 4x4 pose for that stack level.
    """
    t = numpy.copy(base_pose)
    t[2, 3] = base_pose[2, 3] + stack_index * STACK_HEIGHT
    return t


def move_line(arm, x_mm, y_mm, z_mm, yaw_deg, speed):
    """Move robot TCP to one Cartesian point with explicit speed.

    Inputs: arm, xyz in mm, yaw in deg, speed in mm/s.
    Outputs: none.
    """
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


def grasp_cube_fast(arm, cube_pose):
    """Fast pick: open, approach, descend, grip, lift.

    Inputs: arm, cube_pose (4x4 meters).
    Outputs: none.
    """
    xyz = cube_pose[:3, 3]
    x_mm, y_mm, z_mm = (xyz * 1000.0).tolist()
    safe_z_mm = SAFE_Z_M * 1000.0
    grasp_z_mm = z_mm + (GRASP_Z_OFFSET_M * 1000.0)
    lift_z_mm = max(safe_z_mm, grasp_z_mm + (LIFT_Z_DELTA_M * 1000.0))
    cube_r = Rotation.from_matrix(cube_pose[:3, :3])
    _, _, cube_yaw_deg = cube_r.as_euler("xyz", degrees=True)

    arm.open_lite6_gripper()
    time.sleep(GRIPPER_SETTLE_S)
    move_line(arm, x_mm, y_mm, safe_z_mm, cube_yaw_deg, ARM_SPEED_FAST)
    move_line(arm, x_mm, y_mm, grasp_z_mm, cube_yaw_deg, ARM_SPEED_FAST)
    arm.close_lite6_gripper()
    time.sleep(GRIPPER_SETTLE_S)
    move_line(arm, x_mm, y_mm, lift_z_mm, cube_yaw_deg, ARM_SPEED_FAST)


def place_cube_fast(arm, cube_pose):
    """Place with fast travel and slow final descent for stability.

    Inputs: arm, cube_pose (4x4 meters).
    Outputs: none.
    """
    xyz = cube_pose[:3, 3]
    x_mm, y_mm, z_mm = (xyz * 1000.0).tolist()
    safe_z_mm = SAFE_Z_M * 1000.0
    place_z_mm = z_mm + (PLACE_Z_OFFSET_M * 1000.0)
    lift_z_mm = max(safe_z_mm, place_z_mm + (LIFT_Z_DELTA_M * 1000.0))
    cube_r = Rotation.from_matrix(cube_pose[:3, :3])
    _, _, cube_yaw_deg = cube_r.as_euler("xyz", degrees=True)

    move_line(arm, x_mm, y_mm, safe_z_mm, cube_yaw_deg, ARM_SPEED_FAST)
    move_line(arm, x_mm, y_mm, place_z_mm, cube_yaw_deg, ARM_SPEED_PLACE)
    arm.open_lite6_gripper()
    time.sleep(GRIPPER_SETTLE_S)
    move_line(arm, x_mm, y_mm, lift_z_mm, cube_yaw_deg, ARM_SPEED_FAST)


def run_tower_sequence(arm, detected):
    """Stack detected cubes at a fixed stage XY, sorted by proximity.

    Inputs: arm — xArm API; detected — list of detection dicts.
    Outputs: number of cubes placed.
    """
    if not detected:
        return 0

    # Closest cubes first.
    detected = sorted(detected, key=lambda d: float(numpy.linalg.norm(d["t_robot"][:3, 3])))
    X_STAGE = 230.1 / 1000.0
    Y_STAGE = -305.5 / 1000.0

    first = detected[0]["t_robot"]
    grasp_cube_fast(arm, first)
    base = numpy.copy(first)
    base[0, 3] = X_STAGE
    base[1, 3] = Y_STAGE
    place_cube_fast(arm, base)
    arm.stop_lite6_gripper()
    placed = 1

    for k, d in enumerate(detected[1:], start=1):
        grasp_cube_fast(arm, d["t_robot"])
        tgt = make_stack_pose(base, k)
        place_cube_fast(arm, tgt)
        arm.stop_lite6_gripper()
        placed += 1
    return placed


def preview_detection(image, cloud, detector, max_cubes):
    """Render mask and pose axes for all detected cubes.

    Inputs: image/cloud — ZED outputs; detector — MaskCubeDetector; max_cubes — cap.
    Outputs: (detected_list, display_image, key_hint_ok).
    """
    detected, mask, msg = detector.detect_all((image, cloud), max_cubes=max_cubes)
    disp = image.copy()

    if mask is not None:
        mask_vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        cv2.imshow("mask_RRC depth-mask", mask_vis)

    if not detected:
        cv2.putText(
            disp,
            f"detect failed: {msg}",
            (30, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 255),
            2,
        )
        return [], disp, False

    for i, d in enumerate(detected):
        draw_pose_axes(disp, detector.K, d["t_cam"])
        xyz = d["t_robot"][:3, 3]
        cv2.putText(
            disp,
            f"cube{i}: x={xyz[0]:.3f} y={xyz[1]:.3f} z={xyz[2]:.3f}",
            (20, 60 + i * 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

    cv2.putText(disp, f"{len(detected)} cubes detected", (20, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(disp, "press k to stack", (20, 430), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    return detected, disp, True


def main():
    """Run mask-based non-AprilTag cube detection and optional stacking."""
    zed = ZedCamera()
    K = zed.camera_intrinsic
    t_cam_robot = get_camera_to_robot_transform()
    detector = MaskCubeDetector(K, t_cam_robot)

    arm = XArmAPI(robot_ip)
    arm.connect()
    arm.motion_enable(True)
    arm.set_tcp_offset([0, 0, GRIPPER_LENGTH, 0, 0, 0])
    arm.set_mode(0)
    arm.set_state(0)
    arm.move_gohome(wait=True)
    time.sleep(0.5)

    try:
        img = zed.image
        cloud = zed.point_cloud
        detected, disp, ok = preview_detection(img, cloud, detector, MAX_CUBES)

        cv2.imshow("mask_RRC preview", disp)
        key = cv2.waitKey(0)
        cv2.destroyAllWindows()

        if ok and key == ord("k"):
            n = run_tower_sequence(arm, detected)
            print(f"mask_RRC: placed {n} cubes.")
    finally:
        arm.stop_lite6_gripper()
        arm.move_gohome(wait=True)
        arm.disconnect()
        zed.close()


if __name__ == "__main__":
    main()
