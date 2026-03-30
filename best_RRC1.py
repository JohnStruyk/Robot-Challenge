import cv2
import numpy
import open3d as o3d
from scipy.spatial.transform import Rotation
from xarm.wrapper import XArmAPI
import time

from checkpoint0 import get_transform_camera_robot
from checkpoint1 import GRIPPER_LENGTH, robot_ip
from checkpoint4 import STACK_HEIGHT
from utils.vis_utils import draw_pose_axes
from utils.zed_camera import ZedCamera


# Speed profile from mask_RRC.py: fast except final place descent.
ARM_SPEED_FAST = 3000
ARM_SPEED_PLACE = 140
ARM_SPEED_GRASP_DOWN = 220
GRIPPER_SETTLE_S = 0.30
RELEASE_WAIT_S = 0.80

SAFE_Z_M = 0.22
GRASP_Z_OFFSET_M = 0.0001
PLACE_Z_OFFSET_M = 0.002
LIFT_Z_DELTA_M = 0.06
TOOL_ROLL_DEG = 180.0
TOOL_PITCH_DEG = 0.0


def filtered_pcd_mask(image, pcd, K):
    """Project filtered cloud back to image for a truthful detector mask preview.

    Inputs: image, filtered point cloud (meters), camera intrinsics.
    Outputs: BGR mask image.
    """
    h, w = image.shape[:2]
    mask = numpy.zeros((h, w, 3), dtype=numpy.uint8)
    if pcd is None or len(pcd.points) == 0:
        return mask

    pts = numpy.asarray(pcd.points)
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    x = pts[:, 0]
    y = pts[:, 1]
    z = pts[:, 2]
    valid = z > 1e-6
    x = x[valid]
    y = y[valid]
    z = z[valid]
    u = (x * fx / z + cx).astype(int)
    v = (y * fy / z + cy).astype(int)
    ok = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    mask[v[ok], u[ok]] = (255, 255, 255)
    mask = cv2.medianBlur(mask, 5)
    return mask


def points_to_meters_open3d(xyz):
    """Scale XYZ to meters if cloud appears to be in millimeters.

    Inputs: xyz (Nx3 or HxWx3).
    Outputs: (xyz_meters, scale_used).
    """
    if xyz.size == 0:
        return xyz, 1.0
    max_abs = float(numpy.nanmax(numpy.abs(xyz)))
    scale = 0.001 if max_abs > 50.0 else 1.0
    return (xyz * scale).astype(numpy.float64), scale


def isolate_cube_cluster_open3d(pcd, num_cubes):
    """Checkpoint12-style cube segmentation with score and largest-cluster fallback.

    Inputs: pcd (Open3D cloud), num_cubes (max cubes to return).
    Outputs: (list_of_clusters or None, message).
    """
    if len(pcd.points) < 150:
        return None, "too few points after NaN filter"

    pcd = pcd.voxel_down_sample(voxel_size=0.003)
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=25, std_ratio=2.0)
    if len(pcd.points) < 100:
        return None, "too few points after voxel/outlier"

    plane_model, inliers = pcd.segment_plane(distance_threshold=0.010, ransac_n=3, num_iterations=1500)
    plane_n = numpy.asarray(plane_model[:3], dtype=numpy.float64)
    nrm = numpy.linalg.norm(plane_n)
    if nrm > 1e-9:
        plane_n /= nrm
    else:
        plane_n = None
    if len(inliers) > 0 and len(inliers) > 0.12 * len(pcd.points):
        pcd = pcd.select_by_index(inliers, invert=True)
    if len(pcd.points) < 80:
        return None, "nothing left after plane removal"

    labels = numpy.asarray(pcd.cluster_dbscan(eps=0.020, min_points=25, print_progress=False))
    max_label = int(labels.max()) if labels.size else -1
    if max_label < 0:
        return None, "DBSCAN found no clusters"

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
    if scored_clusters:
        return [c for _, c in scored_clusters[:num_cubes]], plane_n, f"{min(num_cubes, len(scored_clusters))} cube-like clusters"

    fallback_clusters.sort(key=lambda x: x[0], reverse=True)
    top_clusters = [c for _, c in fallback_clusters[:num_cubes] if len(c.points) >= 40]
    if top_clusters:
        return top_clusters, plane_n, f"fallback: {len(top_clusters)} largest clusters"
    return None, plane_n, "no cluster passed filters"


def get_cube_transform(cube_pcd, camera_pose, plane_normal_cam):
    """Compute (robot, camera) cube transforms from one segmented cluster.

    Inputs: cluster cloud, camera pose 4x4.
    Outputs: (t_robot_cube, t_cam_cube) or None.
    """
    if cube_pcd is None or len(cube_pcd.points) < 30:
        return None
    obb = cube_pcd.get_oriented_bounding_box()
    pts = numpy.asarray(cube_pcd.points)
    center_med = numpy.median(pts, axis=0)
    center_obb = numpy.asarray(obb.center)
    center = 0.75 * center_med + 0.25 * center_obb

    R_obb = numpy.asarray(obb.R)
    if plane_normal_cam is not None and numpy.linalg.norm(plane_normal_cam) > 1e-9:
        z_axis = plane_normal_cam / numpy.linalg.norm(plane_normal_cam)
    else:
        axes = [R_obb[:, 0], R_obb[:, 1], R_obb[:, 2]]
        z_axis = max(axes, key=lambda a: abs(float(a[2])))
        if z_axis[2] < 0:
            z_axis = -z_axis

    q = pts - center[None, :]
    q_plane = q - (q @ z_axis)[:, None] * z_axis[None, :]
    if q_plane.shape[0] >= 3:
        cov = q_plane.T @ q_plane / max(q_plane.shape[0] - 1, 1)
        w, v = numpy.linalg.eigh(cov)
        x_axis = v[:, int(numpy.argmax(w))]
    else:
        x_axis = R_obb[:, 0]
    x_axis = x_axis - numpy.dot(x_axis, z_axis) * z_axis
    if numpy.linalg.norm(x_axis) < 1e-9:
        tmp = numpy.array([1.0, 0.0, 0.0], dtype=numpy.float64)
        if abs(float(numpy.dot(tmp, z_axis))) > 0.9:
            tmp = numpy.array([0.0, 1.0, 0.0], dtype=numpy.float64)
        x_axis = tmp - numpy.dot(tmp, z_axis) * z_axis
    x_axis /= (numpy.linalg.norm(x_axis) + 1e-12)
    y_axis = numpy.cross(z_axis, x_axis)
    y_axis /= (numpy.linalg.norm(y_axis) + 1e-12)
    R = numpy.column_stack([x_axis, y_axis, z_axis])
    # SO(3) projection
    u, _, vt = numpy.linalg.svd(R)
    R = u @ vt
    if numpy.linalg.det(R) < 0:
        u[:, -1] *= -1.0
        R = u @ vt

    t_cam_cube = numpy.eye(4)
    t_cam_cube[:3, :3] = R
    t_cam_cube[:3, 3] = center
    t_robot_cam = numpy.linalg.inv(camera_pose)
    t_robot_cube = t_robot_cam @ t_cam_cube
    return t_robot_cube, t_cam_cube


class CubePoseDetector:
    """Checkpoint12 vision wrapper for multi-cube detection."""

    def __init__(self, camera_intrinsic):
        self.camera_intrinsic = camera_intrinsic
        self.t_cam_robot = None

    def set_camera_pose(self, t_cam_robot):
        self.t_cam_robot = t_cam_robot

    def get_n_cubes(self, observation, max_cubes=14):
        """Detect up to N cubes from one frame.

        Inputs: observation=(image, cloud), max_cubes.
        Outputs: (list[(t_robot,t_cam)], msg, filtered_pcd).
        """
        image, cloud = observation
        if image is None or cloud is None:
            return None, "no data", None

        xyz = cloud[..., :3]
        xyz_m, _ = points_to_meters_open3d(xyz)
        finite = numpy.isfinite(xyz_m).all(axis=-1)
        pts = xyz_m[finite]
        if pts.shape[0] < 100:
            return None, "too few points", None

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts.astype(numpy.float64))

        clusters, plane_n, msg = isolate_cube_cluster_open3d(pcd, num_cubes=max_cubes)
        if not clusters:
            return None, msg, pcd

        out = []
        for c in clusters:
            tr = get_cube_transform(c, self.t_cam_robot, plane_n)
            if tr is not None:
                out.append(tr)
        if not out:
            return None, "no valid transforms", pcd
        return out, msg, pcd


def move_line(arm, x_mm, y_mm, z_mm, yaw_deg, speed):
    """Move arm with explicit speed.

    Inputs: arm, xyz(mm), yaw(deg), speed(mm/s).
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
    """Fast grasp: fast approach/descend/lift.

    Inputs: arm, cube_pose(4x4 meters).
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
    # Slower final pickup descent for reliable alignment.
    move_line(arm, x_mm, y_mm, grasp_z_mm, cube_yaw_deg, ARM_SPEED_GRASP_DOWN)
    arm.close_lite6_gripper()
    time.sleep(GRIPPER_SETTLE_S)
    move_line(arm, x_mm, y_mm, lift_z_mm, cube_yaw_deg, ARM_SPEED_FAST)


def place_cube_fast(arm, cube_pose):
    """Place with fast travel and slow final descent.

    Inputs: arm, cube_pose(4x4 meters).
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
    time.sleep(RELEASE_WAIT_S)
    move_line(arm, x_mm, y_mm, lift_z_mm, cube_yaw_deg, ARM_SPEED_FAST)


def stop_arm_now(arm):
    """Immediately stop the arm and gripper when user presses 't'."""
    try:
        arm.stop_lite6_gripper()
    except Exception:
        pass
    if hasattr(arm, "emergency_stop"):
        try:
            arm.emergency_stop()
            return
        except Exception:
            pass
    try:
        arm.set_state(4)
    except Exception:
        pass


def run_tower_sequence(arm, poses_robot):
    """Fast_RRC1 movement logic: stage first cube, then stack others on top.

    Inputs: arm, poses_robot dict.
    Outputs: none.
    """
    names = list(poses_robot.keys())
    if not names:
        return

    x_stage = 230.1 / 1000.0
    y_stage = -305.5 / 1000.0

    first = names[0]
    grasp_cube_fast(arm, poses_robot[first])
    base = numpy.copy(poses_robot[first])
    base[0, 3] = x_stage
    base[1, 3] = y_stage
    place_cube_fast(arm, base)
    arm.stop_lite6_gripper()

    for k, name in enumerate(names[1:], start=1):
        if cv2.waitKey(1) == ord("t"):
            stop_arm_now(arm)
            return
        grasp_cube_fast(arm, poses_robot[name])
        tgt = numpy.copy(base)
        tgt[2, 3] += k * STACK_HEIGHT
        place_cube_fast(arm, tgt)
        arm.stop_lite6_gripper()


def preview_n_cubes(image, cloud, k_mat, detector, max_cubes):
    """Preview detections with projected mask + axes.

    Inputs: image, cloud, K, detector, max_cubes.
    Outputs: (poses_robot_dict, display_image, ok_flag).
    """
    t_cam_robot = get_transform_camera_robot(image, k_mat)
    detector.set_camera_pose(t_cam_robot)
    results, msg, pcd_filt = detector.get_n_cubes((image, cloud), max_cubes=max_cubes)

    mask = filtered_pcd_mask(image, pcd_filt, k_mat)
    cv2.imshow("cluster_mask", mask)

    disp = image.copy()
    if results is None:
        cv2.putText(disp, msg, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        return {}, disp, False

    poses_robot = {}
    for i, (t_robot, t_cam) in enumerate(results):
        poses_robot[f"cube_{i}"] = t_robot
        draw_pose_axes(disp, k_mat, t_cam)

    cv2.putText(disp, f"{len(results)} cubes detected", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    cv2.putText(disp, "press k to stack", (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    return poses_robot, disp, True


def main():
    """Run best RRC1: checkpoint12 vision + fast_RRC1 flow + mask_RRC speeds."""
    num_cubes = 14
    zed = ZedCamera()
    k_mat = zed.camera_intrinsic
    detector = CubePoseDetector(k_mat)

    arm = XArmAPI(robot_ip)
    arm.connect()
    arm.motion_enable(True)
    arm.set_tcp_offset([0, 0, GRIPPER_LENGTH, 0, 0, 0])
    arm.set_mode(0)
    arm.set_state(0)
    arm.move_gohome(wait=True)

    img = zed.image
    cloud = zed.point_cloud
    poses_robot, disp, ok = preview_n_cubes(img, cloud, k_mat, detector, num_cubes)

    cv2.imshow("best_RRC1 preview", disp)
    key = cv2.waitKey(0)
    cv2.destroyAllWindows()

    if key == ord("t"):
        stop_arm_now(arm)
        arm.disconnect()
        zed.close()
        return

    if ok and key == ord("k"):
        run_tower_sequence(arm, poses_robot)

    arm.stop_lite6_gripper()
    arm.move_gohome(wait=True)
    arm.disconnect()
    zed.close()


if __name__ == "__main__":
    main()
