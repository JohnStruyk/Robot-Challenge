import cv2
import numpy
import time
import open3d as o3d
from xarm.wrapper import XArmAPI

from utils.vis_utils import draw_pose_axes
from utils.zed_camera import ZedCamera
from checkpoint0 import get_transform_camera_robot
from checkpoint1 import grasp_cube, place_cube, GRIPPER_LENGTH, robot_ip
from checkpoint4 import STACK_HEIGHT
from checkpoint6 import draw_status_overlay

# --- geometry helpers ---

def scale_xyz_to_meters_frame(xyz):
    m = numpy.nanmax(numpy.abs(xyz))
    s = 0.001 if m > 50 else 1.0
    return xyz * s, s

def segment_top_n_cubes_open3d(pcd, max_cubes=3):
    pcd = pcd.voxel_down_sample(0.003)
    pcd, _ = pcd.remove_statistical_outlier(25, 2.0)
    plane, inliers = pcd.segment_plane(0.01, 3, 1500)
    if len(inliers) > 0.1 * len(pcd.points):
        pcd = pcd.select_by_index(inliers, invert=True)

    labels = numpy.asarray(pcd.cluster_dbscan(0.02, 25))
    if labels.size == 0 or labels.max() < 0:
        return [], "no clusters"

    scored = []
    for cid in range(labels.max() + 1):
        idx = numpy.where(labels == cid)[0]
        if idx.size < 30:
            continue
        c = pcd.select_by_index(idx)
        obb = c.get_oriented_bounding_box()
        ext = numpy.sort(numpy.asarray(obb.extent))
        if ext[2] < 1e-9:
            continue
        maxd, mind = float(ext[2]), float(ext[0])
        compact = mind / maxd
        score = idx.size * compact
        scored.append((score, c))

    if not scored:
        return [], "no good clusters"

    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in scored[:max_cubes]], "ok"

def camera_pose_from_cluster_pcd(cluster):
    obb = cluster.get_oriented_bounding_box()
    T = numpy.eye(4)
    T[:3, :3] = numpy.asarray(obb.R)
    T[:3, 3] = numpy.asarray(obb.center)
    return T

# --- detector ---

class CubePoseDetector:
    def __init__(self, camera_intrinsic):
        self.camera_intrinsic = camera_intrinsic
        self.t_cam_robot = None

    def set_camera_pose(self, T):
        self.t_cam_robot = T

    def get_n_cubes(self, observation, max_cubes=3):
        image, cloud = observation
        xyz = cloud[..., :3]
        xyz_m, _ = scale_xyz_to_meters_frame(xyz)
        finite = numpy.isfinite(xyz_m).all(axis=-1)
        pts = xyz_m[finite]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts.astype(numpy.float64))

        clusters, _ = segment_top_n_cubes_open3d(pcd, max_cubes)
        if not clusters:
            return None, "no cubes"

        T_rc = numpy.linalg.inv(self.t_cam_robot)
        out = []
        for c in clusters:
            T_cam = camera_pose_from_cluster_pcd(c)
            T_robot = T_rc @ T_cam
            out.append((T_robot, T_cam))
        return out, "ok"

# --- preview ---

def preview_n_cubes(image, cloud, K, detector, max_cubes):
    T = get_transform_camera_robot(image, K)
    detector.set_camera_pose(T)
    results, msg = detector.get_n_cubes((image, cloud), max_cubes)
    if results is None:
        disp = draw_status_overlay(image, [msg], (0, 0, 255))
        return {}, disp, False

    poses_robot = {f"cube_{i}": r[0] for i, r in enumerate(results)}

    disp = image.copy()
    for _, T_cam in results:
        draw_pose_axes(disp, K, T_cam)
    disp = draw_status_overlay(disp, [f"{len(results)} cubes detected", "press k"], (0, 220, 0))
    return poses_robot, disp, True

# --- stacking ---

def run_tower_sequence(arm, poses_robot):
    names = list(poses_robot.keys())
    bottom_to_top = names
    base = poses_robot[bottom_to_top[0]]

    for k, name in enumerate(bottom_to_top):
        grasp_cube(arm, poses_robot[name])
        T = numpy.copy(base)
        T[2, 3] += k * STACK_HEIGHT
        place_cube(arm, T)
        arm.stop_lite6_gripper()

# --- main ---

def main():
    NUM_CUBES = 6

    zed = ZedCamera()
    K = zed.camera_intrinsic
    detector = CubePoseDetector(K)

    arm = XArmAPI(robot_ip)
    arm.connect()
    arm.motion_enable(True)
    arm.set_tcp_offset([0, 0, GRIPPER_LENGTH, 0, 0, 0])
    arm.set_mode(0)
    arm.set_state(0)
    arm.move_gohome(wait=True)

    poses_robot, disp, ok = preview_n_cubes(
        zed.image, zed.point_cloud, K, detector, NUM_CUBES
    )

    cv2.imshow("preview", disp)
    key = cv2.waitKey(0)
    cv2.destroyAllWindows()

    if ok and key == ord("k"):
        run_tower_sequence(arm, poses_robot)

    arm.stop_lite6_gripper()
    arm.move_gohome(wait=True)
    arm.disconnect()
    zed.close()

if __name__ == "__main__":
    main()
