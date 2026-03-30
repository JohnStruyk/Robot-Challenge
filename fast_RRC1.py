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
    print(f"[scale_xyz] max abs={m:.3f}, scale={s}")
    return xyz * s, s

def segment_top_n_cubes_open3d(pcd, max_cubes=3):
    print(f"[segment] input points: {len(pcd.points)}")
    pcd = pcd.voxel_down_sample(0.003)
    print(f"[segment] after voxel: {len(pcd.points)}")
    pcd, _ = pcd.remove_statistical_outlier(25, 2.0)
    print(f"[segment] after outlier: {len(pcd.points)}")

    if len(pcd.points) == 0:
        print("[segment] no points after filtering")
        return [], "empty"

    plane, inliers = pcd.segment_plane(0.01, 3, 1500)
    print(f"[segment] plane inliers: {len(inliers)} / {len(pcd.points)}")
    if len(inliers) > 0.1 * len(pcd.points):
        pcd = pcd.select_by_index(inliers, invert=True)
        print(f"[segment] after plane removal: {len(pcd.points)}")

    labels = numpy.asarray(pcd.cluster_dbscan(0.02, 25))
    if labels.size == 0:
        print("[segment] DBSCAN labels empty")
        return [], "no clusters"
    print(f"[segment] DBSCAN labels: min={labels.min()}, max={labels.max()}")

    if labels.max() < 0:
        print("[segment] no valid cluster labels")
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
        compact = mind / maxd if maxd > 0 else 0.0
        score = idx.size * compact
        scored.append((score, c))
        print(f"[segment] cluster {cid}: pts={idx.size}, ext={ext}, score={score:.3f}")

    print(f"[segment] scored clusters: {len(scored)}")
    if not scored:
        return [], "no good clusters"

    scored.sort(key=lambda x: x[0], reverse=True)
    top = [c for _, c in scored[:max_cubes]]
    print(f"[segment] returning {len(top)} clusters (max_cubes={max_cubes})")
    return top, "ok"

def camera_pose_from_cluster_pcd(cluster):
    obb = cluster.get_oriented_bounding_box()
    T = numpy.eye(4)
    T[:3, :3] = numpy.asarray(obb.R)
    T[:3, 3] = numpy.asarray(obb.center)
    print(f"[pose] center={T[:3,3]}")
    return T

# --- detector ---

class CubePoseDetector:
    def __init__(self, camera_intrinsic):
        self.camera_intrinsic = camera_intrinsic
        self.t_cam_robot = None

    def set_camera_pose(self, T):
        self.t_cam_robot = T
        print(f"[detector] set_camera_pose:\n{T}")

    def get_n_cubes(self, observation, max_cubes=3):
        image, cloud = observation
        if image is None or cloud is None:
            print("[detector] image or cloud is None")
            return None, "no data"

        xyz = cloud[..., :3]
        print(f"[detector] raw xyz shape: {xyz.shape}")
        xyz_m, _ = scale_xyz_to_meters_frame(xyz)
        finite = numpy.isfinite(xyz_m).all(axis=-1)
        print(f"[detector] finite points: {finite.sum()} / {finite.size}")
        pts = xyz_m[finite]

        if pts.shape[0] < 100:
            print(f"[detector] too few finite pts: {pts.shape[0]}")
            return None, "too few points"

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts.astype(numpy.float64))

        clusters, msg = segment_top_n_cubes_open3d(pcd, max_cubes)
        print(f"[detector] segment msg: {msg}")
        if not clusters:
            return None, "no cubes"

        T_rc = numpy.linalg.inv(self.t_cam_robot)
        out = []
        for i, c in enumerate(clusters):
            T_cam = camera_pose_from_cluster_pcd(c)
            T_robot = T_rc @ T_cam
            print(f"[detector] cube {i} robot xyz={T_robot[:3,3]}")
            out.append((T_robot, T_cam))
        return out, "ok"

# --- preview ---

def preview_n_cubes(image, cloud, K, detector, max_cubes):
    print("[preview] starting preview")
    if image is None or cloud is None:
        print("[preview] image or cloud is None")
        blank = numpy.zeros((720, 1280, 3), dtype=numpy.uint8)
        disp = draw_status_overlay(blank, ["no image/cloud"], (0, 0, 255))
        return {}, disp, False

    T = get_transform_camera_robot(image, K)
    print(f"[preview] T_cam_robot:\n{T}")
    detector.set_camera_pose(T)

    results, msg = detector.get_n_cubes((image, cloud), max_cubes)
    print(f"[preview] get_n_cubes msg: {msg}")
    if results is None:
        disp = draw_status_overlay(image, [msg], (0, 0, 255))
        return {}, disp, False

    poses_robot = {f"cube_{i}": r[0] for i, r in enumerate(results)}
    print(f"[preview] poses_robot keys: {list(poses_robot.keys())}")

    disp = image.copy()
    for i, (_, T_cam) in enumerate(results):
        print(f"[preview] drawing cube {i}")
        draw_pose_axes(disp, K, T_cam)
    disp = draw_status_overlay(disp, [f"{len(results)} cubes detected", "press k"], (0, 220, 0))
    return poses_robot, disp, True

# --- stacking ---

def run_tower_sequence(arm, poses_robot):
    names = list(poses_robot.keys())
    print(f"[stack] cube order (bottom->top): {names}")
    bottom_to_top = names
    base = poses_robot[bottom_to_top[0]]
    print(f"[stack] base pose z={base[2,3]}")

    for k, name in enumerate(bottom_to_top):
        print(f"[stack] level {k}, cube={name}")
        grasp_cube(arm, poses_robot[name])
        T = numpy.copy(base)
        T[2, 3] += k * STACK_HEIGHT
        print(f"[stack] placing at z={T[2,3]}")
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

    print("[main] grabbing initial image/point cloud")
    img = zed.image
    cloud = zed.point_cloud
    print(f"[main] img shape={None if img is None else img.shape}, cloud shape={None if cloud is None else cloud.shape}")

    poses_robot, disp, ok = preview_n_cubes(
        img, cloud, K, detector, NUM_CUBES
    )

    cv2.imshow("preview", disp)
    key = cv2.waitKey(0)
    cv2.destroyAllWindows()
    print(f"[main] key pressed: {key} ({chr(key) if key != -1 else 'none'})")

    if ok and key == ord("k"):
        print("[main] running tower sequence")
        run_tower_sequence(arm, poses_robot)
    else:
        print("[main] skipping tower sequence (ok=", ok, ")")

    arm.stop_lite6_gripper()
    arm.move_gohome(wait=True)
    arm.disconnect()
    zed.close()

if __name__ == "__main__":
    main()
