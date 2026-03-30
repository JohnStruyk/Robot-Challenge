import cv2
import numpy
import open3d as o3d
from xarm.wrapper import XArmAPI

from utils.zed_camera import ZedCamera
from utils.vis_utils import draw_pose_axes
from checkpoint0 import get_transform_camera_robot
from checkpoint1 import grasp_cube, place_cube, GRIPPER_LENGTH, robot_ip
from checkpoint4 import STACK_HEIGHT

# ---------- geometry helpers ----------

def scale_xyz_to_meters_frame(xyz):
    m = numpy.nanmax(numpy.abs(xyz))
    s = 0.001 if m > 50 else 1.0
    print(f"[scale_xyz] max abs={m:.3f}, scale={s}")
    return xyz * s, s

def segment_top_n_cubes_open3d(pcd, max_cubes=3, cube_min=0.02, cube_max=0.06):
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
            print(f"[segment] cluster {cid} rejected: too small ({idx.size})")
            continue

        c = pcd.select_by_index(idx)
        obb = c.get_oriented_bounding_box()
        ext = numpy.sort(numpy.asarray(obb.extent))
        center = numpy.asarray(obb.center)

        print(f"[segment] cluster {cid}: pts={idx.size}, ext={ext}, center={center}")

        maxd = float(ext[2])
        mind = float(ext[0])
        if not (cube_min <= maxd <= cube_max):
            print(f"    rejected: size out of range ({maxd:.3f})")
            continue

        compact = mind / maxd if maxd > 0 else 0.0
        if compact < 0.4:
            print(f"    rejected: not compact enough ({compact:.3f})")
            continue

        score = idx.size * compact
        print(f"    accepted: compact={compact:.3f}, score={score:.1f}")
        scored.append((score, c))

    print(f"[segment] scored cube-like clusters: {len(scored)}")
    if not scored:
        return [], "no cube-like clusters"

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

# ---------- detector ----------

class CubePoseDetector:
    def __init__(self, camera_intrinsic):
        self.camera_intrinsic = camera_intrinsic
        self.t_cam_robot = None

    def set_camera_pose(self, T):
        self.t_cam_robot = T
        print(f"[detector] set_camera_pose:\n{T}")

    def get_n_cubes(self, observation, max_cubes=3, cube_min=0.02, cube_max=0.06):
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

        clusters, msg = segment_top_n_cubes_open3d(
            pcd, max_cubes=max_cubes, cube_min=cube_min, cube_max=cube_max
        )
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

# ---------- staging + stacking ----------

def move_cube_to_staging_area(arm, T_robot_cube, x_fixed, y_fixed):
    T_target = numpy.copy(T_robot_cube)
    T_target[0, 3] = x_fixed
    T_target[1, 3] = y_fixed
    print(f"[stage] moving cube to staging XY=({x_fixed:.3f}, {y_fixed:.3f})")
    place_cube(arm, T_target)
    arm.stop_lite6_gripper()

def run_tower_sequence(arm, poses_robot):
    names = list(poses_robot.keys())
    print(f"[stack] cube order (bottom->top): {names}")

    X_STAGE = 230.1/1000
    Y_STAGE =  -305.5/1000

    staged_poses = {}
    for name in names:
        print(f"[stage] staging {name}")
        grasp_cube(arm, poses_robot[name])
        move_cube_to_staging_area(arm, poses_robot[name], X_STAGE, Y_STAGE)
        T_new = numpy.copy(poses_robot[name])
        T_new[0, 3] = X_STAGE
        T_new[1, 3] = Y_STAGE
        staged_poses[name] = T_new

    bottom_to_top = names
    base = staged_poses[bottom_to_top[0]]
    print(f"[stack] base pose z={base[2,3]}")

    for k, name in enumerate(bottom_to_top):
        print(f"[stack] placing {name} at level {k}")
        grasp_cube(arm, staged_poses[name])
        T = numpy.copy(base)
        T[2, 3] += k * STACK_HEIGHT
        print(f"[stack] placing at z={T[2,3]}")
        place_cube(arm, T)
        arm.stop_lite6_gripper()

    print("[stack] tower complete")

# ---------- preview (axes + k) ----------

def preview_n_cubes(image, cloud, K, detector, max_cubes, cube_min, cube_max):
    print("[preview] starting preview")
    if image is None or cloud is None:
        print("[preview] image or cloud is None")
        blank = numpy.zeros((720, 1280, 3), dtype=numpy.uint8)
        return {}, blank, False

    T = get_transform_camera_robot(image, K)
    print(f"[preview] T_cam_robot:\n{T}")
    detector.set_camera_pose(T)

    results, msg = detector.get_n_cubes(
        (image, cloud), max_cubes=max_cubes, cube_min=cube_min, cube_max=cube_max
    )
    print(f"[preview] get_n_cubes msg: {msg}")

    disp = image.copy()

    if results is None:
        cv2.putText(disp, msg, (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        return {}, disp, False

    poses_robot = {}
    for i, (T_robot, T_cam) in enumerate(results):
        name = f"cube_{i}"
        poses_robot[name] = T_robot
        draw_pose_axes(disp, K, T_cam)

    cv2.putText(disp, f"{len(results)} cubes detected", (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    cv2.putText(disp, "press k to stack", (30, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

    return poses_robot, disp, True

# ---------- main ----------

def main():
    NUM_CUBES = 6
    CUBE_MIN = 0.02
    CUBE_MAX = 0.06

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
        img, cloud, K, detector, NUM_CUBES, CUBE_MIN, CUBE_MAX
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
