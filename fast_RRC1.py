import cv2
import numpy
import open3d as o3d
from xarm.wrapper import XArmAPI

from utils.zed_camera import ZedCamera
from utils.vis_utils import draw_pose_axes
from checkpoint0 import get_transform_camera_robot
from checkpoint1 import grasp_cube, place_cube, GRIPPER_LENGTH, robot_ip
from checkpoint4 import STACK_HEIGHT

# ---------- project filtered point cloud to 2D mask ----------

def filtered_pcd_mask(image, pcd, K):
    """
    Create a 2D mask by projecting the filtered Open3D point cloud
    (after voxel + outlier + plane removal) into the RGB image.
    This shows EXACTLY what the cube detector sees.
    """
    H, W = image.shape[:2]
    mask = numpy.zeros((H, W, 3), dtype=numpy.uint8)

    if pcd is None or len(pcd.points) == 0:
        return mask

    pts = numpy.asarray(pcd.points)  # Nx3

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    X = pts[:, 0]
    Y = pts[:, 1]
    Z = pts[:, 2]

    valid = Z > 1e-6
    X = X[valid]
    Y = Y[valid]
    Z = Z[valid]

    u = (X * fx / Z + cx).astype(int)
    v = (Y * fy / Z + cy).astype(int)

    ok = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    u = u[ok]
    v = v[ok]

    mask[v, u] = (255, 255, 255)
    mask = cv2.medianBlur(mask, 5)
    return mask

# ---------- geometry helpers ----------

def scale_xyz_to_meters_frame(xyz):
    m = numpy.nanmax(numpy.abs(xyz))
    s = 0.001 if m > 50 else 1.0
    return xyz * s, s

def segment_top_n_cubes_open3d(pcd, max_cubes=3, cube_min=0.02, cube_max=0.06):
    pcd = pcd.voxel_down_sample(0.003)
    pcd, _ = pcd.remove_statistical_outlier(25, 2.0)

    if len(pcd.points) == 0:
        return [], "empty"

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

        maxd = float(ext[2])
        mind = float(ext[0])
        if not (cube_min <= maxd <= cube_max):
            continue

        compact = mind / maxd if maxd > 0 else 0.0
        if compact < 0.4:
            continue

        score = idx.size * compact
        scored.append((score, c))

    if not scored:
        return [], "no cube-like clusters"

    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in scored[:max_cubes]], "ok"

def camera_pose_from_cluster_pcd(cluster):
    obb = cluster.get_oriented_bounding_box()
    T = numpy.eye(4)
    T[:3, :3] = numpy.asarray(obb.R)
    T[:3, 3] = numpy.asarray(obb.center)
    return T

# ---------- detector ----------

class CubePoseDetector:
    def __init__(self, camera_intrinsic):
        self.camera_intrinsic = camera_intrinsic
        self.t_cam_robot = None

    def set_camera_pose(self, T):
        self.t_cam_robot = T

    def get_n_cubes(self, observation, max_cubes=3, cube_min=0.02, cube_max=0.06):
        image, cloud = observation
        if image is None or cloud is None:
            return None, "no data", None, None

        xyz = cloud[..., :3]
        xyz_m, _ = scale_xyz_to_meters_frame(xyz)
        finite = numpy.isfinite(xyz_m).all(axis=-1)
        pts = xyz_m[finite]

        if pts.shape[0] < 100:
            return None, "too few points", None, None

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts.astype(numpy.float64))

        clusters, msg = segment_top_n_cubes_open3d(
            pcd, max_cubes=max_cubes, cube_min=cube_min, cube_max=cube_max
        )
        if not clusters:
            return None, msg, pcd, None

        T_rc = numpy.linalg.inv(self.t_cam_robot)
        out = []
        for c in clusters:
            T_cam = camera_pose_from_cluster_pcd(c)
            T_robot = T_rc @ T_cam
            out.append((T_robot, T_cam))

        return out, "ok", pcd, None

# ---------- stacking (unchanged) ----------

def run_tower_sequence(arm, poses_robot):
    names = list(poses_robot.keys())

    X_STAGE = 230.1 / 1000.0
    Y_STAGE = -305.5 / 1000.0

    first = names[0]
    grasp_cube(arm, poses_robot[first])

    base = numpy.copy(poses_robot[first])
    base[0, 3] = X_STAGE
    base[1, 3] = Y_STAGE

    place_cube(arm, base)
    arm.stop_lite6_gripper()

    for k, name in enumerate(names[1:], start=1):
        grasp_cube(arm, poses_robot[name])
        T = numpy.copy(base)
        T[2, 3] += k * STACK_HEIGHT
        place_cube(arm, T)
        arm.stop_lite6_gripper()

# ---------- preview (mask + axes + k) ----------

def preview_n_cubes(image, cloud, K, detector, max_cubes, cube_min, cube_max):
    T = get_transform_camera_robot(image, K)
    detector.set_camera_pose(T)

    results, msg, pcd_filt, _ = detector.get_n_cubes(
        (image, cloud), max_cubes=max_cubes, cube_min=cube_min, cube_max=cube_max
    )

    # --- TRUE FILTERED MASK ---
    mask = filtered_pcd_mask(image, pcd_filt, K)
    cv2.imshow("cluster_mask", mask)
    cv2.waitKey(0)
    cv2.destroyWindow("cluster_mask")

    disp = image.copy()

    if results is None:
        cv2.putText(disp, msg, (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        return {}, disp, False

    poses_robot = {}
    for i, (T_robot, T_cam) in enumerate(results):
        poses_robot[f"cube_{i}"] = T_robot
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

    img = zed.image
    cloud = zed.point_cloud

    poses_robot, disp, ok = preview_n_cubes(
        img, cloud, K, detector, NUM_CUBES, CUBE_MIN, CUBE_MAX
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
