import cv2, numpy, time
import open3d as o3d
from xarm.wrapper import XArmAPI

from utils.vis_utils import draw_pose_axes
from utils.zed_camera import ZedCamera
from checkpoint0 import get_transform_camera_robot
from checkpoint1 import grasp_cube, place_cube, GRIPPER_LENGTH, robot_ip


def isolate_cube_cluster_open3d(pcd: o3d.geometry.PointCloud):
    """
    Segment a single tabletop cube without using image color thresholds.

    Uses Open3D: voxel downsample, outlier removal, RANSAC plane removal (table),
    then DBSCAN clustering and picks the cluster whose oriented bounding box is
    most cube-like (similar edge lengths, size in a plausible range).

    Parameters
    ----------
    pcd : open3d.geometry.PointCloud
        Points in camera frame (same units as ZED depth, typically meters).

    Returns
    -------
    open3d.geometry.PointCloud or None
        Point cloud of the chosen cluster, or None if segmentation fails.
    """
    if len(pcd.points) < 200:
        return None

    pcd = pcd.voxel_down_sample(voxel_size=0.004)
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=30, std_ratio=2.0)

    if len(pcd.points) < 150:
        return None

    # Remove dominant plane (table / large surfaces).
    plane_model, inliers = pcd.segment_plane(
        distance_threshold=0.012,
        ransac_n=3,
        num_iterations=1000,
    )
    if len(inliers) > 0 and len(inliers) > 0.15 * len(pcd.points):
        pcd = pcd.select_by_index(inliers, invert=True)

    if len(pcd.points) < 100:
        return None

    labels = numpy.asarray(
        pcd.cluster_dbscan(eps=0.025, min_points=35, print_progress=False)
    )
    max_label = int(labels.max()) if labels.size else -1
    if max_label < 0:
        return None

    best_cluster = None
    best_score = -1.0
    for cid in range(max_label + 1):
        idx = numpy.where(labels == cid)[0]
        if idx.size < 60:
            continue
        cluster = pcd.select_by_index(idx)
        obb = cluster.get_oriented_bounding_box()
        ext = numpy.sort(numpy.asarray(obb.extent))
        if ext[2] < 1e-6:
            continue
        # Challenge cubes are ~25 mm; allow a band without hardcoding color.
        max_dim = float(ext[2])
        min_dim = float(ext[0])
        if max_dim < 0.010 or max_dim > 0.070:
            continue
        compact = min_dim / max_dim
        score = float(idx.size) * compact * (1.0 if 0.3 < compact < 1.0 else 0.25)
        if score > best_score:
            best_score = score
            best_cluster = cluster

    return best_cluster


def get_transform_cube(observation, camera_intrinsic, camera_pose):
    """
    Estimate cube pose from a single image + dense point cloud using Open3D geometry only.

    The image is not used for color thresholds; the RGB image is still part of the
    observation tuple for API compatibility with the course template.

    Parameters
    ----------
    observation : list or tuple
        ``[image, point_cloud]`` with ``point_cloud[..., :3]`` in camera frame.
    camera_intrinsic : numpy.ndarray
        Unused here but kept for template compatibility.
    camera_pose : numpy.ndarray
        4x4 from checkpoint0 (robot/world -> camera).

    Returns
    -------
    tuple or None
        ``(t_robot_cube, t_cam_cube)`` or None.
    """
    image, point_cloud = observation
    if image is None or point_cloud is None:
        return None

    xyz = point_cloud[..., :3]
    valid_mask = numpy.isfinite(xyz).all(axis=-1)
    valid_points = xyz[valid_mask]

    if valid_points.shape[0] < 200:
        return None

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(valid_points.astype(numpy.float64))

    cube_pcd = isolate_cube_cluster_open3d(pcd)
    if cube_pcd is None or len(cube_pcd.points) < 50:
        return None

    obb = cube_pcd.get_oriented_bounding_box()
    center = numpy.asarray(obb.center)
    R_cam_cube = numpy.asarray(obb.R)

    t_cam_cube = numpy.eye(4)
    t_cam_cube[:3, :3] = R_cam_cube
    t_cam_cube[:3, 3] = center

    t_robot_cam = numpy.linalg.inv(camera_pose)
    t_robot_cube = t_robot_cam @ t_cam_cube

    return t_robot_cube, t_cam_cube


def main():

    # Initialize ZED Camera
    zed = ZedCamera()
    camera_intrinsic = zed.camera_intrinsic

    # Initialize Lite6 Robot
    arm = XArmAPI(robot_ip)
    arm.connect()
    arm.motion_enable(enable=True)
    arm.set_tcp_offset([0, 0, GRIPPER_LENGTH, 0, 0, 0])
    arm.set_mode(0)
    arm.set_state(0)
    arm.move_gohome(wait=True)
    time.sleep(0.5)

    try:
        # Get Observation
        cv_image = zed.image
        point_cloud = zed.point_cloud
        observation = (cv_image, point_cloud)

        # Get Transformation
        t_cam_robot = get_transform_camera_robot(cv_image, camera_intrinsic)
        if t_cam_robot is None:
            return

        t_cam_cube = None
        cube_result = get_transform_cube(observation, camera_intrinsic, t_cam_robot)
        if cube_result is None:
            return
        t_robot_cube, t_cam_cube = cube_result

        # Visualization
        draw_pose_axes(cv_image, camera_intrinsic, t_cam_cube)
        cv2.namedWindow('Verifying Cube Pose', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Verifying Cube Pose', 1280, 720)
        cv2.imshow('Verifying Cube Pose', cv_image)
        key = cv2.waitKey(0)

        if key == ord('k'):
            cv2.destroyAllWindows()

            xyz = t_robot_cube[:3, 3]
            print(f'Cube in robot frame (m): x={xyz[0]:.3f}, y={xyz[1]:.3f}, z={xyz[2]:.3f}')
            grasp_cube(arm, t_robot_cube)
            place_cube(arm, t_robot_cube)

            arm.stop_lite6_gripper()

    finally:
        # Close Lite6 Robot
        arm.stop_lite6_gripper()
        arm.move_gohome(wait=True)
        time.sleep(0.5)
        arm.disconnect()

        # Close ZED Camera
        zed.close()

if __name__ == "__main__":
    main()
