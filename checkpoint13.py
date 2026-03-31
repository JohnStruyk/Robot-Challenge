import traceback
import cv2, numpy, time
import open3d as o3d
from xarm.wrapper import XArmAPI

from utils.vis_utils import draw_pose_axes
from utils.zed_camera import ZedCamera
from checkpoint0 import get_transform_camera_robot
from checkpoint1 import grasp_cube, place_cube, GRIPPER_LENGTH, robot_ip

CUBE_SIZE = 0.025


def points_to_meters_open3d(xyz):
    """
    ZED often returns XYZ in millimeters; Open3D params below assume meters.
    Heuristic: if coordinates look like mm, scale to meters.
    """
    if xyz.size == 0:
        return xyz, 1.0
    max_abs = float(numpy.nanmax(numpy.abs(xyz)))
    scale = 0.001 if max_abs > 50.0 else 1.0
    return (xyz * scale).astype(numpy.float64), scale


def isolate_cube_cluster_open3d(pcd: o3d.geometry.PointCloud, num_cubes):
    """
    Segment a tabletop cube: voxel downsample, outliers, RANSAC plane, DBSCAN,
    then pick the most cube-like cluster (with a size fallback).
    """
    if len(pcd.points) < 150:
        return None, "too few points after NaN filter"

    pcd = pcd.voxel_down_sample(voxel_size=0.003)
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=25, std_ratio=2.0)

    if len(pcd.points) < 100:
        return None, "too few points after voxel/outlier"

    plane_model, inliers = pcd.segment_plane(
        distance_threshold=0.010,
        ransac_n=3,
        num_iterations=1500,
    )
    if len(inliers) > 0 and len(inliers) > 0.12 * len(pcd.points):
        pcd = pcd.select_by_index(inliers, invert=True)

    if len(pcd.points) < 80:
        return None, "nothing left after plane removal"

    labels = numpy.asarray(
        pcd.cluster_dbscan(eps=0.020, min_points=25, print_progress=False)
    )
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

        # keep track of all clusters for fallback
        fallback_clusters.append((idx.size, cluster))

        if chosen is not None:
            scored_clusters.append((sc, chosen))

    # sort by score descending
    scored_clusters.sort(key=lambda x: x[0], reverse=True)

    if len(scored_clusters) > 0:
        top_clusters = [c for (_, c) in scored_clusters[:num_cubes]]
        return top_clusters, f"{len(top_clusters)} cube-like clusters"

    # fallback: largest clusters by size
    fallback_clusters.sort(key=lambda x: x[0], reverse=True)

    if len(fallback_clusters) > 0:
        top_clusters = [c for (_, c) in fallback_clusters[:num_cubes] if len(c.points) >= 40]
        if len(top_clusters) > 0:
            return top_clusters, f"fallback: {len(top_clusters)} largest clusters"

    return None, "no cluster passed filters"

def get_cube_transform(cube_pcd, camera_pose):
    if cube_pcd is None or len(cube_pcd.points) < 30:
        return None

    obb = cube_pcd.get_oriented_bounding_box()
    center = numpy.asarray(obb.center)

    # Make sure this is a writable array
    R_raw = numpy.array(obb.R, dtype=float, copy=True)

    # --- 2. Extract stable Z axis (copy, not view) ---
    z_axis = numpy.array(R_raw[:, 2], dtype=float, copy=True)
    z_axis /= numpy.linalg.norm(z_axis)

    # --- 3. Project OBB X axis onto table plane to fix yaw ---
    x_axis = numpy.array(R_raw[:, 0], dtype=float, copy=True)
    x_axis = x_axis - numpy.dot(x_axis, z_axis) * z_axis
    n = numpy.linalg.norm(x_axis)
    if n < 1e-6:
        tmp = numpy.array([1.0, 0.0, 0.0], dtype=float)
        if abs(numpy.dot(tmp, z_axis)) > 0.9:
            tmp = numpy.array([0.0, 1.0, 0.0], dtype=float)
        x_axis = tmp - numpy.dot(tmp, z_axis) * z_axis
    x_axis /= numpy.linalg.norm(x_axis)

    # --- 4. Recompute Y axis ---
    y_axis = numpy.cross(z_axis, x_axis)
    y_axis /= numpy.linalg.norm(y_axis)

    # --- 5. Final stable rotation matrix ---
    R_fixed = numpy.column_stack([x_axis, y_axis, z_axis])

    # --- 6. Build camera-frame transform ---
    t_cam_cube = numpy.eye(4)
    t_cam_cube[:3, :3] = R_fixed
    t_cam_cube[:3, 3] = center

    # --- 7. Convert to robot frame ---
    t_robot_cam = numpy.linalg.inv(camera_pose)
    t_robot_cube = t_robot_cam @ t_cam_cube

    return (t_robot_cube, t_cam_cube)

def get_cube_transforms(observation, camera_intrinsic, camera_pose):
    """
    Estimate cube pose from image + point cloud (geometry; image for API only).

    Returns
    -------
    tuple
        On success: ``((t_robot_cube, t_cam_cube), status_message)``.
        On failure: ``(None, status_message)``.
    """
    image, point_cloud = observation
    if image is None or point_cloud is None:
        return None, "missing image or point_cloud"

    xyz = point_cloud[..., :3]
    valid_mask = numpy.isfinite(xyz).all(axis=-1)
    valid_points = xyz[valid_mask]

    if valid_points.shape[0] < 100:
        return None, f"too few finite points: {valid_points.shape[0]}"

    valid_points_m, _scale = points_to_meters_open3d(valid_points)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(valid_points_m)

    cube_pcds, _ = isolate_cube_cluster_open3d(pcd, num_cubes=12)

    transforms = []

    for pcd in cube_pcds:
        transforms.append(get_cube_transform(pcd, camera_pose))

    return transforms


def draw_status_overlay(image_bgra, lines, color=(0, 220, 0)):
    """Draw multiline status on BGRA image (copy)."""
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
    """
    Full pure-vision path: calibration + get_transform_cube + overlay for display.

    Used by checkpoint 6 and checkpoint 7.

    Parameters
    ----------
    cv_image : numpy.ndarray or None
        BGRA frame from ZED.
    point_cloud : numpy.ndarray or None
        Dense XYZ point cloud aligned with the image.
    camera_intrinsic : numpy.ndarray
        3x3 intrinsics.

    Returns
    -------
    tuple
        ``(t_robot_cube, t_cam_cube, display_bgra, status_message)``
        ``t_robot_cube`` / ``t_cam_cube`` may be ``None`` if perception fails.
    """
    if cv_image is None:
        blank = numpy.zeros((720, 1280, 4), dtype=numpy.uint8)
        disp = draw_status_overlay(blank, ["ZED image is None"], (0, 0, 255))
        return None, None, disp, "no image"

    t_cam_robot = get_transform_camera_robot(cv_image, camera_intrinsic)
    if t_cam_robot is None:
        disp = draw_status_overlay(
            cv_image,
            ["Calibration FAILED (checkpoint0 tags / PnP)"],
            (0, 0, 255),
        )
        return None, None, disp, "calibration failed"


    cube_transforms = get_cube_transforms(
        (cv_image, point_cloud), camera_intrinsic, t_cam_robot
    )

    t_robot_cube, t_cam_cube = cube_transforms[0]

    lines = [
        "LALALALALA"
    ]
    disp = cv_image.copy()

    for transform in cube_transforms:
        t_cam_cube = transform[1]
        if t_cam_cube is not None and numpy.isfinite(t_cam_cube).all():
            draw_pose_axes(disp, camera_intrinsic, t_cam_cube)
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

        t_robot_cube, _t_cam_cube = cube_transforms[0]

        t_robot_cube_2, _ = cube_transforms[1]


        cv2.namedWindow("Verifying Cube Pose", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Verifying Cube Pose", 1280, 720)
        cv2.imshow("Verifying Cube Pose", disp)
        key = cv2.waitKey(0)

        if key == ord("k") and t_robot_cube is not None:
            cv2.destroyAllWindows()
            xyz = t_robot_cube[:3, 3]
            print(
                f"Cube in robot frame (m): x={xyz[0]:.3f}, y={xyz[1]:.3f}, z={xyz[2]:.3f}"
            )



            t_robot_stack = t_robot_cube

            for i in range(7):
                t_robot_cube, _ = cube_transforms[i]
                grasp_cube(arm, t_robot_cube)
                place_cube(arm, t_robot_stack)
                t_robot_stack[2, 3] += CUBE_SIZE

            arm.stop_lite6_gripper()
        else:
            cv2.destroyAllWindows()

    finally:
        arm.stop_lite6_gripper()
        arm.move_gohome(wait=True)
        time.sleep(0.5)
        arm.disconnect()
        zed.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
