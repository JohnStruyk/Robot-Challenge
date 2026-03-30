from __future__ import annotations

import time
import traceback

import cv2
import numpy
import open3d as o3d
from scipy.spatial.transform import Rotation
from xarm.wrapper import XArmAPI
from pupil_apriltags import Detector

from utils.zed_camera import ZedCamera
from checkpoint1 import robot_ip

from checkpoint0 import get_pnp_pairs, get_transform_camera_robot


################################################################### Constants
NUM_CUBE = 9

GRIPPER_LENGTH = 0.067 * 1000.0

####################################################################### Nominal physical height
CUBE_PHYSICAL_HEIGHT_M = 0.03

###################################################################### Fixed vertical pitch
STACK_HEIGHT_M = 0.045

##################################################################### Manipulation
GRASP_Z_OFFSET = 0.0001
LIFT_Z_DELTA = 0.04
PLACE_Z_OFFSET = 0.002
TOOL_ROLL_DEG = 180.0
TOOL_PITCH_DEG = 0.0

# Fixed clearance plane (m) above table for horizontal travel.
SAFE_Z = 0.22



#################################################################### Geometry helpers


def points_to_meters_open3d(xyz):
    """Scale point units to meters if values look like millimeters.

    Inputs: xyz — Nx3 array of 3D points.
    Outputs: (scaled_xyz, scale) — scaled points and the scale factor used (0.001 or 1.0).
    """
    if xyz.size == 0:
        return xyz, 1.0
    max_abs = float(numpy.nanmax(numpy.abs(xyz)))
    scale = 0.001 if max_abs > 50.0 else 1.0
    return (xyz * scale).astype(numpy.float64), scale


def segment_all_cubes_open3d(pcd: o3d.geometry.PointCloud):
    """Segment the point cloud into multiple cube-like clusters on the table.

    Inputs: pcd — point cloud in meters.
    Outputs: (clusters, message) where clusters is a list of point clouds.
    """
    if len(pcd.points) < 150:
        return [], "too few points after NaN filter"

    # Downsample and denoise
    pcd = pcd.voxel_down_sample(voxel_size=0.003)
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=25, std_ratio=2.0)

    if len(pcd.points) < 100:
        return [], "too few points after voxel/outlier"

    # Remove dominant plane (table)
    plane_model, inliers = pcd.segment_plane(
        distance_threshold=0.010,
        ransac_n=3,
        num_iterations=1500,
    )
    if len(inliers) > 0 and len(inliers) > 0.12 * len(pcd.points):
        pcd = pcd.select_by_index(inliers, invert=True)

    if len(pcd.points) < 80:
        return [], "nothing left after plane removal"

    # Cluster with DBSCAN
    labels = numpy.asarray(
        pcd.cluster_dbscan(eps=0.020, min_points=25, print_progress=False)
    )
    max_label = int(labels.max()) if labels.size else -1
    if max_label < 0:
        return [], "DBSCAN found no clusters"

    clusters = []

    def score_cluster(cluster, idx):
        """Score one cluster by size and shape; higher is better cube-like."""
        obb = cluster.get_oriented_bounding_box()
        ext = numpy.sort(numpy.asarray(obb.extent))
        if ext[2] < 1e-9:
            return -1.0
        max_dim = float(ext[2])
        min_dim = float(ext[0])
        compact = min_dim / max_dim if max_dim > 0 else 0.0
        size_ok = 0.008 <= max_dim <= 0.090
        if not size_ok:
            return float(idx.size) * 0.01
        qual = 1.0 if 0.25 < compact <= 1.0 else 0.3
        return float(idx.size) * compact * qual

    for cid in range(max_label + 1):
        idx = numpy.where(labels == cid)[0]
        if idx.size < 30:
            continue
        cluster = pcd.select_by_index(idx)
        sc = score_cluster(cluster, idx)
        if sc <= 0.0:
            continue
        clusters.append(cluster)

    if not clusters:
        return [], "no cluster passed filters"

    return clusters, f"{len(clusters)} cube-like clusters"

def segment_top_n_cubes_open3d(pcd: o3d.geometry.PointCloud, max_cubes=1):
    """
    Return the top-N cube-like clusters based on the same scoring used in
    segment_all_cubes_open3d.

    Inputs:
        pcd        — Open3D point cloud (meters)
        max_cubes  — number of clusters to return

    Outputs:
        (clusters, message)
    """
    clusters, msg = segment_all_cubes_open3d(pcd)
    if not clusters:
        return [], msg

    # Score clusters using the same scoring logic
    scored = []
    for cluster in clusters:
        obb = cluster.get_oriented_bounding_box()
        ext = numpy.sort(numpy.asarray(obb.extent))
        if ext[2] < 1e-9:
            continue
        max_dim = float(ext[2])
        min_dim = float(ext[0])
        compact = min_dim / max_dim if max_dim > 0 else 0.0
        size_ok = 0.008 <= max_dim <= 0.090
        if not size_ok:
            score = float(len(cluster.points)) * 0.01
        else:
            qual = 1.0 if 0.25 < compact <= 1.0 else 0.3
            score = float(len(cluster.points)) * compact * qual

        scored.append((score, cluster))

    if not scored:
        return [], "no cluster passed scoring"

    # Sort by score descending
    scored.sort(key=lambda x: x[0], reverse=True)

    # Return top N clusters
    top_clusters = [c for (_, c) in scored[:max_cubes]]
    return top_clusters, f"returned {len(top_clusters)} clusters"



def cluster_to_pose(cluster: o3d.geometry.PointCloud):
    """Convert a cube cluster to a pose and height in camera frame.

    Inputs: cluster — Open3D point cloud.
    Outputs: (t_cam_cube_4x4, height_m).
    """
    obb = cluster.get_oriented_bounding_box()
    center = numpy.asarray(obb.center)
    R_cam_cube = numpy.asarray(obb.R)
    ext = numpy.sort(numpy.asarray(obb.extent))
    height_m = float(ext[2])

    t_cam_cube = numpy.eye(4)
    t_cam_cube[:3, :3] = R_cam_cube
    t_cam_cube[:3, 3] = center

    return t_cam_cube, height_m


def detect_all_cubes_geometry(observation, camera_intrinsic, t_cam_robot):
    """Detect all cube poses from the depth cloud using oriented boxes around clusters.

    Inputs: observation — (image, point_cloud); camera_intrinsic — K; t_cam_robot — camera-to-robot 4x4.
    Outputs: (list_of_results, status_msg) where each result is (t_robot_cube, t_cam_cube, height_m).
    """
    image, point_cloud = observation
    if image is None or point_cloud is None:
        return [], "missing image or point_cloud"

    xyz = point_cloud[..., :3]
    valid_mask = numpy.isfinite(xyz).all(axis=-1)
    valid_points = xyz[valid_mask]

    if valid_points.shape[0] < 100:
        return [], f"too few finite points: {valid_points.shape[0]}"

    valid_points_m, _ = points_to_meters_open3d(valid_points)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(valid_points_m)

    clusters, seg_msg = segment_top_n_cubes_open3d(pcd, max_cubes= NUM_CUBES)
    if not clusters:
        return [], seg_msg

    results = []
    for cluster in clusters:
        t_cam_cube, height_m = cluster_to_pose(cluster)
        # camera-to-robot transform: robot_T_cube = robot_T_cam * cam_T_cube
        t_robot_cube = t_cam_robot @ t_cam_cube
        results.append((t_robot_cube, t_cam_cube, height_m))

    return results, seg_msg


def detect_all_cubes_unified(observation, camera_intrinsic, t_cam_robot):
    """Unified multi-cube detection wrapper.

    Inputs: observation — (image, point_cloud); camera_intrinsic — K; t_cam_robot — camera-to-robot 4x4.
    Outputs: (results, source_or_error) where results is a list of (t_robot, t_cam, h_m).
    """
    results, msg = detect_all_cubes_geometry(observation, camera_intrinsic, t_cam_robot)
    if not results:
        return [], msg
    return results, "geometry"


def choose_next_cube(cube_results):
    """Choose the next cube to pick based on proximity to robot origin in XY.

    Inputs: cube_results — list of (t_robot_cube, t_cam_cube, height_m).
    Outputs: (t_robot_cube, t_cam_cube, height_m) for the chosen cube.
    """
    if not cube_results:
        return None

    def xy_dist_sq(t_robot_cube):
        x, y, _ = t_robot_cube[:3, 3]
        return x * x + y * y

    best = min(cube_results, key=lambda r: xy_dist_sq(r[0]))
    return best


#################################################################### Manipulation

def grasp_cube(arm, cube_pose):
    """Move to the cube, close the gripper, and lift up.

    Inputs: arm — xArm API; cube_pose — 4x4 pose of the cube in robot frame (meters).
    Outputs: none (moves the robot).
    """
    xyz = cube_pose[:3, 3]
    x_mm, y_mm, z_mm = (xyz * 1000.0).tolist()
    safe_z_mm = SAFE_Z * 1000.0
    grasp_z_mm = z_mm + (GRASP_Z_OFFSET * 1000.0)
    lift_z_mm = max(safe_z_mm, grasp_z_mm + (LIFT_Z_DELTA * 1000.0))

    cube_r = Rotation.from_matrix(cube_pose[:3, :3])
    _, _, cube_yaw_deg = cube_r.as_euler("xyz", degrees=True)

    arm.open_lite6_gripper()
    time.sleep(1)

    arm.set_position(
        x_mm,
        y_mm,
        safe_z_mm,
        TOOL_ROLL_DEG,
        TOOL_PITCH_DEG,
        cube_yaw_deg,
        is_radian=False,
        wait=True,
    )
    arm.set_position(
        x_mm,
        y_mm,
        grasp_z_mm,
        TOOL_ROLL_DEG,
        TOOL_PITCH_DEG,
        cube_yaw_deg,
        is_radian=False,
        wait=True,
    )
    arm.close_lite6_gripper()
    time.sleep(1)
    arm.set_position(
        x_mm,
        y_mm,
        lift_z_mm,
        TOOL_ROLL_DEG,
        TOOL_PITCH_DEG,
        cube_yaw_deg,
        is_radian=False,
        wait=True,
    )


def place_cube(arm, cube_pose):
    """Move above the drop pose, place the cube, open gripper, and lift away.

    Inputs: arm — xArm API; cube_pose — 4x4 target pose in robot frame (meters).
    Outputs: none (moves the robot).
    """
    xyz = cube_pose[:3, 3]
    x_mm, y_mm, z_mm = (xyz * 1000.0).tolist()
    safe_z_mm = SAFE_Z * 1000.0
    place_z_mm = z_mm + (PLACE_Z_OFFSET * 1000.0)
    lift_z_mm = max(safe_z_mm, place_z_mm + (LIFT_Z_DELTA * 1000.0))

    cube_r = Rotation.from_matrix(cube_pose[:3, :3])
    _, _, cube_yaw_deg = cube_r.as_euler("xyz", degrees=True)

    arm.set_position(
        x_mm,
        y_mm,
        safe_z_mm,
        TOOL_ROLL_DEG,
        TOOL_PITCH_DEG,
        cube_yaw_deg,
        is_radian=False,
        wait=True,
    )
    arm.set_position(
        x_mm,
        y_mm,
        place_z_mm,
        TOOL_ROLL_DEG,
        TOOL_PITCH_DEG,
        cube_yaw_deg,
        is_radian=False,
        wait=True,
    )
    arm.open_lite6_gripper()
    time.sleep(1)
    arm.set_position(
        x_mm,
        y_mm,
        lift_z_mm,
        TOOL_ROLL_DEG,
        TOOL_PITCH_DEG,
        cube_yaw_deg,
        is_radian=False,
        wait=True,
    )


def make_standard_tower_target_pose(base_pose, stack_index):
    """Copy the base pose but raise Z for stacking higher cubes.

    Inputs: base_pose — 4x4; stack_index — 0 = first cube, 1 = one on top, etc.
    Outputs: new 4x4 pose with Z increased by stack_index * STACK_HEIGHT_M.
    """
    t = numpy.copy(base_pose)
    t[2, 3] = base_pose[2, 3] + stack_index * STACK_HEIGHT_M
    return t


#################################################################### Visualization helpers


def draw_pose_axes(image, camera_intrinsic, pose, size=0.1):
    """Draw RGB axes on the image for a pose in camera frame (for debugging).

    Inputs: image — BGR image to draw on; camera_intrinsic — K; pose — 4x4 in camera frame; size — axis length in meters.
    Outputs: none (draws on image in place).
    """
    rvec, _ = cv2.Rodrigues(pose[:3, :3])
    tvec = pose[:3, 3]
    frame_points = (
        numpy.array(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
            dtype=numpy.float64,
        ).reshape(-1, 3)
        * size
    )
    ipoints, _ = cv2.projectPoints(frame_points, rvec, tvec, camera_intrinsic, None)
    ipoints = numpy.round(ipoints).astype(int)
    origin = tuple(ipoints[0].ravel())
    unit_x = tuple(ipoints[1].ravel())
    unit_y = tuple(ipoints[2].ravel())
    unit_z = tuple(ipoints[3].ravel())
    cv2.line(image, origin, unit_x, (0, 0, 255), 2)
    cv2.line(image, origin, unit_y, (0, 255, 0), 2)
    cv2.line(image, origin, unit_z, (255, 0, 0), 2)


def draw_status_overlay(image_bgra, lines, color=(0, 220, 0)):
    """Draw a few lines of text on a copy of the image.

    Inputs: image_bgra — image; lines — list of strings; color — BGR color tuple.
    Outputs: new image with text drawn.
    """
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


#################################################################### Challenge 1 – multi-cube tower


def run_challenge_standard_tower(
    arm,
    zed,
    *,
    max_cubes=10,
    time_limit_s=60.0,
    dry_run_preview=True,
):
    """Run the tower challenge: pick cubes and stack them in a straight column.

    Inputs: arm — xArm API; zed — camera; max_cubes — cap; time_limit_s — seconds;
            dry_run_preview — if True, show one frame and wait for 'k'.
    Outputs: number of cubes placed (int).
    """
    camera_intrinsic = zed.camera_intrinsic

    # Grab an initial frame
    cv_image = zed.image
    point_cloud = zed.point_cloud

    if dry_run_preview:
        if cv_image is None:
            print("No image from ZED.")
            return 0
        t_cam_robot = get_transform_camera_robot(cv_image, camera_intrinsic)
        if t_cam_robot is None:
            print("Calibration failed (arena tags).")
            return 0
        obs = (cv_image, point_cloud)
        results, src = detect_all_cubes_unified(obs, camera_intrinsic, t_cam_robot)
        if not results:
            print("Preview detect failed:", src)
            return 0

        disp = cv_image.copy()
        for _t_r, t_c, _h in results:
            draw_pose_axes(disp, camera_intrinsic, t_c)
        disp = draw_status_overlay(
            disp,
            [
                f"RRC1 preview source={src}, cubes={len(results)}",
                "Press k to run tower, any other key to abort",
            ],
            (0, 220, 0),
        )
        cv2.namedWindow("RRC1", cv2.WINDOW_NORMAL)
        cv2.imshow("RRC1", disp)
        key = cv2.waitKey(0)
        cv2.destroyAllWindows()
        if key != ord("k"):
            print("Aborted.")
            return 0

    placed = 0
    start = time.time()

    # Recompute transform and base pose from a fresh frame
    cv_image = zed.image
    point_cloud = zed.point_cloud
    t_cam_robot = get_transform_camera_robot(cv_image, camera_intrinsic)
    if t_cam_robot is None:
        print("Calibration failed.")
        return 0

    base_obs = (cv_image, point_cloud)
    base_results, src = detect_all_cubes_unified(base_obs, camera_intrinsic, t_cam_robot)
    if not base_results:
        print("Could not get base pose:", src)
        return 0

    # Choose base cube (e.g., nearest to robot)
    base_robot_pose, _base_cam_pose, _base_h = choose_next_cube(base_results)
    t_base = base_robot_pose

    for idx in range(max_cubes):
        if time.time() - start > time_limit_s:
            break

        # Refresh camera data each iteration (assuming ZedCamera updates internally)
        cv_image = zed.image
        point_cloud = zed.point_cloud
        obs = (cv_image, point_cloud)

        results, src = detect_all_cubes_unified(obs, camera_intrinsic, t_cam_robot)
        if not results:
            print("Detect failed:", src)
            break

        chosen = choose_next_cube(results)
        if chosen is None:
            print("No suitable cube found.")
            break

        t_src, _t_cam_src, _h = chosen
        t_tgt = make_standard_tower_target_pose(t_base, idx)

        try:
            grasp_cube(arm, t_src)
            place_cube(arm, t_tgt)
            arm.stop_lite6_gripper()
            placed += 1
        except Exception as exc:
            traceback.print_exc()
            print("Motion error:", exc)
            break

    time.sleep(0.5)
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
    time.sleep(0.5)

    try:
        n = run_challenge_standard_tower(
            arm,
            zed,
            max_cubes=10,
            time_limit_s=60.0,
            dry_run_preview=True,
        )
        print(f"RRC1: placed {n} cube(s).")
    finally:
        arm.stop_lite6_gripper()
        arm.move_gohome(wait=True)
        time.sleep(0.5)
        arm.disconnect()
        zed.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
