from __future__ import annotations

import time
import traceback

import cv2
import numpy
import open3d as o3d
from scipy.spatial.transform import Rotation
from xarm.wrapper import XArmAPI

from utils.zed_camera import ZedCamera
from checkpoint1 import grasp_cube, place_cube


################################################################### Constants

robot_ip = ""

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


def get_transform_camera_robot(observation, camera_intrinsic):
    """Return a default camera->robot transform when external markers are unavailable.

    Inputs: observation — camera image (unused); camera_intrinsic — intrinsics (unused).
    Outputs: identity 4x4 transform.
    """
    _ = observation, camera_intrinsic
    return numpy.eye(4)


#################################################################### Geometry 

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


def isolate_cube_cluster_open3d(pcd: o3d.geometry.PointCloud):
    """Clean the cloud and pick the best blob that looks like a cube on the table.

    Inputs: pcd — point cloud in meters.
    Outputs: (cluster_pcd, message) on success, or (None, error string) on failure.
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
        """Score one cluster by size and shape; higher is better cube-like."""
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

    best_cluster = None
    best_score = -1.0
    fallback_largest = None
    fallback_n = 0

    for cid in range(max_label + 1):
        idx = numpy.where(labels == cid)[0]
        if idx.size < 30:
            continue
        cluster = pcd.select_by_index(idx)
        sc, chosen = score_cluster(cluster, idx)
        if idx.size > fallback_n:
            fallback_n = idx.size
            fallback_largest = cluster
        if sc > best_score and chosen is not None:
            best_score = sc
            best_cluster = chosen

    if best_cluster is not None:
        return best_cluster, "cube-like cluster"

    if fallback_largest is not None and len(fallback_largest.points) >= 40:
        return fallback_largest, "fallback: largest cluster"

    return None, "no cluster passed filters"


def get_transform_cube_geometry(observation, camera_intrinsic, camera_pose):
    """Get cube pose from the depth cloud using an oriented box around the cube cluster.

    Inputs: observation — (image, point_cloud); camera_intrinsic — K; camera_pose — world-to-camera 4x4.
    Outputs: ((robot_cube_4x4, cam_cube_4x4), status_msg, height_m), or (None, msg, None) if it fails.
    """
    image, point_cloud = observation
    if image is None or point_cloud is None:
        return None, "missing image or point_cloud", None

    xyz = point_cloud[..., :3]
    valid_mask = numpy.isfinite(xyz).all(axis=-1)
    valid_points = xyz[valid_mask]

    if valid_points.shape[0] < 100:
        return None, f"too few finite points: {valid_points.shape[0]}", None

    valid_points_m, _ = points_to_meters_open3d(valid_points)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(valid_points_m)

    cube_pcd, seg_msg = isolate_cube_cluster_open3d(pcd)
    if cube_pcd is None or len(cube_pcd.points) < 30:
        return None, seg_msg, None

    obb = cube_pcd.get_oriented_bounding_box()
    center = numpy.asarray(obb.center)
    R_cam_cube = numpy.asarray(obb.R)
    ext = numpy.sort(numpy.asarray(obb.extent))
    height_m = float(ext[2])
    t_cam_cube = numpy.eye(4)
    t_cam_cube[:3, :3] = R_cam_cube
    t_cam_cube[:3, 3] = center

    t_robot_cam = numpy.linalg.inv(camera_pose)
    t_robot_cube = t_robot_cam @ t_cam_cube

    return (t_robot_cube, t_cam_cube), seg_msg, height_m


def detect_cube_pose_unified(observation, camera_intrinsic, t_cam_robot):
    """Detect cube pose from point-cloud geometry only.

    Inputs: observation — (image, point_cloud); camera_intrinsic — K; t_cam_robot — camera-to-robot 4x4.
    Outputs: (pose_robot, pose_cam, height_m, "geometry") or (None, None, None, error_text).
    """
    g = get_transform_cube_geometry(observation, camera_intrinsic, t_cam_robot)
    if g[0] is None:
        return None, None, None, g[1]
    (t_r, t_c), _msg, h_m = g
    return t_r, t_c, h_m, "geometry"


#################################################################### Manipulation
# grasp_cube / place_cube: imported from checkpoint1 (fast travel + short gripper dwell).


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


#################################################################### Challenge 1    

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
        det = detect_cube_pose_unified(obs, camera_intrinsic, t_cam_robot)
        if det[0] is None:
            print("Preview detect failed:", det[3])
            return 0
        t_r, t_c, _h, src = det
        disp = cv_image.copy()
        draw_pose_axes(disp, camera_intrinsic, t_c)
        disp = draw_status_overlay(
            disp,
            [
                f"RRC1 preview source={src}",
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

    t_cam_robot = get_transform_camera_robot(zed.image, camera_intrinsic)
    if t_cam_robot is None:
        print("Calibration failed.")
        return 0

    base_det = detect_cube_pose_unified(
        (zed.image, zed.point_cloud),
        camera_intrinsic,
        t_cam_robot,
    )
    if base_det[0] is None:
        print("Could not get base pose:", base_det[3])
        return 0
    t_base, _, _, _src = base_det

    for idx in range(max_cubes):
        if time.time() - start > time_limit_s:
            break

        det = detect_cube_pose_unified(
            (zed.image, zed.point_cloud),
            camera_intrinsic,
            t_cam_robot,
        )
        if det[0] is None:
            print("Detect failed:", det[3])
            break

        t_src, _, _h, _src = det
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

    time.sleep(0.15)
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
    time.sleep(0.15)

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
        time.sleep(0.15)
        arm.disconnect()
        zed.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()