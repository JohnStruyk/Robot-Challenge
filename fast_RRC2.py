from __future__ import annotations

import time
import traceback

import cv2
import numpy
import open3d as o3d
from scipy.spatial.transform import Rotation
from xarm.wrapper import XArmAPI

from utils.zed_camera import ZedCamera

######################################################## Constants 

robot_ip = ""

GRIPPER_LENGTH = 0.067 * 1000.0

CUBE_PHYSICAL_HEIGHT_M = 0.03
REF_CUBE_HEIGHT_M = CUBE_PHYSICAL_HEIGHT_M

GRASP_Z_OFFSET = 0.0001
LIFT_Z_DELTA = 0.04
PLACE_Z_OFFSET = 0.002
TOOL_ROLL_DEG = 180.0
TOOL_PITCH_DEG = 0.0

SAFE_CLEARANCE_CUBE_HEIGHTS = 1.5
MIN_GRASP_ABOVE_CUBE_M = 0.038
SAFE_Z_MIN_M = 0.10
SAFE_Z_MAX_M = 0.36

ARM_SPEED_FAST = 2200
ARM_SPEED_PLACE = 180
ARM_SPEED_LIFT = 2000

GRIPPER_SETTLE_GRASP_S = 0.35
GRIPPER_SETTLE_PLACE_S = 0.45

POSE_SAMPLES = 3
POSE_SAMPLE_DT_S = 0.04


def get_transform_camera_robot(observation, camera_intrinsic):
    """Return a default camera->robot transform when external markers are unavailable.

    Inputs: observation — camera image (unused); camera_intrinsic — intrinsics (unused).
    Outputs: identity 4x4 transform.
    """
    _ = observation, camera_intrinsic
    return numpy.eye(4)


################################################################### Geometry

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


def orthonormalize_rotation(R):
    """Turn a 3x3 matrix into a proper rotation (fix small numeric errors).

    Inputs: R — 3x3 matrix.
    Outputs: 3x3 rotation matrix in SO(3).
    """
    U, _, Vt = numpy.linalg.svd(R)
    Rn = U @ Vt
    if numpy.linalg.det(Rn) < 0:
        U[:, -1] *= -1.0
        Rn = U @ Vt
    return Rn


def average_rotation_matrices(R_list):
    """Average rotations on SO(3) using a unit quaternion mean (signs unified).

    Averaging raw 3x3 matrices breaks orthogonality badly; this is the standard fix.

    Inputs: R_list — list of 3x3 rotation matrices.
    Outputs: one 3x3 rotation matrix, or None if R_list is empty.
    """
    if not R_list:
        return None
    if len(R_list) == 1:
        return numpy.asarray(R_list[0], dtype=numpy.float64)
    Rs = numpy.stack(
        [orthonormalize_rotation(numpy.asarray(R, dtype=numpy.float64)) for R in R_list],
        axis=0,
    )
    quats = Rotation.from_matrix(Rs).as_quat()
    ref = quats[0].copy()
    aligned = numpy.empty_like(quats)
    aligned[0] = ref
    for i in range(1, quats.shape[0]):
        q = quats[i]
        if numpy.dot(ref, q) < 0.0:
            q = -q
        aligned[i] = q
    q_mean = numpy.mean(aligned, axis=0)
    nrm = numpy.linalg.norm(q_mean)
    if nrm < 1e-12:
        return Rs[0]
    q_mean /= nrm
    return Rotation.from_quat(q_mean).as_matrix()


def refine_pose_cam_from_cluster_pcd(cube_pcd: o3d.geometry.PointCloud):
    """Blend robust centroid and OBB center; rotation from OBB projected to SO(3).

    Inputs: cube_pcd — segmented cube point cloud in camera frame.
    Outputs: (t_cam_cube_4x4, height_m) — pose and tallest box edge as height.
    """
    pts = numpy.asarray(cube_pcd.points)
    # Coordinate-wise median resists depth outliers better than the mean.
    centroid = numpy.median(pts, axis=0)
    obb = cube_pcd.get_oriented_bounding_box()
    c_obb = numpy.asarray(obb.center)
    center = 0.62 * centroid + 0.38 * c_obb
    R = orthonormalize_rotation(numpy.asarray(obb.R))
    ext = numpy.sort(numpy.asarray(obb.extent))
    height_m = float(ext[2])
    t_cam = numpy.eye(4)
    t_cam[:3, :3] = R
    t_cam[:3, 3] = center
    return t_cam, height_m


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
    """Get cube pose from the depth cloud using refined pose from the cube cluster.

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

    t_cam_cube, height_m = refine_pose_cam_from_cluster_pcd(cube_pcd)

    t_robot_cam = numpy.linalg.inv(camera_pose)
    t_robot_cube = t_robot_cam @ t_cam_cube

    return (t_robot_cube, t_cam_cube), seg_msg, height_m


def average_pose_matrices(mats):
    """Merge several 4x4 poses: median translation, quaternion mean of rotations.

    Inputs: mats — list of 4x4 numpy arrays.
    Outputs: single 4x4 pose, or None if mats is empty.
    """
    if not mats:
        return None
    T = numpy.stack(mats, axis=0)
    pos = numpy.median(T[:, :3, 3], axis=0)
    R_list = [T[i, :3, :3] for i in range(T.shape[0])]
    Rm = average_rotation_matrices(R_list)
    if Rm is None:
        return None
    out = numpy.eye(4)
    out[:3, :3] = Rm
    out[:3, 3] = pos
    return out


def detect_cube_pose_once(observation, camera_intrinsic, t_cam_robot):
    """One geometry-only detection attempt (single frame).

    Inputs: observation — (image, point_cloud); camera_intrinsic — K; t_cam_robot — camera-to-robot.
    Outputs: (pose_robot, pose_cam, height_m, "geometry") or (None, None, None, error_text).
    """
    g = get_transform_cube_geometry(observation, camera_intrinsic, t_cam_robot)
    if g[0] is None:
        return None, None, None, g[1]
    (t_r, t_c), _msg, h_m = g
    return t_r, t_c, h_m, "geometry"


def detect_cube_pose_unified(
    observation,
    camera_intrinsic,
    t_cam_robot,
    n_samples=None,
    sample_dt_s=None,
):
    """Detect cube pose; optionally average several frames for a steadier pose.

    Inputs: observation — (image, point_cloud); camera_intrinsic — K; t_cam_robot — camera-to-robot;
            n_samples — frames to fuse (1 = no fusion);
            sample_dt_s — sleep between frames when n_samples > 1.
    Outputs: (pose_robot, pose_cam, height_m, source) or (None, None, None, error_text).
    """
    n_samples = POSE_SAMPLES if n_samples is None else n_samples
    sample_dt_s = POSE_SAMPLE_DT_S if sample_dt_s is None else sample_dt_s

    if n_samples <= 1:
        r = detect_cube_pose_once(
            observation, camera_intrinsic, t_cam_robot
        )
        if r[0] is None:
            return None, None, None, r[3]
        return r[0], r[1], r[2], r[3]

    ok = []
    last_err = "no samples"
    for _ in range(n_samples):
        r = detect_cube_pose_once(
            observation, camera_intrinsic, t_cam_robot
        )
        if r[0] is not None:
            ok.append(r)
        else:
            last_err = r[3]
        time.sleep(sample_dt_s)

    if not ok:
        return None, None, None, last_err

    trs = [x[0] for x in ok]
    tcs = [x[1] for x in ok]
    t_r = average_pose_matrices(trs)
    t_c = average_pose_matrices(tcs)
    h_m = float(numpy.median([x[2] for x in ok]))
    src = ok[0][3]
    return t_r, t_c, h_m, src


def compute_safe_clearance_z_mm(tower_top_z_m, cube_center_z_m, cube_height_m):
    """Safe tool height in mm so the arm clears the stack (not a fixed 0.22 m).

    Inputs: tower_top_z_m — world z of top of stack; cube_center_z_m — cube center z;
            cube_height_m — reference cube height for margin.
    Outputs: safe z in millimeters for the arm (clamped).
    """
    ref_h = max(float(cube_height_m), REF_CUBE_HEIGHT_M)
    clearance = SAFE_CLEARANCE_CUBE_HEIGHTS * ref_h
    z_m = max(tower_top_z_m + clearance, float(cube_center_z_m) + MIN_GRASP_ABOVE_CUBE_M)
    z_m = float(numpy.clip(z_m, SAFE_Z_MIN_M, SAFE_Z_MAX_M))
    return z_m * 1000.0


#################################################################### Manipulation

def set_line(arm, x_mm, y_mm, z_mm, yaw_deg, speed):
    """Move the arm to one Cartesian point with a set speed (helper).

    Inputs: arm — xArm API; x_mm, y_mm, z_mm — position; yaw_deg — tool yaw; speed — mm/s.
    Outputs: none (blocks until move finishes).
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


def grasp_cube(arm, cube_pose, tower_top_z_m, cube_height_m):
    """Grasp with fast moves and adaptive safe height above the current stack top.

    Inputs: arm — xArm API; cube_pose — 4x4 in robot frame; tower_top_z_m — stack top z (m);
            cube_height_m — used for clearance margin.
    Outputs: none (moves the robot).
    """
    xyz = cube_pose[:3, 3]
    x_mm, y_mm, z_mm = (xyz * 1000.0).tolist()
    z_c_m = float(xyz[2])
    safe_z_mm = compute_safe_clearance_z_mm(tower_top_z_m, z_c_m, cube_height_m)
    grasp_z_mm = z_mm + (GRASP_Z_OFFSET * 1000.0)
    lift_z_mm = max(safe_z_mm, grasp_z_mm + (LIFT_Z_DELTA * 1000.0))

    cube_r = Rotation.from_matrix(cube_pose[:3, :3])
    _, _, cube_yaw_deg = cube_r.as_euler("xyz", degrees=True)

    arm.open_lite6_gripper()
    time.sleep(GRIPPER_SETTLE_GRASP_S)

    set_line(arm, x_mm, y_mm, safe_z_mm, cube_yaw_deg, ARM_SPEED_FAST)
    set_line(arm, x_mm, y_mm, grasp_z_mm, cube_yaw_deg, ARM_SPEED_FAST)
    arm.close_lite6_gripper()
    time.sleep(GRIPPER_SETTLE_GRASP_S)
    set_line(arm, x_mm, y_mm, lift_z_mm, cube_yaw_deg, ARM_SPEED_LIFT)


def place_cube(arm, cube_pose, tower_top_z_m, cube_height_m):
    """Place with fast travel, slower final down move, then fast lift away.

    Inputs: arm — xArm API; cube_pose — 4x4 target in robot frame; tower_top_z_m — stack top z;
            cube_height_m — used for clearance margin.
    Outputs: none (moves the robot).
    """
    xyz = cube_pose[:3, 3]
    x_mm, y_mm, z_mm = (xyz * 1000.0).tolist()
    z_c_m = float(xyz[2])
    safe_z_mm = compute_safe_clearance_z_mm(tower_top_z_m, z_c_m, cube_height_m)
    place_z_mm = z_mm + (PLACE_Z_OFFSET * 1000.0)
    lift_z_mm = max(safe_z_mm, place_z_mm + (LIFT_Z_DELTA * 1000.0))

    cube_r = Rotation.from_matrix(cube_pose[:3, :3])
    _, _, cube_yaw_deg = cube_r.as_euler("xyz", degrees=True)

    set_line(arm, x_mm, y_mm, safe_z_mm, cube_yaw_deg, ARM_SPEED_FAST)
    set_line(arm, x_mm, y_mm, place_z_mm, cube_yaw_deg, ARM_SPEED_PLACE)
    arm.open_lite6_gripper()
    time.sleep(GRIPPER_SETTLE_PLACE_S)
    set_line(arm, x_mm, y_mm, lift_z_mm, cube_yaw_deg, ARM_SPEED_LIFT)


def apply_delta_z_to_pose(t_pose, delta_z_m):
    """Shift a pose up or down along world Z only.

    Inputs: t_pose — 4x4; delta_z_m — meters to add to the Z position.
    Outputs: new 4x4 pose (copy).
    """
    out = numpy.copy(t_pose)
    out[2, 3] += delta_z_m
    return out


#################################################################### Visualization

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


#################################################################### Challenge 2 

def run_challenge_irregular_skyscraper(
    arm,
    zed,
    *,
    max_cubes=10,
    time_limit_s=60.0,
    dry_run_preview=True,
):
    """Fast irregular stack: multi-sample detect, adaptive safe-Z, timed gripper.

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
        t_r, t_c, h_m, src = det
        disp = cv_image.copy()
        draw_pose_axes(disp, camera_intrinsic, t_c)
        disp = draw_status_overlay(
            disp,
            [
                f"RRC2 preview h={h_m:.3f}m source={src}",
                "Press k to run irregular stack, any other key to abort",
            ],
            (0, 220, 0),
        )
        cv2.namedWindow("RRC2", cv2.WINDOW_NORMAL)
        cv2.imshow("RRC2", disp)
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

    first = detect_cube_pose_unified(
        (zed.image, zed.point_cloud),
        camera_intrinsic,
        t_cam_robot,
    )
    if first[0] is None:
        print("Initial detect failed:", first[3])
        return 0

    _t0, _c0, h0, _src0 = first
    z_center0 = float(_t0[2, 3])
    current_top_z = z_center0 + h0 / 2.0

    for _ in range(max_cubes):
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

        t_src, _t_c, h_m, src = det
        z_c = float(t_src[2, 3])
        desired_center_z = current_top_z + h_m / 2.0
        delta_z = desired_center_z - z_c
        t_tgt = apply_delta_z_to_pose(t_src, delta_z)

        tower_top_z_m = current_top_z
        h_grasp = max(float(h_m), REF_CUBE_HEIGHT_M)

        try:
            grasp_cube(arm, t_src, tower_top_z_m, h_grasp)
            place_cube(arm, t_tgt, tower_top_z_m, float(h_m))
            arm.stop_lite6_gripper()
            current_top_z += h_m
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
        n = run_challenge_irregular_skyscraper(
            arm,
            zed,
            max_cubes=10,
            time_limit_s=60.0,
            dry_run_preview=True,
        )
        print(f"RRC2: placed {n} cube(s).")
    finally:
        arm.stop_lite6_gripper()
        arm.move_gohome(wait=True)
        time.sleep(0.5)
        arm.disconnect()
        zed.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()