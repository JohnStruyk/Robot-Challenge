import traceback
import cv2, numpy, time
import open3d as o3d
from xarm.wrapper import XArmAPI

from utils.vis_utils import draw_pose_axes
from utils.zed_camera import ZedCamera
from checkpoint0 import TAG_CENTER_COORDINATES, get_transform_camera_robot
from checkpoint1 import grasp_cube, place_cube, GRIPPER_LENGTH, robot_ip
from best_RRC1 import grasp_cube_fast, place_cube_fast
from orientation_RRC import physical_cube_pose_from_points

CUBE_SIZE = 0.025
STACK_HEIGHT_GOAL = 9

# Camera-frame depth band (meters after mm→m scaling) — drops far wall / floor junk outside workspace.
GPU_Z_MIN_M = 0.28
GPU_Z_MAX_M = 1.4

# Play mat = axis-aligned rectangle from AprilTag centers (robot XY), with inset.
_TAG_CENTERS_XY = numpy.asarray(TAG_CENTER_COORDINATES, dtype=numpy.float64)
PLAY_AREA_XY_MARGIN_M = 0.015
PLAY_AREA_X_MIN_M = float(_TAG_CENTERS_XY[:, 0].min() + PLAY_AREA_XY_MARGIN_M)
PLAY_AREA_X_MAX_M = float(_TAG_CENTERS_XY[:, 0].max() - PLAY_AREA_XY_MARGIN_M)
PLAY_AREA_Y_MIN_M = float(_TAG_CENTERS_XY[:, 1].min() + PLAY_AREA_XY_MARGIN_M)
PLAY_AREA_Y_MAX_M = float(_TAG_CENTERS_XY[:, 1].max() - PLAY_AREA_XY_MARGIN_M)
WORKSPACE_Z_ROBOT_M = (-0.12, 0.48)



LIL_CUBE_SIZE = 0.0225
LIL_CUBES_GOAL = 3
MID_CUBE_SIZE = 0.025
MID_CUBES_GOAL = 3
BIG_CUBE_SIZE = 0.03
BIG_CUBES_GOAL = 3

# Largest three blobs: richer cloud + table-plane pose (30 mm nominal).
LARGE_DENSE_CROP_MARGIN_M = 0.012
# Prefer refined pose when OBB suggests ~30 mm class (avoids mis-applying 30 mm model to 25 mm).
LARGE_EXTENT_MIN_M = 0.0265


def filter_points_playmat_cam_frame(pts_m: numpy.ndarray, T_cam_robot: numpy.ndarray) -> numpy.ndarray:
    """Keep points whose robot/world position lies inside the tag-bounded play mat."""
    if pts_m.shape[0] == 0:
        return pts_m
    T_robot_cam = numpy.linalg.inv(numpy.asarray(T_cam_robot, dtype=numpy.float64))
    hom = numpy.ones((pts_m.shape[0], 4), dtype=numpy.float64)
    hom[:, :3] = pts_m
    pr = (T_robot_cam @ hom.T).T[:, :3]
    z0, z1 = WORKSPACE_Z_ROBOT_M
    m = (
        (pr[:, 0] >= PLAY_AREA_X_MIN_M)
        & (pr[:, 0] <= PLAY_AREA_X_MAX_M)
        & (pr[:, 1] >= PLAY_AREA_Y_MIN_M)
        & (pr[:, 1] <= PLAY_AREA_Y_MAX_M)
        & (pr[:, 2] >= z0)
        & (pr[:, 2] <= z1)
    )
    return pts_m[m]


def is_center_on_playmat_robot(p_robot_xyz: numpy.ndarray) -> bool:
    x, y, z = float(p_robot_xyz[0]), float(p_robot_xyz[1]), float(p_robot_xyz[2])
    z0, z1 = WORKSPACE_Z_ROBOT_M
    return (
        (PLAY_AREA_X_MIN_M <= x <= PLAY_AREA_X_MAX_M)
        and (PLAY_AREA_Y_MIN_M <= y <= PLAY_AREA_Y_MAX_M)
        and (z0 <= z <= z1)
    )


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
    Segment tabletop cubes: voxel, outliers, RANSAC table plane, DBSCAN, score.

    Returns ``(clusters, message, plane_np)`` where ``plane_np`` is the first table
    plane (4-vector) for physical cube pose on the largest cubes.
    """
    if len(pcd.points) < 150:
        return None, "too few points after NaN filter", None

    pcd = pcd.voxel_down_sample(voxel_size=0.003)
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=25, std_ratio=2.0)

    if len(pcd.points) < 100:
        return None, "too few points after voxel/outlier", None

    plane_model, inliers = pcd.segment_plane(
        distance_threshold=0.010,
        ransac_n=3,
        num_iterations=1500,
    )
    plane_np = numpy.asarray(plane_model, dtype=numpy.float64).copy()
    if len(inliers) > 0 and len(inliers) > 0.12 * len(pcd.points):
        pcd = pcd.select_by_index(inliers, invert=True)

    if len(pcd.points) < 80:
        return None, "nothing left after plane removal", plane_np

    labels = numpy.asarray(
        pcd.cluster_dbscan(eps=0.020, min_points=25, print_progress=False)
    )
    max_label = int(labels.max()) if labels.size else -1
    if max_label < 0:
        return None, "DBSCAN found no clusters", plane_np

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

    if len(scored_clusters) > 0:
        top_clusters = [c for (_, c) in scored_clusters[:num_cubes]]
        return top_clusters, f"{len(top_clusters)} cube-like clusters", plane_np

    fallback_clusters.sort(key=lambda x: x[0], reverse=True)

    if len(fallback_clusters) > 0:
        top_clusters = [c for (_, c) in fallback_clusters[:num_cubes] if len(c.points) >= 40]
        if len(top_clusters) > 0:
            return top_clusters, f"fallback: {len(top_clusters)} largest clusters", plane_np

    return None, "no cluster passed filters", plane_np


def extract_dense_cluster_points(
    full_pts_m: numpy.ndarray,
    cluster: o3d.geometry.PointCloud,
    margin_m: float,
) -> numpy.ndarray:
    """Re-crop full-resolution cloud around cluster OBB (more points for 30 mm pose)."""
    pts = numpy.asarray(cluster.points, dtype=numpy.float64)
    if pts.shape[0] == 0:
        return pts
    lo = numpy.min(pts, axis=0) - margin_m
    hi = numpy.max(pts, axis=0) + margin_m
    m = numpy.all((full_pts_m >= lo) & (full_pts_m <= hi), axis=1)
    dense = full_pts_m[m]
    if dense.shape[0] < 45:
        return pts
    return dense

def get_cube_transform(cube_pcd, camera_pose):
    if cube_pcd is None or len(cube_pcd.points) < 30:
        return None

    # --- 1. Extract OBB and center ---
    obb = cube_pcd.get_oriented_bounding_box()
    center = numpy.asarray(obb.center)

    # --- 2. Raw PCA rotation ---
    R_raw = numpy.array(obb.R, dtype=float, copy=True)

    # --- 3. Compute world Z axis in camera frame ---
    # camera_pose = T_cam_robot, so invert to get T_robot_cam
    R_cam_robot = camera_pose[:3, :3]
    z_world_cam = R_cam_robot @ numpy.array([0, 0, 1], float)
    z_axis = z_world_cam / numpy.linalg.norm(z_world_cam)

    # --- 4. Project PCA X axis into table plane ---
    x_raw = numpy.array(R_raw[:, 0], float)
    x_proj = x_raw - numpy.dot(x_raw, z_axis) * z_axis
    x_proj /= numpy.linalg.norm(x_proj)

    # --- 5. Extract yaw from projected X axis ---
    yaw = numpy.arctan2(x_proj[1], x_proj[0])

    # --- 6. Build clean rotation matrix (roll=pitch=0) ---
    cy, sy = numpy.cos(yaw), numpy.sin(yaw)
    R_fixed = numpy.array([
        [ cy, -sy, 0.0],
        [ sy,  cy, 0.0],
        [0.0, 0.0, 1.0]
    ])

    # --- 7. Build camera-frame transform ---
    t_cam_cube = numpy.eye(4)
    t_cam_cube[:3, :3] = R_fixed
    t_cam_cube[:3, 3] = center

    # --- 8. Convert to robot frame ---
    t_robot_cam = numpy.linalg.inv(camera_pose)
    t_robot_cube = t_robot_cam @ t_cam_cube

    return (t_robot_cube, t_cam_cube)


def pose_large_cube_physical(
    cluster: o3d.geometry.PointCloud,
    full_pts_m: numpy.ndarray,
    plane_model: numpy.ndarray,
    camera_pose: numpy.ndarray,
) -> tuple[numpy.ndarray, numpy.ndarray] | None:
    """
    Bottom-face footprint + min-area yaw + bound centering at 30 mm — better XY/Z than OBB
    for the biggest cubes when the table plane is trustworthy.
    """
    pts = extract_dense_cluster_points(
        full_pts_m, cluster, margin_m=LARGE_DENSE_CROP_MARGIN_M
    )
    if pts.shape[0] < 35:
        pts = numpy.asarray(cluster.points, dtype=numpy.float64)
    if pts.shape[0] < 30:
        return None
    return physical_cube_pose_from_points(
        pts,
        plane_model,
        camera_pose,
        BIG_CUBE_SIZE,
        bottom_layer_frac=0.11,
        bound_center_iters=5,
        yaw_source="blend",
    )


def get_cube_transforms(observation, camera_intrinsic, camera_pose):
    """
    Estimate cube pose from image + point cloud (geometry; image for API only).

    Returns
    -------
    tuple
        ``(list_of_(t_robot, t_cam), status_str)`` on success (list may be short).
        ``(None, status_str)`` on hard failure (no cloud / clustering failed).
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
    zm = valid_points_m[:, 2]
    valid_points_m = valid_points_m[(zm > GPU_Z_MIN_M) & (zm < GPU_Z_MAX_M)]
    if valid_points_m.shape[0] < 100:
        return None, f"too few points after Z band: {valid_points_m.shape[0]}"

    valid_points_m = filter_points_playmat_cam_frame(valid_points_m, camera_pose)
    if valid_points_m.shape[0] < 100:
        return None, f"too few points after play-mat crop: {valid_points_m.shape[0]}"

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(valid_points_m)

    cube_pcds, status, plane_np = isolate_cube_cluster_open3d(pcd, num_cubes=STACK_HEIGHT_GOAL)
    if cube_pcds is None:
        return None, status or "clustering failed"

    candidates: list[tuple[float, tuple[numpy.ndarray, numpy.ndarray]]] = []

    for cp in cube_pcds:
        obb = cp.get_oriented_bounding_box()
        extent = numpy.asarray(obb.extent)
        size = float(numpy.max(extent))

        t = get_cube_transform(cp, camera_pose)
        if t is None:
            continue
        t_robot, _t_cam = t
        if not is_center_on_playmat_robot(numpy.asarray(t_robot[:3, 3], dtype=numpy.float64)):
            continue
        candidates.append((size, t))

    candidates.sort(key=lambda x: x[0], reverse=True)
    candidates = candidates[:STACK_HEIGHT_GOAL]

    out: list[tuple[numpy.ndarray, numpy.ndarray]] = []
    n_refined = 0
    for rank, (_size, t_obb) in enumerate(candidates):
        # Match sparse cluster by camera-frame center (clusters are in camera frame).
        obb_c = numpy.asarray(t_obb[1][:3, 3], dtype=numpy.float64)
        best_cp = None
        best_d = 1e9
        for c in cube_pcds:
            c_ctr = numpy.asarray(c.get_oriented_bounding_box().center, dtype=numpy.float64)
            d = float(numpy.linalg.norm(c_ctr - obb_c))
            if d < best_d:
                best_d, best_cp = d, c
        cp = best_cp

        use_refined = (
            rank < BIG_CUBES_GOAL
            and plane_np is not None
            and cp is not None
            and float(_size) >= LARGE_EXTENT_MIN_M
        )
        t_final = t_obb
        if use_refined:
            phys = pose_large_cube_physical(cp, valid_points_m, plane_np, camera_pose)
            if phys is not None:
                t_robot_p, _t_cam_p = phys
                if is_center_on_playmat_robot(
                    numpy.asarray(t_robot_p[:3, 3], dtype=numpy.float64)
                ):
                    t_final = phys
                    n_refined += 1
        out.append(t_final)

    msg = f"{len(out)} cube(s) on play mat; {n_refined} large pose(s) refined (30 mm physical)"
    return out, msg


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
        return [], disp

    t_cam_robot = get_transform_camera_robot(cv_image, camera_intrinsic)
    if t_cam_robot is None:
        disp = draw_status_overlay(
            cv_image,
            ["Calibration FAILED (checkpoint0 tags / PnP)"],
            (0, 0, 255),
        )
        return [], disp

    cube_transforms, status = get_cube_transforms(
        (cv_image, point_cloud), camera_intrinsic, t_cam_robot
    )
    if cube_transforms is None:
        cube_transforms = []
        lines = [status or "perception failed"]
        disp = draw_status_overlay(cv_image.copy(), lines, (0, 0, 255))
        return [], disp

    lines = [
        status,
        "play mat crop + Z band; poses filtered to mat centers",
    ]
    disp = cv_image.copy()
    for transform in cube_transforms:
        _t_robot, t_cam_cube = transform
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

        cv2.namedWindow("Verifying Cube Pose", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Verifying Cube Pose", 1280, 720)
        cv2.imshow("Verifying Cube Pose", disp)
        key = cv2.waitKey(0)

        need = STACK_HEIGHT_GOAL
        if len(cube_transforms) < need:
            print(
                f"Need {need} cubes in play area, got {len(cube_transforms)} — adjust scene or tags."
            )
            if key == ord("k"):
                print("Not running motion with incomplete detections.")
            cv2.destroyAllWindows()
            return

        t_robot_cube, _t_cam_cube = cube_transforms[0]

        if key == ord("k") and t_robot_cube is not None:
            cv2.destroyAllWindows()
            xyz = t_robot_cube[:3, 3]
            print(
                f"Cube in robot frame (m): x={xyz[0]:.3f}, y={xyz[1]:.3f}, z={xyz[2]:.3f}"
            )

            X_FIXED = 0.370
            Y_FIXED = 0.020

            # Copy the detected transform
            t_robot_stack = t_robot_cube.copy()

            # Overwrite only X and Y
            t_robot_stack[0, 3] = X_FIXED
            t_robot_stack[1, 3] = Y_FIXED

            for i in range(BIG_CUBES_GOAL):
                t_robot_cube, _ = cube_transforms[i]
                grasp_cube(arm, t_robot_cube)
                place_cube(arm, t_robot_stack)
                t_robot_stack[2, 3] += BIG_CUBE_SIZE
            
            for i in range(BIG_CUBES_GOAL, BIG_CUBES_GOAL + MID_CUBES_GOAL):
                t_robot_cube, _ = cube_transforms[i]
                grasp_cube(arm, t_robot_cube)
                place_cube(arm, t_robot_stack)
                t_robot_stack[2, 3] += MID_CUBE_SIZE

            for i in range(BIG_CUBES_GOAL + MID_CUBES_GOAL, BIG_CUBES_GOAL + MID_CUBES_GOAL + LIL_CUBES_GOAL):
                t_robot_cube, _ = cube_transforms[i]
                grasp_cube(arm, t_robot_cube)
                place_cube(arm, t_robot_stack)
                t_robot_stack[2, 3] += LIL_CUBE_SIZE

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
