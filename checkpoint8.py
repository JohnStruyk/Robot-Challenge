import traceback
import cv2, numpy, time
import open3d as o3d
from xarm.wrapper import XArmAPI

from utils.vis_utils import draw_pose_axes
from utils.zed_camera import ZedCamera
from checkpoint0 import get_transform_camera_robot
from checkpoint1 import grasp_cube, place_cube, GRIPPER_LENGTH, robot_ip
from checkpoint6 import (
    draw_status_overlay,
    isolate_cube_cluster_open3d,
)

cube_prompt = "red cube"

# Arena / play mat in world–robot frame (meters), aligned with checkpoint0 tag layout.
# Tag centers span roughly x in [0, 0.38], y in [-0.4, 0.4]; margins keep walls / off-mat out.
PLAY_MAT_X_ROBOT_M = (-0.10, 0.52)
PLAY_MAT_Y_ROBOT_M = (-0.55, 0.55)
PLAY_MAT_Z_ROBOT_M = (-0.12, 0.48)
# Drop top of frame (ceiling / back wall); keep lower area where the mat usually is.
PLAY_MAT_IMAGE_TOP_CROP_FRAC = 0.12


def scale_xyz_to_meters_frame(xyz):
    """Same mm→m heuristic as ``checkpoint6.points_to_meters_open3d`` for a dense HxWx3 cloud."""
    valid = numpy.isfinite(xyz).all(axis=-1)
    if not numpy.any(valid):
        return xyz.astype(numpy.float64), 1.0
    max_abs = float(numpy.nanmax(numpy.abs(xyz[valid])))
    s = 0.001 if max_abs > 50.0 else 1.0
    return (xyz * s).astype(numpy.float64), s


def play_mat_workspace_mask(xyz_m, t_cam_robot):
    """
    Pixels whose 3D point lies on/near the arena in robot/world frame.

    ``t_cam_robot`` is the OpenCV PnP world→camera matrix from checkpoint0
    (same frame as ``TAG_CENTER_COORDINATES``). Points are transformed with
    ``p_world = inv(t_cam_robot) @ p_cam``.
    """
    H, W = xyz_m.shape[:2]
    finite = numpy.isfinite(xyz_m).all(axis=-1)
    T_wc = numpy.linalg.inv(t_cam_robot)
    flat = xyz_m.reshape(-1, 3)
    f = finite.reshape(-1)
    out = numpy.zeros(H * W, dtype=bool)
    idx = numpy.flatnonzero(f)
    if idx.size == 0:
        return out.reshape(H, W)
    ph = numpy.ones((idx.size, 4), dtype=numpy.float64)
    ph[:, :3] = flat[idx]
    pw = (T_wc @ ph.T).T[:, :3]
    x0, x1 = PLAY_MAT_X_ROBOT_M
    y0, y1 = PLAY_MAT_Y_ROBOT_M
    z0, z1 = PLAY_MAT_Z_ROBOT_M
    inside = (
        (pw[:, 0] >= x0)
        & (pw[:, 0] <= x1)
        & (pw[:, 1] >= y0)
        & (pw[:, 1] <= y1)
        & (pw[:, 2] >= z0)
        & (pw[:, 2] <= z1)
    )
    out[idx] = inside
    return out.reshape(H, W)


def image_arena_roi_mask(shape_hw):
    """Boolean mask: keep rows below the top crop (table / mat region vs wall strip)."""
    h = shape_hw[0]
    row = numpy.arange(h, dtype=numpy.int32)[:, None]
    keep = row >= int(PLAY_MAT_IMAGE_TOP_CROP_FRAC * h)
    return numpy.broadcast_to(keep, (h, shape_hw[1]))


def prompt_to_color_name(cube_prompt):
    """Map a phrase like 'blue cube' to 'red' | 'green' | 'blue'."""
    lower = cube_prompt.lower()
    for name in ("red", "green", "blue"):
        if name in lower:
            return name
    raise ValueError(f"Prompt must mention red, green, or blue: {cube_prompt!r}")


def _largest_connected_component(mask_uint8):
    """Keep the largest 8-connected foreground blob; reduces stray specular hits."""
    num, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask_uint8, connectivity=8
    )
    if num <= 1:
        return mask_uint8
    best = 1
    best_area = stats[1, cv2.CC_STAT_AREA]
    for i in range(2, num):
        if stats[i, cv2.CC_STAT_AREA] > best_area:
            best_area = stats[i, cv2.CC_STAT_AREA]
            best = i
    out = numpy.zeros_like(mask_uint8)
    out[labels == best] = 255
    return out


def _hsv_ranges_for_color(color_name, relaxed=False):
    """
    HSV inRange tuples for OpenCV. ``relaxed`` widens S/V and hue bands for hard scenes.

    **Red** and **blue** use color-specific S/V floors (brown rejection vs dark-navy capture).
    Green still uses the shared defaults below.
    """
    if color_name == "red":
        # Strict: drop dull browns (low S) and near-black (low V).
        if relaxed:
            lo_s, lo_v = 45, 38
            h_low_end, h_high_start = 15, 155
        else:
            # Still well above typical brown/wood (S often <55, V can be low).
            lo_s, lo_v = 68, 52
            h_low_end, h_high_start = 11, 164
        hi_s, hi_v = 255, 255
        return [
            ((0, lo_s, lo_v), (h_low_end, hi_s, hi_v)),
            ((h_high_start, lo_s, lo_v), (179, hi_s, hi_v)),
        ]

    if color_name == "blue":
        # Dark / navy cubes: low V; shared green/red floors were too bright (lo_v 35).
        # Allow moderate S so desaturated navy still segments vs black background.
        if relaxed:
            lo_s, lo_v = 22, 12
            h_lo, h_hi = 76, 152
        else:
            lo_s, lo_v = 38, 20
            h_lo, h_hi = 80, 146
        hi_s, hi_v = 255, 255
        return [((h_lo, lo_s, lo_v), (h_hi, hi_s, hi_v))]

    lo_s, hi_s = (40, 255) if not relaxed else (25, 255)
    lo_v, hi_v = (35, 255) if not relaxed else (20, 255)
    if color_name == "green":
        return [((32 if not relaxed else 28, lo_s, lo_v), (95 if not relaxed else 98, hi_s, hi_v))]
    raise ValueError(color_name)


def color_mask_bgr(image_bgra, color_name, relaxed=False):
    """
    Binary mask for cube color: blur -> HSV -> union of ranges -> open/close ->
    largest connected component -> light dilate (fills depth holes at edges).
    """
    bgr = image_bgra[:, :, :3]
    bgr = cv2.GaussianBlur(bgr, (5, 5), 0)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = numpy.zeros(hsv.shape[:2], dtype=numpy.uint8)
    for lo, hi in _hsv_ranges_for_color(color_name, relaxed=relaxed):
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lo, hi))
    k3 = numpy.ones((3, 3), numpy.uint8)
    k5 = numpy.ones((5, 5), numpy.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k3, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k5, iterations=2)
    mask = _largest_connected_component(mask)
    mask = cv2.dilate(mask, k5, iterations=1)
    return mask > 0


def orthonormalize_rotation(R):
    """Project a 3x3 matrix to SO(3) (stable draw / grasp)."""
    U, _, Vt = numpy.linalg.svd(R)
    Rn = U @ Vt
    if numpy.linalg.det(Rn) < 0:
        U[:, -1] *= -1.0
        Rn = U @ Vt
    return Rn


def isolate_masked_color_cube_open3d(pcd: o3d.geometry.PointCloud):
    """
    Segment a **color-masked** cube cloud. Skips RANSAC plane removal (no table in mask).

    Uses tighter voxel / DBSCAN than :func:`checkpoint6.isolate_cube_cluster_open3d`,
    which is tuned for full-scene tabletop data.
    """
    if len(pcd.points) < 32:
        return None, "too few masked points"

    pcd = pcd.voxel_down_sample(voxel_size=0.002)
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=15, std_ratio=2.0)
    if len(pcd.points) < 22:
        return None, "too few after voxel/outlier (masked)"

    labels = numpy.asarray(
        pcd.cluster_dbscan(eps=0.014, min_points=10, print_progress=False)
    )
    max_label = int(labels.max()) if labels.size else -1

    def score_cluster(cluster):
        obb = cluster.get_oriented_bounding_box()
        ext = numpy.sort(numpy.asarray(obb.extent))
        if ext[2] < 1e-9:
            return -1.0, None
        max_dim = float(ext[2])
        mid_dim = float(ext[1])
        min_dim = float(ext[0])
        compact = min_dim / max_dim if max_dim > 0 else 0.0
        # Prefer compact ~cube-ish boxes in expected size band (lab cubes ~2–5 cm).
        size_ok = 0.007 <= max_dim <= 0.12
        span_ok = abs(max_dim - mid_dim) / max_dim < 0.75 if max_dim > 1e-6 else False
        qual = 1.0 if size_ok else 0.15
        qual *= 1.0 if (0.2 < compact <= 1.0) else 0.4
        qual *= 1.1 if span_ok else 0.85
        return float(len(cluster.points)) * compact * qual, cluster

    if max_label < 0:
        # DBSCAN marked everything noise — often one tight blob; use whole cloud.
        sc, chosen = score_cluster(pcd)
        if chosen is not None and sc > 0:
            return pcd, "masked monolithic blob"
        return pcd, "masked blob (weak score)"

    best_cluster = None
    best_score = -1.0
    fallback_largest = None
    fallback_n = 0

    for cid in range(max_label + 1):
        idx = numpy.where(labels == cid)[0]
        if idx.size < 12:
            continue
        cluster = pcd.select_by_index(idx)
        sc, chosen = score_cluster(cluster)
        if idx.size > fallback_n:
            fallback_n = idx.size
            fallback_largest = cluster
        if sc > best_score and chosen is not None:
            best_score = sc
            best_cluster = chosen

    if best_cluster is not None:
        return best_cluster, "masked DBSCAN cluster"

    if fallback_largest is not None and len(fallback_largest.points) >= 18:
        return fallback_largest, "masked fallback: largest cluster"

    return None, "no masked cluster"


def camera_pose_from_cluster_pcd(cube_pcd: o3d.geometry.PointCloud):
    """
    Cube pose in camera frame: blend OBB center with point centroid (reduces depth noise),
    rotation from OBB with SVD projection to SO(3).
    """
    pts = numpy.asarray(cube_pcd.points)
    centroid = pts.mean(axis=0)
    obb = cube_pcd.get_oriented_bounding_box()
    c_obb = numpy.asarray(obb.center)
    center = 0.62 * centroid + 0.38 * c_obb
    R = orthonormalize_rotation(numpy.asarray(obb.R))
    t_cam_cube = numpy.eye(4)
    t_cam_cube[:3, :3] = R
    t_cam_cube[:3, 3] = center
    return t_cam_cube


class CubePoseDetector:
    """
    Pure-vision target selection: prompt -> 2D color mask -> arena workspace + ROI ->
    masked 3D points -> cluster -> pose.

    Color hits outside the AprilTag-defined play mat (or in the top image strip) are
    dropped so background walls / clutter are ignored.
    """

    def __init__(self, camera_intrinsic):
        self.camera_intrinsic = camera_intrinsic
        self.t_cam_robot = None

    def set_camera_pose(self, t_cam_robot):
        """4x4 from checkpoint0 (robot/world -> camera)."""
        self.t_cam_robot = t_cam_robot

    def get_transforms(self, observation, cube_prompt):
        """
        Returns
        -------
        tuple
            On success: ``((t_robot_cube, t_cam_cube), status_message)``.
            On failure: ``(None, status_message)``.
        """
        if self.t_cam_robot is None:
            return None, "camera pose not set (call set_camera_pose)"

        image, point_cloud = observation
        if image is None or point_cloud is None:
            return None, "missing image or point_cloud"

        try:
            color_name = prompt_to_color_name(cube_prompt)
        except ValueError as exc:
            return None, str(exc)

        if image.shape[:2] != point_cloud.shape[:2]:
            return None, "image / point_cloud shape mismatch"

        xyz = point_cloud[..., :3]
        xyz_m, _ = scale_xyz_to_meters_frame(xyz)
        finite = numpy.isfinite(xyz_m).all(axis=-1)

        ws = play_mat_workspace_mask(xyz_m, self.t_cam_robot)
        roi = image_arena_roi_mask(image.shape)
        scene = finite & ws & roi

        min_pts = 55
        mask_2d = color_mask_bgr(image, color_name, relaxed=False)
        combined = scene & mask_2d
        pts = xyz_m[combined]
        seg_note = "strict mask+arena"
        if pts.shape[0] < min_pts:
            mask_2d = color_mask_bgr(image, color_name, relaxed=True)
            combined = scene & mask_2d
            pts = xyz_m[combined]
            seg_note = "relaxed mask+arena"

        if pts.shape[0] < min_pts:
            return None, (
                f"too few points on play mat: {pts.shape[0]} ({seg_note})"
            )

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts.astype(numpy.float64))

        cube_pcd, seg_msg = isolate_masked_color_cube_open3d(pcd)
        if cube_pcd is None or len(cube_pcd.points) < 18:
            cube_pcd, seg_msg = isolate_cube_cluster_open3d(pcd)
            seg_note = f"{seg_note}; fallback cp6: {seg_msg}"
        else:
            seg_note = f"{seg_note}; {seg_msg}"

        if cube_pcd is None or len(cube_pcd.points) < 15:
            return None, seg_msg if cube_pcd is None else "cluster too small"

        t_cam_cube = camera_pose_from_cluster_pcd(cube_pcd)

        t_robot_cam = numpy.linalg.inv(self.t_cam_robot)
        t_robot_cube = t_robot_cam @ t_cam_cube

        return (t_robot_cube, t_cam_cube), f"{color_name}: {seg_note}"


def run_pure_vision_target_perception(
    cv_image, point_cloud, camera_intrinsic, prompt
):
    """
    Same pattern as checkpoint6.run_pure_vision_perception, but prompt-based target selection.

    Returns
    -------
    tuple
        ``(t_robot_cube, t_cam_cube, display_bgra, status_message)``
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

    detector = CubePoseDetector(camera_intrinsic)
    detector.set_camera_pose(t_cam_robot)

    try:
        result = detector.get_transforms((cv_image, point_cloud), prompt)
        if result is None or result[0] is None:
            msg = result[1] if isinstance(result, tuple) and len(result) > 1 else "unknown"
            disp = draw_status_overlay(
                cv_image,
                [f"Target ({prompt}): {msg}"],
                (0, 165, 255),
            )
            return None, None, disp, msg

        (t_robot_cube, t_cam_cube), seg_msg = result
        lines = [
            f"Prompt: {prompt}",
            f"Segmentation: {seg_msg}",
            "OK - press k to run grasp/place, any other key to quit",
        ]
        disp = cv_image.copy()
        if t_cam_cube is not None and numpy.isfinite(t_cam_cube).all():
            draw_pose_axes(disp, camera_intrinsic, t_cam_cube)
        disp = draw_status_overlay(disp, lines, (0, 220, 0))
        return t_robot_cube, t_cam_cube, disp, seg_msg
    except Exception as exc:
        traceback.print_exc()
        disp = draw_status_overlay(
            cv_image,
            [f"Exception: {exc!s}"],
            (0, 0, 255),
        )
        return None, None, disp, str(exc)


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
        t_robot_cube, _t_cam_cube, disp, _status = run_pure_vision_target_perception(
            cv_image, point_cloud, camera_intrinsic, cube_prompt
        )

        cv2.namedWindow("Verifying Cube Pose", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Verifying Cube Pose", 1280, 720)
        cv2.imshow("Verifying Cube Pose", disp)
        key = cv2.waitKey(0)

        if key == ord("k") and t_robot_cube is not None:
            cv2.destroyAllWindows()
            xyz = t_robot_cube[:3, 3]
            print(
                f"Target in robot frame (m): x={xyz[0]:.3f}, y={xyz[1]:.3f}, z={xyz[2]:.3f}"
            )
            grasp_cube(arm, t_robot_cube)
            place_cube(arm, t_robot_cube)
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
