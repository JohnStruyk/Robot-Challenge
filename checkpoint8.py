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
    points_to_meters_open3d,
)

cube_prompt = "red cube"


def prompt_to_color_name(cube_prompt):
    """Map a phrase like 'blue cube' to 'red' | 'green' | 'blue'."""
    lower = cube_prompt.lower()
    for name in ("red", "green", "blue"):
        if name in lower:
            return name
    raise ValueError(f"Prompt must mention red, green, or blue: {cube_prompt!r}")


def color_mask_bgr(image_bgra, color_name):
    """
    Build a binary mask for the given cube color in the BGRA image (HSV thresholds).
    """
    bgr = image_bgra[:, :, :3]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = numpy.zeros(hsv.shape[:2], dtype=numpy.uint8)
    if color_name == "red":
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, (0, 70, 50), (12, 255, 255)))
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, (165, 70, 50), (179, 255, 255)))
    elif color_name == "green":
        mask = cv2.inRange(hsv, (35, 50, 50), (92, 255, 255))
    elif color_name == "blue":
        mask = cv2.inRange(hsv, (90, 50, 50), (135, 255, 255))
    else:
        raise ValueError(color_name)
    kernel = numpy.ones((5, 5), numpy.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.dilate(mask, kernel, iterations=1)
    return mask > 0


class CubePoseDetector:
    """
    Pure-vision target selection: prompt -> 2D color mask -> masked 3D points -> OBB (Open3D).

    Aligns with the PDF: combine color segmentation (checkpoint 3 style) with
    point-cloud pose (checkpoint 6 style); filter NaNs before Open3D.
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

        mask_2d = color_mask_bgr(image, color_name)
        xyz = point_cloud[..., :3]
        finite = numpy.isfinite(xyz).all(axis=-1)
        combined = finite & mask_2d
        pts = xyz[combined]

        if pts.shape[0] < 80:
            return None, f"too few masked 3D points: {pts.shape[0]}"

        pts_m, _ = points_to_meters_open3d(pts)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts_m.astype(numpy.float64))

        cube_pcd, seg_msg = isolate_cube_cluster_open3d(pcd)
        if cube_pcd is None or len(cube_pcd.points) < 25:
            return None, seg_msg

        obb = cube_pcd.get_oriented_bounding_box()
        center = numpy.asarray(obb.center)
        R_cam_cube = numpy.asarray(obb.R)

        t_cam_cube = numpy.eye(4)
        t_cam_cube[:3, :3] = R_cam_cube
        t_cam_cube[:3, 3] = center

        t_robot_cam = numpy.linalg.inv(self.t_cam_robot)
        t_robot_cube = t_robot_cam @ t_cam_cube

        return (t_robot_cube, t_cam_cube), f"{color_name}: {seg_msg}"


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
