from __future__ import annotations

import cv2
import numpy
import time
from typing import NamedTuple, Optional, Tuple
from xarm.wrapper import XArmAPI

from utils.vis_utils import draw_pose_axes
from utils.zed_camera import ZedCamera
from checkpoint0 import get_transform_camera_robot
from checkpoint1 import grasp_cube, place_cube, GRIPPER_LENGTH, robot_ip
from checkpoint4 import STACK_HEIGHT
from checkpoint6 import draw_status_overlay
from checkpoint8 import CubePoseDetector

WINDOW_NAME = "Stacking (cp9)"

class StackingView(NamedTuple):
    """Pure data for one preview frame (no I/O)."""

    display_bgra: numpy.ndarray
    ready: bool
    t_robot_red: Optional[numpy.ndarray]
    t_robot_green: Optional[numpy.ndarray]


def robot_pose_from_detection(
    result: Optional[Tuple[object, ...]],
) -> Optional[numpy.ndarray]:
    if result is None or result[0] is None:
        return None
    (t_robot, _), _ = result
    return t_robot


def stack_pose_above_green(
    t_robot_green: numpy.ndarray, stack_height_m: float
) -> numpy.ndarray:
    """Pure: copy green pose and offset Z by stack height (meters)."""
    out = numpy.copy(t_robot_green)
    out[2, 3] += stack_height_m
    return out


def build_dual_cube_view(
    cv_image: numpy.ndarray,
    point_cloud: numpy.ndarray,
    camera_intrinsic: numpy.ndarray,
    detector: CubePoseDetector,
    stack_height_m: float,
) -> StackingView:
    """
    After calibration: run red/green detection and compose overlay (no windows).
    """
    obs = (cv_image, point_cloud)
    res_red = detector.get_transforms(obs, "red cube")
    res_green = detector.get_transforms(obs, "green cube")

    ok_red = res_red is not None and res_red[0] is not None
    ok_green = res_green is not None and res_green[0] is not None

    disp = cv_image.copy()
    lines = [
        f"STACK_HEIGHT = {stack_height_m:.4f} m (checkpoint4)",
        f"red: {res_red[1] if res_red else 'n/a'}",
        f"green: {res_green[1] if res_green else 'n/a'}",
    ]

    t_robot_red = robot_pose_from_detection(res_red)
    t_robot_green = robot_pose_from_detection(res_green)

    if ok_red:
        (_tr, t_cam_red), _ = res_red
        draw_pose_axes(disp, camera_intrinsic, t_cam_red)
    if ok_green:
        (_tg, t_cam_green), _ = res_green
        draw_pose_axes(disp, camera_intrinsic, t_cam_green)

    ready = ok_red and ok_green
    if ready:
        lines.append(
            "OK - press k to grasp RED and stack on GREEN, else any key to quit"
        )
        color = (0, 220, 0)
    else:
        lines.append("Need both cubes — fix lighting/HSV or scene")
        color = (0, 165, 255)

    disp = draw_status_overlay(disp, lines, color)
    return StackingView(disp, ready, t_robot_red, t_robot_green)


def show_stacking_window(display_bgra: numpy.ndarray) -> int:
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 1280, 720)
    cv2.imshow(WINDOW_NAME, display_bgra)
    return cv2.waitKey(0)


def run_stack_sequence(
    arm: XArmAPI,
    t_robot_red: numpy.ndarray,
    t_robot_green: numpy.ndarray,
    stack_height_m: float,
) -> None:
    print("Picking up RED cube...")
    grasp_cube(arm, t_robot_red)
    target = stack_pose_above_green(t_robot_green, stack_height_m)
    print("Placing on GREEN (stack pose)...")
    place_cube(arm, target)
    arm.stop_lite6_gripper()


def main():
    zed = ZedCamera()
    camera_intrinsic = zed.camera_intrinsic

    cube_pose_detector = CubePoseDetector(camera_intrinsic)

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

        if cv_image is None:
            blank = numpy.zeros((720, 1280, 4), dtype=numpy.uint8)
            disp = draw_status_overlay(blank, ["ZED image is None"], (0, 0, 255))
            show_stacking_window(disp)
            return

        t_cam_robot = get_transform_camera_robot(cv_image, camera_intrinsic)
        if t_cam_robot is None:
            disp = draw_status_overlay(
                cv_image,
                ["Calibration FAILED (checkpoint0 tags / PnP)"],
                (0, 0, 255),
            )
            show_stacking_window(disp)
            return

        cube_pose_detector.set_camera_pose(t_cam_robot)
        view = build_dual_cube_view(
            cv_image,
            point_cloud,
            camera_intrinsic,
            cube_pose_detector,
            STACK_HEIGHT,
        )
        key = show_stacking_window(view.display_bgra)

        if key == ord("k") and view.ready:
            cv2.destroyAllWindows()
            run_stack_sequence(
                arm,
                view.t_robot_red,
                view.t_robot_green,
                STACK_HEIGHT,
            )
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
