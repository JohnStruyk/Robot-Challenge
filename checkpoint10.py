"""
Checkpoint 10 (pure vision): build a three-cube tower from ``stacking_order`` (top → bottom).

Same placement logic as checkpoint 5: stack the middle color on the bottom cube, then the
top color at ``bottom_z + 2 * STACK_HEIGHT`` (see ``checkpoint4.STACK_HEIGHT``).
"""

import cv2
import numpy
import time
from xarm.wrapper import XArmAPI

from utils.vis_utils import draw_pose_axes
from utils.zed_camera import ZedCamera
from checkpoint0 import get_transform_camera_robot
from checkpoint1 import grasp_cube, place_cube, GRIPPER_LENGTH, robot_ip
from checkpoint4 import STACK_HEIGHT
from checkpoint6 import draw_status_overlay
from checkpoint8 import CubePoseDetector

WINDOW_NAME = "Triple stack (cp10)"

# Top → bottom (PDF / arena.yaml ``stack_order_top_to_bottom``).
stacking_order = ["red cube", "green cube", "blue cube"]


def _unpack_pose_result(result):
    """``get_transforms`` -> (t_robot, t_cam) or None."""
    if result is None or result[0] is None:
        return None
    (t_robot, t_cam), _msg = result
    return t_robot, t_cam


def detect_all_cubes(cv_image, point_cloud, camera_intrinsic, detector, prompts):
    """
    Calibrate, run ``get_transforms`` for each prompt, return dicts and composite display.

    Returns
    -------
    tuple
        ``(poses_robot, poses_cam, display_bgra, ok_all, first_error)``
    """
    if cv_image is None:
        blank = numpy.zeros((720, 1280, 4), dtype=numpy.uint8)
        disp = draw_status_overlay(blank, ["ZED image is None"], (0, 0, 255))
        return {}, {}, disp, False, "no image"

    t_cam_robot = get_transform_camera_robot(cv_image, camera_intrinsic)
    if t_cam_robot is None:
        disp = draw_status_overlay(
            cv_image,
            ["Calibration FAILED (checkpoint0 tags / PnP)"],
            (0, 0, 255),
        )
        return {}, {}, disp, False, "calibration failed"

    detector.set_camera_pose(t_cam_robot)
    obs = (cv_image, point_cloud)

    poses_robot = {}
    poses_cam = {}
    lines = [f"STACK_HEIGHT = {STACK_HEIGHT:.4f} m", "Detecting (top → bottom):"]
    missing = []

    for name in prompts:
        raw = detector.get_transforms(obs, name)
        pair = _unpack_pose_result(raw)
        if pair is None:
            err = raw[1] if isinstance(raw, tuple) and len(raw) > 1 else "unknown"
            lines.append(f"  {name}: FAIL ({err})")
            missing.append(name)
            continue
        t_r, t_c = pair
        poses_robot[name] = t_r
        poses_cam[name] = t_c
        lines.append(f"  {name}: OK")

    disp = cv_image.copy()
    for name in prompts:
        if name in poses_cam:
            draw_pose_axes(disp, camera_intrinsic, poses_cam[name])

    ok_all = len(missing) == 0
    if ok_all:
        lines.append("OK — press k to run tower sequence, any other key to quit")
        color = (0, 220, 0)
    else:
        lines.append(f"Missing: {', '.join(missing)}")
        color = (0, 165, 255)

    disp = draw_status_overlay(disp, lines, color)
    return poses_robot, poses_cam, disp, ok_all, None if ok_all else "incomplete"


def run_tower_sequence(arm, poses_robot, order_top_to_bottom):
    """
    Stack middle on bottom, then top on bottom + 2*h (same as checkpoint 5).
    ``order_top_to_bottom`` has length 3: [top, middle, bottom] prompt strings.
    """
    top, middle, bottom = order_top_to_bottom[0], order_top_to_bottom[1], order_top_to_bottom[2]

    print(f"Stacking {middle} on {bottom}...")
    grasp_cube(arm, poses_robot[middle])
    t1 = numpy.copy(poses_robot[bottom])
    t1[2, 3] += STACK_HEIGHT
    place_cube(arm, t1)

    print(f"Stacking {top} on tower (base {bottom})...")
    grasp_cube(arm, poses_robot[top])
    t2 = numpy.copy(poses_robot[bottom])
    t2[2, 3] += 2.0 * STACK_HEIGHT
    place_cube(arm, t2)
    print("Triple stack complete.")
    arm.stop_lite6_gripper()


def main():
    zed = ZedCamera()
    camera_intrinsic = zed.camera_intrinsic
    detector = CubePoseDetector(camera_intrinsic)

    arm = XArmAPI(robot_ip)
    arm.connect()
    arm.motion_enable(enable=True)
    arm.set_tcp_offset([0, 0, GRIPPER_LENGTH, 0, 0, 0])
    arm.set_mode(0)
    arm.set_state(0)
    arm.move_gohome(wait=True)
    time.sleep(0.5)

    try:
        poses_robot, _poses_cam, disp, ok_all, _err = detect_all_cubes(
            zed.image,
            zed.point_cloud,
            camera_intrinsic,
            detector,
            stacking_order,
        )

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, 1280, 720)
        cv2.imshow(WINDOW_NAME, disp)
        key = cv2.waitKey(0)

        if key == ord("k") and ok_all:
            cv2.destroyAllWindows()
            run_tower_sequence(arm, poses_robot, stacking_order)
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
