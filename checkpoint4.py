from checkpoint3 import CubePoseDetector
from checkpoint0 import get_transform_camera_robot

import cv2, time
import numpy as np

from xarm.wrapper import XArmAPI
from utils.vis_utils import draw_pose_axes
from utils.zed_camera import ZedCamera
from checkpoint1 import grasp_cube, place_cube, GRIPPER_LENGTH, robot_ip

# TODO: Determine a suitable height yourself
# The cube is 0.025m. We need to clear the green cube then drop.
CUBE_SIZE = 0.020
STACK_HEIGHT = CUBE_SIZE # + 0.005

def main():

    # Initialize ZED Camera
    zed = ZedCamera()
    camera_intrinsic = zed.camera_intrinsic

    # Initialize Cube Pose Detector
    cube_pose_detector = CubePoseDetector(camera_intrinsic)

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

        # Establish Camera-to-Robot Transform
        t_cam_robot = get_transform_camera_robot(cv_image, camera_intrinsic)

        if t_cam_robot is None:

            print("Failed to find Camera-to-Robot transform.")
            return

        cube_pose_detector.set_camera_pose(t_cam_robot)

        # TODO
        # Locate the RED cube 
        res_red = cube_pose_detector.get_transforms(cv_image, "red cube")
        
        # Locate the GREEN cube (The one we stack on)
        res_green = cube_pose_detector.get_transforms(cv_image, "green cube")

        if res_red is None or res_green is None:

            print("Could not find both cubes.")
            return

        t_robot_red, t_cam_red = res_red
        t_robot_green, t_cam_green = res_green

        # Draw axes for both to verify
        draw_pose_axes(cv_image, camera_intrinsic, t_cam_red)
        draw_pose_axes(cv_image, camera_intrinsic, t_cam_green)

        cv2.imshow('Verifying Cube Poses', cv_image)
        key = cv2.waitKey(0)

        # TODO: Sequence the robot's actions
        if key == ord('k'):

            cv2.destroyAllWindows()

            # --- PICK UP RED ---
            print("Picking up RED cube...")
            grasp_cube(arm, t_robot_red)

            # --- PREPARE STACK POSE ---
            # We take the green cube's location and add our STACK_HEIGHT to the Z axis
            t_robot_stack = np.copy(t_robot_green)
            t_robot_stack[2, 3] += STACK_HEIGHT

            # --- PLACE ON GREEN ---
            print("Stacking on GREEN cube...")
            place_cube(arm, t_robot_stack)

    finally:
        # Close Lite6 Robot
        arm.move_gohome(wait=True)
        time.sleep(0.5)
        arm.disconnect()

        # Close ZED Camera
        zed.close()

if __name__ == "__main__":
    main()