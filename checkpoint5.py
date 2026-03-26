import cv2
import time
import numpy as np
from xarm.wrapper import XArmAPI

from checkpoint3 import CubePoseDetector
from checkpoint0 import get_transform_camera_robot
from utils.vis_utils import draw_pose_axes
from utils.zed_camera import ZedCamera
from checkpoint1 import grasp_cube, place_cube, GRIPPER_LENGTH, robot_ip
from checkpoint4 import STACK_HEIGHT

# Defined as Top, Middle, Bottom
stacking_order = ['red cube', 'green cube', 'blue cube']

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
        # Get initial observation
        cv_image = zed.image

        # Establish Camera-to-Robot Transform
        t_cam_robot = get_transform_camera_robot(cv_image, camera_intrinsic)
        if t_cam_robot is None:
            print("Failed to find Camera-to-Robot transform.")
            return
        
        cube_pose_detector.set_camera_pose(t_cam_robot)

        # Locate all cubes before moving
        # store them in a dictionary
        cube_poses_robot = {}
        cube_poses_cam = {}
        
        print("Detecting cubes...")
        for name in stacking_order:
            result = cube_pose_detector.get_transforms(cv_image, name)
            if result is not None:
                t_robot_cube, t_cam_cube = result
                cube_poses_robot[name] = t_robot_cube
                cube_poses_cam[name] = t_cam_cube
                # Visual verification for each detected cube
                draw_pose_axes(cv_image, camera_intrinsic, t_cam_cube)
            else:
                print(f"Could not find: {name}")
                return

        # Show the detection to the user
        cv2.imshow('Verifying Triple Stack Poses', cv_image)
        print("Verify axes on screen. Press 'k' to start the sequence.")
        key = cv2.waitKey(0)

        if key == ord('k'):
            cv2.destroyAllWindows()

            # --- STEP 1: STACK GREEN ON BLUE ---
            # Blue is the bottom 
            # Target for Green is Blue's XY + (1 * STACK_HEIGHT)
            print("Moving GREEN to BLUE...")
            grasp_cube(arm, cube_poses_robot['green cube'])
            
            t_target_1 = np.copy(cube_poses_robot['blue cube'])
            t_target_1[2, 3] += STACK_HEIGHT
            place_cube(arm, t_target_1)

            # --- STEP 2: STACK RED ON GREEN ---
            # Target for Red is Blue's XY + (2 * STACK_HEIGHT)
            print("Moving RED to GREEN...")
            grasp_cube(arm, cube_poses_robot['red cube'])
            
            t_target_2 = np.copy(cube_poses_robot['blue cube'])
            t_target_2[2, 3] += (2 * STACK_HEIGHT)
            place_cube(arm, t_target_2)

            print("Sequential Stacking Complete!")

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        # Safety: always bring the arm home and disconnect
        print("Cleaning up...")
        arm.stop_lite6_gripper()
        time.sleep(0.5)
        arm.move_gohome(wait=True)
        time.sleep(0.5)
        arm.disconnect()

        # Close ZED Camera
        zed.close()

if __name__ == "__main__":
    main()