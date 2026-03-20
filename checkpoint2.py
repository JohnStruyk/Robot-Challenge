import cv2, numpy, time
from xarm.wrapper import XArmAPI

from utils.vis_utils import draw_pose_axes
from utils.zed_camera import ZedCamera
from checkpoint0 import get_transform_camera_robot
from checkpoint1 import grasp_cube, get_transform_cube, GRIPPER_LENGTH

# TODO
BASKET_POSE = None # Measure it using the robot's free drive mode.

robot_ip = ''

def place_in_basket(arm, basket_pose, vaccum_gripper=False):
    """
    Move the robot arm to the basket location and release the grasped object.

    Parameters
    ----------
    arm : xarm.wrapper.XArmAPI
        The initialized XArm API object controlling the Lite6 robot.
    basket_pose : list or numpy.ndarray
        A 6-element array representing the target drop-off pose in the robot 
        base frame formatted as [x, y, z, roll, pitch, yaw]. 
        Translational units (x, y, z) are in meters, and rotational units 
        (roll, pitch, yaw) are in radians.
    vaccum_gripper : bool, optional
        If True, uses the vacuum gripper logic instead of the standard Lite6 
        gripper. Defaults to False.
    """
    # TODO
    pass

def main():

    # Initialize ZED Camera
    zed = ZedCamera()
    camera_intrinsic = zed.camera_intrinsic

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

        t_cam_cube = None
        #TODO
        
        # Visualization
        draw_pose_axes(cv_image, camera_intrinsic, t_cam_cube)
        cv2.namedWindow('Verifying Cube Pose', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Verifying Cube Pose', 1280, 720)
        cv2.imshow('Verifying Cube Pose', cv_image)
        key = cv2.waitKey(0)

        if key == ord('k'):
            cv2.destroyAllWindows()

            # TODO
    
    finally:
        # Close Lite6 Robot
        arm.move_gohome(wait=True)
        time.sleep(0.5)
        arm.disconnect()

        # Close ZED Camera
        zed.close()

if __name__ == "__main__":
    main()
