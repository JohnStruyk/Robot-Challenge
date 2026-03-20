import cv2, numpy, time
from xarm.wrapper import XArmAPI

from utils.vis_utils import draw_pose_axes
from utils.zed_camera import ZedCamera
from checkpoint0 import get_transform_camera_robot
from checkpoint1 import grasp_cube, get_transform_cube, GRIPPER_LENGTH, _initialize_gripper

# If measured with the xArm web UI, values are usually in millimeters/degrees.
BASKET_POSE = [230.1, -305.5, 151.1, 178.4, -1.3, -32.8]  # Update using free-drive measurements.

robot_ip = '192.168.1.183'

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
    if basket_pose is None:
        raise ValueError('BASKET_POSE is not set.')
    if len(basket_pose) not in (3, 6):
        raise ValueError('basket_pose must have 3 or 6 elements.')

    if len(basket_pose) == 3:
        x_mm, y_mm, z_mm = [float(v) for v in basket_pose]
        roll_deg, pitch_deg, yaw_deg = 180.0, 0.0, 90.0
    else:
        x_mm, y_mm, z_mm, roll_deg, pitch_deg, yaw_deg = [float(v) for v in basket_pose]

    safe_z_mm = max(220.0, z_mm + 80.0)

    arm.set_position(x_mm, y_mm, safe_z_mm, roll_deg, pitch_deg, yaw_deg, is_radian=False, wait=True)
    arm.set_position(x_mm, y_mm, z_mm, roll_deg, pitch_deg, yaw_deg, is_radian=False, wait=True)

    # This checkpoint is configured for the parallel gripper (not vacuum).
    if vaccum_gripper:
        raise ValueError('vaccum_gripper must be False for this setup (parallel gripper only).')

    if hasattr(arm, 'open_lite6_gripper'):
        arm.open_lite6_gripper()
        time.sleep(0.6)
        if hasattr(arm, 'stop_lite6_gripper'):
            arm.stop_lite6_gripper()
    elif hasattr(arm, 'set_gripper_position'):
        arm.set_gripper_position(850, wait=True)
    else:
        raise RuntimeError('No supported parallel-gripper release method found.')

    arm.set_position(x_mm, y_mm, safe_z_mm, roll_deg, pitch_deg, yaw_deg, is_radian=False, wait=True)

def main():

    # Initialize ZED Camera
    zed = ZedCamera()
    camera_intrinsic = zed.camera_intrinsic

    # Initialize Lite6 Robot
    arm = XArmAPI(robot_ip)
    arm.connect()
    arm.motion_enable(enable=True)
    _initialize_gripper(arm)
    arm.set_tcp_offset([0, 0, GRIPPER_LENGTH, 0, 0, 0])
    arm.set_mode(0)
    arm.set_state(0)
    arm.move_gohome(wait=True)
    time.sleep(0.5)

    try:
        # Get Observation
        cv_image = zed.image

        t_cam_cube = None
        t_cam_robot = get_transform_camera_robot(cv_image, camera_intrinsic)
        if t_cam_robot is None:
            return

        cube_result = get_transform_cube(cv_image, camera_intrinsic, t_cam_robot)
        if cube_result is None:
            return
        t_robot_cube, t_cam_cube = cube_result
        
        # Visualization
        draw_pose_axes(cv_image, camera_intrinsic, t_cam_cube)
        cv2.namedWindow('Verifying Cube Pose', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Verifying Cube Pose', 1280, 720)
        cv2.imshow('Verifying Cube Pose', cv_image)
        key = cv2.waitKey(0)

        if key == ord('k'):
            cv2.destroyAllWindows()

            xyz = t_robot_cube[:3, 3]
            print(f'Cube in robot frame (m): x={xyz[0]:.3f}, y={xyz[1]:.3f}, z={xyz[2]:.3f}')
            grasp_cube(arm, t_robot_cube)
            place_in_basket(arm, BASKET_POSE, vaccum_gripper=False)
    
    finally:
        # Close Lite6 Robot
        arm.move_gohome(wait=True)
        time.sleep(0.5)
        arm.disconnect()

        # Close ZED Camera
        zed.close()

if __name__ == "__main__":
    main()
