import cv2, numpy, time
from pupil_apriltags import Detector
from xarm.wrapper import XArmAPI
from scipy.spatial.transform import Rotation

from utils.vis_utils import draw_pose_axes
from utils.zed_camera import ZedCamera
from checkpoint0 import get_transform_camera_robot

GRIPPER_LENGTH = 0.067 * 1000
CUBE_TAG_FAMILY = 'tag36h11'
CUBE_TAG_ID = 4
CUBE_TAG_SIZE = 0.02

robot_ip = '192.168.1.182'

# Motion constants (meters / degrees)
SAFE_Z = 0.22
GRASP_Z_OFFSET = 0.0001
LIFT_Z_DELTA = 0.06
PLACE_Z_OFFSET = 0.02

# Keep tool mostly vertical; only yaw is adapted from cube pose.
TOOL_ROLL_DEG = 180.0
TOOL_PITCH_DEG = 0.0

def grasp_cube(arm, cube_pose):
    """
    Execute a pick sequence to grasp a cube at a specified pose.

    Parameters
    ----------
    arm : xarm.wrapper.XArmAPI
        The initialized XArm API object controlling the Lite6 robot.
    cube_pose : numpy.ndarray
        A 4x4 transformation matrix representing the cube's pose in the robot base frame.
        All translational units in this matrix are in meters.
    """
    xyz = cube_pose[:3, 3]
    x_mm, y_mm, z_mm = (xyz * 1000.0).tolist()
    safe_z_mm = SAFE_Z * 1000.0
    grasp_z_mm = z_mm + (GRASP_Z_OFFSET * 1000.0)
    lift_z_mm = max(safe_z_mm, grasp_z_mm + (LIFT_Z_DELTA * 1000.0))

    # Align tool yaw with cube yaw so the parallel jaws are more likely to seat cleanly.
    cube_r = Rotation.from_matrix(cube_pose[:3, :3])
    _, _, cube_yaw_deg = cube_r.as_euler('xyz', degrees=True)

    # Ensure gripper is open before approach.
    arm.open_lite6_gripper()
    #arm.stop_lite6_gripper()
    time.sleep(1)

    # Approach -> descend -> grasp -> lift.
    arm.set_position(x_mm, y_mm, safe_z_mm, TOOL_ROLL_DEG, TOOL_PITCH_DEG, cube_yaw_deg, is_radian=False, wait=True)
    arm.set_position(x_mm, y_mm, grasp_z_mm, TOOL_ROLL_DEG, TOOL_PITCH_DEG, cube_yaw_deg, is_radian=False, wait=True)
    arm.close_lite6_gripper()
    #arm.stop_lite6_gripper()
    time.sleep(1)
    arm.set_position(x_mm, y_mm, lift_z_mm, TOOL_ROLL_DEG, TOOL_PITCH_DEG, cube_yaw_deg, is_radian=False, wait=True)

def place_cube(arm, cube_pose):
    """
    Execute a place sequence to release a cube at a specified pose.

    Parameters
    ----------
    arm : xarm.wrapper.XArmAPI
        The initialized XArm API object controlling the Lite6 robot.
    cube_pose : numpy.ndarray
        A 4x4 transformation matrix representing the target placement pose in the robot base frame.
        All translational units in this matrix are in meters.
    """
    xyz = cube_pose[:3, 3]
    x_mm, y_mm, z_mm = (xyz * 1000.0).tolist()
    safe_z_mm = SAFE_Z * 1000.0
    place_z_mm = z_mm + (PLACE_Z_OFFSET * 1000.0)
    lift_z_mm = max(safe_z_mm, place_z_mm + (LIFT_Z_DELTA * 1000.0))

    cube_r = Rotation.from_matrix(cube_pose[:3, :3])
    _, _, cube_yaw_deg = cube_r.as_euler('xyz', degrees=True)

    arm.set_position(x_mm, y_mm, safe_z_mm, TOOL_ROLL_DEG, TOOL_PITCH_DEG, cube_yaw_deg, is_radian=False, wait=True)
    arm.set_position(x_mm, y_mm, place_z_mm, TOOL_ROLL_DEG, TOOL_PITCH_DEG, cube_yaw_deg, is_radian=False, wait=True)
    arm.open_lite6_gripper()
    #arm.stop_lite6_gripper()
    time.sleep(1)
    arm.set_position(x_mm, y_mm, lift_z_mm, TOOL_ROLL_DEG, TOOL_PITCH_DEG, cube_yaw_deg, is_radian=False, wait=True)

def get_transform_cube(observation, camera_intrinsic, camera_pose):
    """
    Calculate the transformation matrix for the cube relative to the robot base frame, 
    as well as relative to the camera frame.

    This function uses visual fiducial detection to find the cube's pose in the camera's view, 
    then transforms that pose into the robot's global coordinate system. 

    Parameters
    ----------
    observation : numpy.ndarray
        The input image from the camera. Can be a color (BGRA/BGR) or grayscale image.
    camera_intrinsic : numpy.ndarray
        The 3x3 intrinsic camera matrix.
    camera_pose : numpy.ndarray
        A 4x4 transformation matrix representing the camera's pose in the robot base frame (t_cam_robot).
        All translations are in meters.

    Returns
    -------
    tuple or None
        If successful, returns a tuple (t_robot_cube, t_cam_cube) where both 
        are 4x4 transformation matrices with translations in meters. 
        If no cube tag is detected, returns None.
    """
    detector = Detector(families=CUBE_TAG_FAMILY)

    if len(observation.shape) > 2:
        gray = cv2.cvtColor(observation, cv2.COLOR_BGRA2GRAY)
    else:
        gray = observation

    tags = detector.detect(
        gray,
        estimate_tag_pose=True,
        camera_params=(
            float(camera_intrinsic[0, 0]),
            float(camera_intrinsic[1, 1]),
            float(camera_intrinsic[0, 2]),
            float(camera_intrinsic[1, 2]),
        ),
        tag_size=CUBE_TAG_SIZE,
    )

    cube_tag = None
    for tag in tags:
        if tag.tag_id == CUBE_TAG_ID:
            cube_tag = tag
            break

    if cube_tag is None:
        print(f'Cube tag id={CUBE_TAG_ID} not detected.')
        return None

    t_cam_cube = numpy.eye(4)
    t_cam_cube[:3, :3] = cube_tag.pose_R
    t_cam_cube[:3, 3] = cube_tag.pose_t.flatten()

    # checkpoint0 solvePnP output maps robot->camera. We need camera->robot here.
    t_robot_cam = numpy.linalg.inv(camera_pose)
    t_robot_cube = t_robot_cam @ t_cam_cube
    return t_robot_cube, t_cam_cube

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

        # Get Transformation
        t_cam_robot = get_transform_camera_robot(cv_image, camera_intrinsic)
        if t_cam_robot is None:
            return
        
        t_cam_cube = None
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
            place_cube(arm, t_robot_cube)

            arm.stop_lite6_gripper()
    
    finally:
        # Close Lite6 Robot
        arm.stop_lite6_gripper()
        arm.move_gohome(wait=True)
        time.sleep(0.5)
        arm.disconnect()

        # Close ZED Camera
        zed.close()

if __name__ == "__main__":
    main()