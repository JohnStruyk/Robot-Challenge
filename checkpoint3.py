import cv2, numpy, time
from pupil_apriltags import Detector
from xarm.wrapper import XArmAPI

from utils.vis_utils import draw_pose_axes
from utils.zed_camera import ZedCamera
from checkpoint0 import get_transform_camera_robot
from checkpoint1 import (
    grasp_cube,
    place_cube,
    GRIPPER_LENGTH,
    CUBE_TAG_FAMILY,
    CUBE_TAG_SIZE,
    robot_ip,
)

cube_prompt = 'blue cube'

class CubePoseDetector:
    """
    A detector to robustly identify and locate a specific cube in the scene.

    This class leverages text prompts to semantically segment a specific cube (e.g., 
    'blue cube') and determine the cube's pose by the AprilTags.
    """

    def __init__(self, camera_intrinsic):
        """
        Initialize the CubePoseDetector with camera parameters.

        Parameters
        ----------
        camera_intrinsic : numpy.ndarray
            The 3x3 intrinsic camera matrix.
        """
        self.camera_intrinsic = camera_intrinsic
        self.detector = Detector(families=CUBE_TAG_FAMILY)
        self.t_cam_robot = None

        # HSV ranges for OpenCV (H in [0, 179]).
        self.color_ranges = {
            'red': [
                (numpy.array([0, 80, 80]), numpy.array([10, 255, 255])),
                (numpy.array([170, 80, 80]), numpy.array([179, 255, 255])),
            ],
            'green': [
                (numpy.array([35, 80, 80]), numpy.array([85, 255, 255])),
            ],
            'blue': [
                (numpy.array([90, 80, 80]), numpy.array([140, 255, 255])),
            ],
        }

    def set_camera_pose(self, t_cam_robot):
        """
        Set the camera pose returned by checkpoint0 registration.

        Parameters
        ----------
        t_cam_robot : numpy.ndarray
            4x4 solvePnP transform from robot/world frame to camera frame.
        """
        self.t_cam_robot = t_cam_robot

    def _prompt_to_color(self, cube_prompt):
        """
        Convert a text prompt into one of {'red', 'green', 'blue'}.

        Parameters
        ----------
        cube_prompt : str
            Prompt like 'blue cube'.

        Returns
        -------
        str
            Canonical color label.
        """
        lower = cube_prompt.lower()
        for color in ('red', 'green', 'blue'):
            if color in lower:
                return color
        raise ValueError(f"Unsupported prompt '{cube_prompt}'. Use red/green/blue cube.")

    def _is_color_match(self, hsv_pixel, target_color):
        """
        Check whether one HSV pixel lies in a target color range.

        Parameters
        ----------
        hsv_pixel : numpy.ndarray
            Pixel HSV triplet [h, s, v].
        target_color : str
            Target color label.

        Returns
        -------
        bool
            True if the pixel matches the color threshold.
        """
        h, s, v = hsv_pixel
        for low, high in self.color_ranges[target_color]:
            if low[0] <= h <= high[0] and low[1] <= s <= high[1] and low[2] <= v <= high[2]:
                return True
        return False

    def get_transforms(self, observation, cube_prompt):
        """
        Calculate the transformation matrix for a specific prompted cube relative to the robot base frame,
        as well as relative to the camera frame.

        Parameters
        ----------
        observation : numpy.ndarray
            The input image from the camera. Can be a color (BGRA/BGR) or grayscale image.
        cube_prompt : str
            The text prompt used to segment the target object (e.g., 'blue cube').

        Returns
        -------
        tuple or None
            If successful, returns a tuple (t_robot_cube, t_cam_cube) where both 
            are 4x4 transformation matrices with translations in meters. 
            If no matching object or tag is found, returns None.
        """
        if self.t_cam_robot is None:
            raise RuntimeError('Camera pose not initialized. Call set_camera_pose(...) first.')

        if len(observation.shape) > 2:
            bgr = observation[:, :, :3]
            gray = cv2.cvtColor(observation, cv2.COLOR_BGRA2GRAY)
            hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        else:
            gray = observation
            hsv = None

        target_color = self._prompt_to_color(cube_prompt)
        tags = self.detector.detect(
            gray,
            estimate_tag_pose=True,
            camera_params=(
                float(self.camera_intrinsic[0, 0]),
                float(self.camera_intrinsic[1, 1]),
                float(self.camera_intrinsic[0, 2]),
                float(self.camera_intrinsic[1, 2]),
            ),
            tag_size=CUBE_TAG_SIZE,
        )

        # Filter by color at tag center pixel. Choose the closest matching tag.
        matched_tag = None
        min_depth = float('inf')
        for tag in tags:
            if hsv is None:
                continue
            u, v = int(round(tag.center[0])), int(round(tag.center[1]))
            if v < 0 or v >= hsv.shape[0] or u < 0 or u >= hsv.shape[1]:
                continue

            if self._is_color_match(hsv[v, u], target_color):
                depth = float(tag.pose_t.flatten()[2])
                if depth < min_depth:
                    min_depth = depth
                    matched_tag = tag

        if matched_tag is None:
            print(f"No AprilTag matched prompt '{cube_prompt}'.")
            return None

        t_cam_cube = numpy.eye(4)
        t_cam_cube[:3, :3] = matched_tag.pose_R
        t_cam_cube[:3, 3] = matched_tag.pose_t.flatten()

        # checkpoint0 solvePnP output maps robot->camera. Invert for camera->robot.
        t_robot_cam = numpy.linalg.inv(self.t_cam_robot)
        t_robot_cube = t_robot_cam @ t_cam_cube
        return t_robot_cube, t_cam_cube

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

        t_cam_cube = None
        t_cam_robot = get_transform_camera_robot(cv_image, camera_intrinsic)
        if t_cam_robot is None:
            return
        cube_pose_detector.set_camera_pose(t_cam_robot)

        result = cube_pose_detector.get_transforms(cv_image, cube_prompt)
        if result is None:
            return
        t_robot_cube, t_cam_cube = result

        # Visualization
        draw_pose_axes(cv_image, camera_intrinsic, t_cam_cube)
        cv2.namedWindow('Verifying Cube Pose', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Verifying Cube Pose', 1280, 720)
        cv2.imshow('Verifying Cube Pose', cv_image)
        key = cv2.waitKey(0)
    
        if key == ord('k'):
            cv2.destroyAllWindows()

            xyz = t_robot_cube[:3, 3]
            print(f"Target '{cube_prompt}' in robot frame (m): x={xyz[0]:.3f}, y={xyz[1]:.3f}, z={xyz[2]:.3f}")
            grasp_cube(arm, t_robot_cube)
            place_cube(arm, t_robot_cube)
            
    finally:
        # Close Lite6 Robot
        arm.move_gohome(wait=True)
        time.sleep(0.5)
        arm.disconnect()

        # Close ZED Camera
        zed.close()

if __name__ == "__main__":
    main()
