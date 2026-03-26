import cv2, numpy, time
import itertools
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

cube_prompt = 'red cube'

class CubePoseDetector:
    def __init__(self, camera_intrinsic):
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
                (numpy.array([85, 60, 40]), numpy.array([145, 255, 255])),
            ],
        }

    def set_camera_pose(self, t_cam_robot):
        self.t_cam_robot = t_cam_robot

    def _prompt_to_color(self, cube_prompt):
        lower = cube_prompt.lower()
        for color in ('red', 'green', 'blue'):
            if color in lower:
                return color
        raise ValueError(f"Unsupported prompt '{cube_prompt}'. Use red/green/blue cube.")

    def _color_ratio_in_patch(self, hsv, u, v, target_color, patch_radius=14):
        h, w = hsv.shape[:2]
        u0, u1 = max(0, u - patch_radius), min(w, u + patch_radius + 1)
        v0, v1 = max(0, v - patch_radius), min(h, v + patch_radius + 1)
        patch = hsv[v0:v1, u0:u1]
        if patch.size == 0:
            return 0.0