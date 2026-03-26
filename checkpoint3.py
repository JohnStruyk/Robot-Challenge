
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
                (numpy.array([85, 60, 40]), numpy.array([145, 255, 255])),
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

    def _color_ratio_in_patch(self, hsv, u, v, target_color, patch_radius=10):
        """
        Compute the fraction of pixels matching a target color in a local patch.

        Parameters
        ----------
        hsv : numpy.ndarray
            HSV image.
        u : int
            Patch center x pixel coordinate.
        v : int
            Patch center y pixel coordinate.
        target_color : str
            Target color label.
        patch_radius : int, optional
            Half-size of the square patch around (u, v).

        Returns
        -------
        float
            Ratio in [0, 1] of pixels classified as target color.
        """
        h, w = hsv.shape[:2]
        u0, u1 = max(0, u - patch_radius), min(w, u + patch_radius + 1)
        v0, v1 = max(0, v - patch_radius), min(h, v + patch_radius + 1)
        patch = hsv[v0:v1, u0:u1]
        if patch.size == 0:
            return 0.0

        mask = numpy.zeros(patch.shape[:2], dtype=bool)
        for low, high in self.color_ranges[target_color]:
            m = (
                (patch[:, :, 0] >= low[0]) & (patch[:, :, 0] <= high[0]) &
                (patch[:, :, 1] >= low[1]) & (patch[:, :, 1] <= high[1]) &
                (patch[:, :, 2] >= low[2]) & (patch[:, :, 2] <= high[2])
            )
            mask |= m
        return float(mask.mean())

    def _build_color_mask(self, hsv, target_color):
        """
        Build a cleaned binary mask for the target cube color over the full image.

        Parameters
        ----------
        hsv : numpy.ndarray
            HSV image.
        target_color : str
            Target color label.

        Returns
        -------
        numpy.ndarray
            uint8 mask with 255 at target-color pixels.
        """
        mask = numpy.zeros(hsv.shape[:2], dtype=numpy.uint8)
        used_cuda = False
        # Use CUDA path when available (NVIDIA GPU), fallback to CPU.
        if hasattr(cv2, "cuda") and cv2.cuda.getCudaEnabledDeviceCount() > 0:
            try:
                gpu_hsv = cv2.cuda_GpuMat()
                gpu_hsv.upload(hsv)
                gpu_mask = None
                for low, high in self.color_ranges[target_color]:
                    low_b = tuple(int(x) for x in low.tolist())
                    high_b = tuple(int(x) for x in high.tolist())
                    partial = cv2.cuda.inRange(gpu_hsv, low_b, high_b)
                    gpu_mask = partial if gpu_mask is None else cv2.cuda.bitwise_or(gpu_mask, partial)
                if gpu_mask is not None:
                    mask = gpu_mask.download()
                    used_cuda = True
            except Exception:
                used_cuda = False

        if not used_cuda:
            for low, high in self.color_ranges[target_color]:
                partial = cv2.inRange(hsv, low, high)
                mask = cv2.bitwise_or(mask, partial)

        # Morphological cleanup: remove speckles and fill small holes.
        kernel = numpy.ones((5, 5), dtype=numpy.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        return mask
    
    def visualize_detections(self, hsv, bgr, tags):
        """
        Debug helper to show what the detector 'sees' for every visible tag.
        """
        debug_img = bgr.copy()
        print(f"\n--- Debug: Analyzing {len(tags)} detected tags ---")
        
        for i, tag in enumerate(tags):
            u, v = int(round(tag.center[0])), int(round(tag.center[1]))
            patch_radius = 15
            
            # Extract crop for display
            h, w = hsv.shape[:2]
            u0, u1 = max(0, u - patch_radius), min(w, u + patch_radius + 1)
            v0, v1 = max(0, v - patch_radius), min(h, v + patch_radius + 1)
            patch_bgr = bgr[v0:v1, u0:u1]

            # Calculate scores for all colors
            scores = {c: self._color_ratio_in_patch(hsv, u, v, c, patch_radius) for c in self.color_ranges}
            best_color = max(scores, key=scores.get)
            max_score = scores[best_color]

            # Logic for debug label
            if max_score < 0.15:
                label = f"ID:{tag.tag_id} NEUTRAL (Score:{max_score:.2f})"
                color_bgr = (128, 128, 128)
            else:
                label = f"ID:{tag.tag_id} {best_color.upper()} ({max_score:.2f})"
                color_map = {'red': (0, 0, 255), 'green': (0, 255, 0), 'blue': (255, 0, 0)}
                color_bgr = color_map.get(best_color, (255, 255, 255))

            # Draw on main image
            cv2.rectangle(debug_img, (u0, v0), (u1, v1), color_bgr, 2)
            cv2.putText(debug_img, label, (u0, v0 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_bgr, 2)
            
            # Show zoomed patch
            if patch_bgr.size > 0:
                patch_zoom = cv2.resize(patch_bgr, (150, 150), interpolation=cv2.INTER_NEAREST)
                cv2.imshow(f"Tag {tag.tag_id} Patch", patch_zoom)

        cv2.imshow("Debug: Color Association", debug_img)
        print("Check the pop-up windows. Press any key to proceed...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def _largest_color_centroid(self, hsv, target_color):
        """
        Find centroid of the largest connected component for a target cube color.

        Parameters
        ----------
        hsv : numpy.ndarray
            HSV image.
        target_color : str
            Target color label.

        Returns
        -------
        tuple or None
            `(cx, cy, area, mask)` if found, otherwise `None`.
        """
        mask = self._build_color_mask(hsv, target_color)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        largest = max(contours, key=cv2.contourArea)
        area = float(cv2.contourArea(largest))
        if area < 300.0:
            return None

        m = cv2.moments(largest)
        if abs(m["m00"]) < 1e-6:
            return None
        cx = int(m["m10"] / m["m00"])
        cy = int(m["m01"] / m["m00"])
        return cx, cy, area, mask

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
            if observation.shape[2] == 4:
                gray = cv2.cvtColor(observation, cv2.COLOR_BGRA2GRAY)
            else:
                gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        else:
            gray = observation
            hsv = None

        target_color = self._prompt_to_color(cube_prompt)
        all_colors = ('red', 'green', 'blue')
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

        # Trigger Visualizer
        #self.visualize_detections(hsv, bgr, tags)

        # Robust association strategy:
        # 1) Detect color centroids for all colors.
        # 2) Assign visible color centroids to unique tags by minimum total distance.
        # 3) If target color is not visible, infer it as the remaining unassigned tag.
        matched_tag = None
        color_centroids = {}
        if hsv is not None:
            for c in all_colors:
                blob = self._largest_color_centroid(hsv, c)
                if blob is not None:
                    cx, cy, area, _ = blob
                    color_centroids[c] = (float(cx), float(cy), float(area))
                    print(f"Detected {c} region centroid at ({cx}, {cy}), area={area:.1f}px")

        tag_indices = list(range(len(tags)))
        visible_colors = [c for c in all_colors if c in color_centroids]
        assigned_color_to_tag = {}

        if tags and visible_colors:
            best_cost = float('inf')
            best_assign = None
            # Brute-force over permutations (tiny set: <= 3).
            for perm in itertools.permutations(tag_indices, min(len(visible_colors), len(tag_indices))):
                cost = 0.0
                valid = True
                for ci, color in enumerate(visible_colors[:len(perm)]):
                    tx, ty = tags[perm[ci]].center
                    cx, cy, _ = color_centroids[color]
                    d2 = (float(tx) - cx) ** 2 + (float(ty) - cy) ** 2
                    cost += d2
                if valid and cost < best_cost:
                    best_cost = cost
                    best_assign = perm
            if best_assign is not None:
                for ci, color in enumerate(visible_colors[:len(best_assign)]):
                    assigned_color_to_tag[color] = best_assign[ci]

        # Primary path: target color directly assigned.
        if 1 < 0 and target_color in assigned_color_to_tag:
            print("PRIMARY PATH")
            matched_tag = tags[assigned_color_to_tag[target_color]]
        else:
            print("SECONDARY PATH")
            # Fallback 1: infer missing target as remaining tag when 3 tags are present.
            if 1 < 0 and len(tags) >= 3 and len(assigned_color_to_tag) >= 2:
                used = set(assigned_color_to_tag.values())
                remaining = [idx for idx in tag_indices if idx not in used]
                if remaining:
                    matched_tag = tags[remaining[0]]
                    print(f"Inferred {target_color} as remaining unassigned tag index={remaining[0]}.")
            # Fallback 2: local patch score if inference not possible.
            if matched_tag is None and hsv is not None:
                best_score = -1.0
                for tag in tags:
                    u, v = int(round(tag.center[0])), int(round(tag.center[1]))
                    if v < 0 or v >= hsv.shape[0] or u < 0 or u >= hsv.shape[1]:
                        continue
                    score = self._color_ratio_in_patch(hsv, u, v, target_color, patch_radius=44)
                    if score > best_score:
                        best_score = score
                        matched_tag = tag
                if best_score >= 0.0:
                    print(f"Fallback patch score for {target_color}: {best_score:.3f}")

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
        if t_cam_cube is not None and numpy.isfinite(t_cam_cube).all():
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
        arm.stop_lite6_gripper()
        time.sleep(0.5)
        arm.move_gohome(wait=True)
        time.sleep(0.5)
        arm.disconnect()

        # Close ZED Camera
        zed.close()

if __name__ == "__main__":
    main()
