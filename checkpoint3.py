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

        mask = numpy.zeros(patch.shape[:2], dtype=bool)
        for low, high in self.color_ranges[target_color]:
            m = (
                (patch[:, :, 0] >= low[0]) & (patch[:, :, 0] <= high[0]) &
                (patch[:, :, 1] >= low[1]) & (patch[:, :, 1] <= high[1]) &
                (patch[:, :, 2] >= low[2]) & (patch[:, :, 2] <= high[2])
            )
            mask |= m
        return float(mask.mean())

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
        mask = numpy.zeros(hsv.shape[:2], dtype=numpy.uint8)
        for low, high in self.color_ranges[target_color]:
            partial = cv2.inRange(hsv, low, high)
            mask = cv2.bitwise_or(mask, partial)

        kernel = numpy.ones((5, 5), dtype=numpy.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: return None
        largest = max(contours, key=cv2.contourArea)
        area = float(cv2.contourArea(largest))
        if area < 300.0: return None
        m = cv2.moments(largest)
        if abs(m["m00"]) < 1e-6: return None
        return int(m["m10"] / m["m00"]), int(m["m01"] / m["m00"]), area, mask

    def get_transforms(self, observation, cube_prompt):
        if self.t_cam_robot is None:
            raise RuntimeError('Camera pose not initialized.')

        # Process Image
        bgr = observation[:, :, :3]
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

        target_color = self._prompt_to_color(cube_prompt)
        all_colors = ('red', 'green', 'blue')
        
        tags = self.detector.detect(
            gray, estimate_tag_pose=True,
            camera_params=(float(self.camera_intrinsic[0, 0]), float(self.camera_intrinsic[1, 1]),
                           float(self.camera_intrinsic[0, 2]), float(self.camera_intrinsic[1, 2])),
            tag_size=CUBE_TAG_SIZE,
        )

        # Trigger Visualizer
        self.visualize_detections(hsv, bgr, tags)

        # Association Strategy (Simplified for Debugging)
        color_centroids = {}
        for c in all_colors:
            blob = self._largest_color_centroid(hsv, c)
            if blob:
                cx, cy, area, _ = blob
                color_centroids[c] = (float(cx), float(cy), float(area))

        matched_tag = None
        tag_indices = list(range(len(tags)))
        visible_colors = [c for c in all_colors if c in color_centroids]
        assigned_color_to_tag = {}

        if tags and visible_colors:
            best_cost = float('inf')
            best_assign = None
            for perm in itertools.permutations(tag_indices, min(len(visible_colors), len(tag_indices))):
                cost = 0.0
                for ci, color in enumerate(visible_colors[:len(perm)]):
                    tx, ty = tags[perm[ci]].center
                    cx, cy, _ = color_centroids[color]
                    cost += (float(tx) - cx) ** 2 + (float(ty) - cy) ** 2
                if cost < best_cost:
                    best_cost = cost
                    best_assign = perm
            if best_assign:
                for ci, color in enumerate(visible_colors[:len(best_assign)]):
                    assigned_color_to_tag[color] = best_assign[ci]

        if target_color in assigned_color_to_tag:
            matched_tag = tags[assigned_color_to_tag[target_color]]
        
        if matched_tag is None:
            print(f"No tag matched {cube_prompt}.")
            return None

        t_cam_cube = numpy.eye(4)
        t_cam_cube[:3, :3] = matched_tag.pose_R
        t_cam_cube[:3, 3] = matched_tag.pose_t.flatten()
        t_robot_cam = numpy.linalg.inv(self.t_cam_robot)
        return t_robot_cam @ t_cam_cube, t_cam_cube

def main():
    zed = ZedCamera()
    cube_pose_detector = CubePoseDetector(zed.camera_intrinsic)
    arm = XArmAPI(robot_ip)
    arm.connect()
    arm.motion_enable(enable=True)
    arm.set_tcp_offset([0, 0, GRIPPER_LENGTH, 0, 0, 0])
    arm.set_mode(0)
    arm.set_state(0)
    arm.move_gohome(wait=True)

    try:
        cv_image = zed.image
        t_cam_robot = get_transform_camera_robot(cv_image, zed.camera_intrinsic)
        if t_cam_robot is None: return
        
        cube_pose_detector.set_camera_pose(t_cam_robot)
        result = cube_pose_detector.get_transforms(cv_image, cube_prompt)
        
        if result:
            t_robot_cube, t_cam_cube = result
            draw_pose_axes(cv_image, zed.camera_intrinsic, t_cam_cube)
            cv2.imshow('Final Cube Pose Verification', cv_image)
            if cv2.waitKey(0) == ord('k'):
                grasp_cube(arm, t_robot_cube)
                place_cube(arm, t_robot_cube)
    finally:
        arm.move_gohome(wait=True)
        arm.disconnect()
        zed.close()

if __name__ == "__main__":
    main()