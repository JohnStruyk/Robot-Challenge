import cv2, time
from xarm.wrapper import XArmAPI

from checkpoint6 import run_pure_vision_perception
from utils.zed_camera import ZedCamera
from checkpoint1 import grasp_cube, GRIPPER_LENGTH, robot_ip
from checkpoint2 import place_in_basket, BASKET_POSE


def main():
    zed = ZedCamera()
    camera_intrinsic = zed.camera_intrinsic

    arm = XArmAPI(robot_ip)
    arm.connect()
    arm.motion_enable(enable=True)
    arm.set_tcp_offset([0, 0, GRIPPER_LENGTH, 0, 0, 0])
    arm.set_mode(0)
    arm.set_state(0)
    arm.move_gohome(wait=True)
    time.sleep(0.5)

    try:
        cv_image = zed.image
        point_cloud = zed.point_cloud
        t_robot_cube, _t_cam_cube, disp, _status = run_pure_vision_perception(
            cv_image, point_cloud, camera_intrinsic
        )

        cv2.namedWindow("Verifying Cube Pose", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Verifying Cube Pose", 1280, 720)
        cv2.imshow("Verifying Cube Pose", disp)
        key = cv2.waitKey(0)

        if key == ord("k") and t_robot_cube is not None:
            cv2.destroyAllWindows()
            xyz = t_robot_cube[:3, 3]
            print(
                f"Cube in robot frame (m): x={xyz[0]:.3f}, y={xyz[1]:.3f}, z={xyz[2]:.3f}"
            )
            grasp_cube(arm, t_robot_cube)
            place_in_basket(arm, BASKET_POSE, vaccum_gripper=False)
            arm.stop_lite6_gripper()
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
