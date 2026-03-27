import traceback
import cv2, numpy, time
import open3d as o3d
from xarm.wrapper import XArmAPI

from utils.vis_utils import draw_pose_axes
from utils.zed_camera import ZedCamera
from checkpoint0 import get_transform_camera_robot
from checkpoint1 import grasp_cube, place_cube, GRIPPER_LENGTH, robot_ip


def points_to_meters_open3d(xyz):
    """
    ZED often returns XYZ in millimeters; Open3D params below assume meters.
    Heuristic: if coordinates look like mm, scale to meters.
    """
    if xyz.size == 0:
        return xyz, 1.0
    max_abs = float(numpy.nanmax(numpy.abs(xyz)))
    scale = 0.001 if max_abs > 50.0 else 1.0
    return (xyz * scale).astype(numpy.float64), scale


def isolate_cube_cluster_open3d(pcd: o3d.geometry.PointCloud):
    """
    Segment a tabletop cube: voxel downsample, outliers, RANSAC plane, DBSCAN,
    then pick the most cube-like cluster (with a size fallback).
    """
    if len(pcd.points) < 150:
        return None, "too few points after NaN filter"

    pcd = pcd.voxel_down_sample(voxel_size=0.003)
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=25, std_ratio=2.0)

    if len(pcd.points) < 100:
        return None, "too few points after voxel/outlier"

    plane_model, inliers = pcd.segment_plane(
        distance_threshold=0.010,
        ransac_n=3,
        num_iterations=1500,
    )
    if len(inliers) > 0 and len(inliers) > 0.12 * len(pcd.points):
        pcd = pcd.select_by_index(inliers, invert=True)

    if len(pcd.points) < 80:
        return None, "nothing left after plane removal"

    labels = numpy.asarray(
        pcd.cluster_dbscan(eps=0.020, min_points=25, print_progress=False)
    )
    max_label = int(labels.max()) if labels.size else -1
    if max_label < 0:
        return None, "DBSCAN found no clusters"

    def score_cluster(cluster, idx):
        obb = cluster.get_oriented_bounding_box()
        ext = numpy.sort(numpy.asarray(obb.extent))
        if ext[2] < 1e-9:
            return -1.0, None
        max_dim = float(ext[2])
        min_dim = float(ext[0])
        compact = min_dim / max_dim if max_dim > 0 else 0.0
        size_ok = 0.008 <= max_dim <= 0.090
        if not size_ok:
            return float(idx.size) * 0.01, None
        qual = 1.0 if 0.25 < compact <= 1.0 else 0.3
        return float(idx.size) * compact * qual, cluster

    best_cluster = None
    best_score = -1.0
    fallback_largest = None
    fallback_n = 0

    for cid in range(max_label + 1):
        idx = numpy.where(labels == cid)[0]
        if idx.size < 30:
            continue
        cluster = pcd.select_by_index(idx)
        sc, chosen = score_cluster(cluster, idx)
        if idx.size > fallback_n:
            fallback_n = idx.size
            fallback_largest = cluster
        if sc > best_score and chosen is not None:
            best_score = sc
            best_cluster = chosen

    if best_cluster is not None:
        return best_cluster, "cube-like cluster"

    if fallback_largest is not None and len(fallback_largest.points) >= 40:
        return fallback_largest, "fallback: largest cluster (tune thresholds)"

    return None, "no cluster passed filters"


def get_transform_cube(observation, camera_intrinsic, camera_pose):
    """
    Estimate cube pose from image + point cloud (geometry; image for API only).

    Returns
    -------
    tuple
        On success: ``((t_robot_cube, t_cam_cube), status_message)``.
        On failure: ``(None, status_message)``.
    """
    image, point_cloud = observation
    if image is None or point_cloud is None:
        return None, "missing image or point_cloud"

    xyz = point_cloud[..., :3]
    valid_mask = numpy.isfinite(xyz).all(axis=-1)
    valid_points = xyz[valid_mask]

    if valid_points.shape[0] < 100:
        return None, f"too few finite points: {valid_points.shape[0]}"

    valid_points_m, _scale = points_to_meters_open3d(valid_points)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(valid_points_m)

    cube_pcd, seg_msg = isolate_cube_cluster_open3d(pcd)
    if cube_pcd is None or len(cube_pcd.points) < 30:
        return None, seg_msg

    obb = cube_pcd.get_oriented_bounding_box()
    center = numpy.asarray(obb.center)
    R_cam_cube = numpy.asarray(obb.R)

    t_cam_cube = numpy.eye(4)
    t_cam_cube[:3, :3] = R_cam_cube
    t_cam_cube[:3, 3] = center

    t_robot_cam = numpy.linalg.inv(camera_pose)
    t_robot_cube = t_robot_cam @ t_cam_cube

    return (t_robot_cube, t_cam_cube), seg_msg


def draw_status_overlay(image_bgra, lines, color=(0, 220, 0)):
    """Draw multiline status on BGRA image (mutates copy)."""
    out = image_bgra.copy()
    y = 36
    for line in lines:
        cv2.putText(
            out,
            line[:120],
            (16, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            cv2.LINE_AA,
        )
        y += 28
    return out


def main():
    zed = None
    arm = None
    try:
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

        cv_image = zed.image
        point_cloud = zed.point_cloud
        if cv_image is None:
            blank = numpy.zeros((720, 1280, 4), dtype=numpy.uint8)
            cv2.putText(blank, "ZED image is None", (40, 360), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
            cv2.namedWindow("Checkpoint 6", cv2.WINDOW_NORMAL)
            cv2.imshow("Checkpoint 6", blank)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            return

        lines = []
        t_cam_cube = None
        t_robot_cube = None

        t_cam_robot = get_transform_camera_robot(cv_image, camera_intrinsic)
        if t_cam_robot is None:
            lines.append("Calibration FAILED (checkpoint0 tags / PnP)")
            disp = draw_status_overlay(cv_image, lines, (0, 0, 255))
        else:
            try:
                result = get_transform_cube((cv_image, point_cloud), camera_intrinsic, t_cam_robot)
                if result is None or (isinstance(result, tuple) and result[0] is None):
                    if isinstance(result, tuple) and len(result) >= 2:
                        lines.append(f"Cube pose: {result[1]}")
                    else:
                        lines.append("Cube pose: unknown failure")
                    disp = draw_status_overlay(cv_image, lines, (0, 165, 255))
                else:
                    (t_robot_cube, t_cam_cube), seg_msg = result
                    lines.append(f"Segmentation: {seg_msg}")
                    lines.append("OK - press k to run grasp/place, any other key to quit")
                    disp = cv_image.copy()
                    if t_cam_cube is not None and numpy.isfinite(t_cam_cube).all():
                        draw_pose_axes(disp, camera_intrinsic, t_cam_cube)
                    disp = draw_status_overlay(disp, lines, (0, 220, 0))
            except Exception as exc:
                traceback.print_exc()
                lines.append(f"Exception: {exc!s}")
                disp = draw_status_overlay(cv_image, lines, (0, 0, 255))

        cv2.namedWindow("Checkpoint 6", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Checkpoint 6", 1280, 720)
        cv2.imshow("Checkpoint 6", disp)
        key = cv2.waitKey(0)

        if key == ord("k") and t_robot_cube is not None:
            cv2.destroyAllWindows()
            xyz = t_robot_cube[:3, 3]
            print(
                f"Cube in robot frame (m): x={xyz[0]:.3f}, y={xyz[1]:.3f}, z={xyz[2]:.3f}"
            )
            grasp_cube(arm, t_robot_cube)
            place_cube(arm, t_robot_cube)
            arm.stop_lite6_gripper()
        else:
            cv2.destroyAllWindows()

    except Exception:
        traceback.print_exc()
        try:
            if zed is not None and zed.image is not None:
                err_img = draw_status_overlay(
                    zed.image,
                    ["Fatal error - see console"],
                    (0, 0, 255),
                )
                cv2.namedWindow("Checkpoint 6", cv2.WINDOW_NORMAL)
                cv2.imshow("Checkpoint 6", err_img)
                cv2.waitKey(0)
        except Exception:
            pass
    finally:
        if arm is not None:
            try:
                arm.stop_lite6_gripper()
            except Exception:
                pass
            try:
                arm.move_gohome(wait=True)
            except Exception:
                pass
            time.sleep(0.5)
            try:
                arm.disconnect()
            except Exception:
                pass
        if zed is not None:
            try:
                zed.close()
            except Exception:
                pass
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass


if __name__ == "__main__":
    main()
