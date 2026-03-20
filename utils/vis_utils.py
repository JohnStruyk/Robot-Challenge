import cv2, numpy

def draw_pose_axes(image, camera_intrinsic, pose, size=0.1):

    rvec, _ = cv2.Rodrigues(pose[:3,:3])
    tvec = pose[:3, 3]

    # origin and 3 unit vector of the frame
    frame_points = numpy.array([[0, 0, 0],
                                [1, 0, 0],
                                [0, 1, 0],
                                [0, 0, 1]]).reshape(-1,3) * size

    ipoints, _ = cv2.projectPoints(frame_points, rvec, tvec, camera_intrinsic, None)
    ipoints = numpy.round(ipoints).astype(int)

    origin = tuple(ipoints[0].ravel())
    unit_x = tuple(ipoints[1].ravel())
    unit_y = tuple(ipoints[2].ravel())
    unit_z = tuple(ipoints[3].ravel())

    cv2.line(image, origin, unit_x, (0,0,255), 2)
    cv2.line(image, origin, unit_y, (0,255,0), 2)
    cv2.line(image, origin, unit_z, (255,0,0), 2)