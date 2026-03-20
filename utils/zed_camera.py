import numpy, time, threading
import pyzed.sl as sl

class ZedCamera:

    def __init__(self, resolution=sl.RESOLUTION.HD2K, fps=15, exposure=20):

        # Initialize ZED Camera
        self._zed = sl.Camera()
        init_params = sl.InitParameters()
        init_params.enable_image_validity_check = True
        init_params.camera_resolution = resolution
        init_params.camera_fps = fps
        self._runtime_parameters = sl.RuntimeParameters()

        # Open ZED Camera
        err = self._zed.open(init_params)
        if err > sl.ERROR_CODE.SUCCESS:
            print("Camera Open : "+repr(err)+". Exit program.")
            exit(-1)

        # Warmup ZED Camera
        self._zed.set_camera_settings(sl.VIDEO_SETTINGS.AEC_AGC, 1)
        self._zed.set_camera_settings(sl.VIDEO_SETTINGS.WHITEBALANCE_AUTO, 1)
        for _ in range(50):
            self._zed.grab(sl.RuntimeParameters())

        # Setup Explosure for Better Image Quality
        self._zed.set_camera_settings(sl.VIDEO_SETTINGS.EXPOSURE, exposure)

        # Get Camera Intrinsic
        camera_info = self._zed.get_camera_information()
        left_camera_param = camera_info.camera_configuration.calibration_parameters.left_cam
        self._camera_intrinsic = numpy.eye(3)
        self._camera_intrinsic[0, 0] = left_camera_param.fx
        self._camera_intrinsic[1, 1] = left_camera_param.fy
        self._camera_intrinsic[0, 2] = left_camera_param.cx
        self._camera_intrinsic[1, 2] = left_camera_param.cy

        # Setup Variable for the Thread
        self._image_mat = sl.Mat()
        self._measure_XYZ = sl.Mat()
        self._image = None
        self._point_cloud = None

        # Setup Background Thread
        self._running = True
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._update, daemon=True)
        self._thread.start()
        while self._image is None or self._point_cloud is None:
            time.sleep(0.1)

    def _update(self):
        while self._running:

            if self._zed.grab(self._runtime_parameters) == sl.ERROR_CODE.SUCCESS:
                self._zed.retrieve_image(self._image_mat, sl.VIEW.LEFT)
                self._zed.retrieve_measure(self._measure_XYZ, sl.MEASURE.XYZ)

                with self._lock:
                    self._image = self._image_mat.get_data().copy()
                    self._point_cloud = self._measure_XYZ.get_data().copy()
            else:
                time.sleep(0.01)

    def close(self):
        self._running = False
        if hasattr(self, '_thread'):
            self._thread.join()
        self._zed.close()

    @property
    def image(self):
        with self._lock:
            return self._image.copy() if self._image is not None else None
    
    @property
    def point_cloud(self):
        with self._lock:
            return self._point_cloud.copy() if self._point_cloud is not None else None

    @property
    def camera_intrinsic(self):
        return self._camera_intrinsic