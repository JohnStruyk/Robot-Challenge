import numpy, time, threading
import pyzed.sl as sl

# Depth mode notes (Stereolabs; depends on camera + SDK version):
# - NEURAL: default on many recent SDKs — good balance, strong in low texture / varied lighting.
# - NEURAL_PLUS: highest AI detail / accuracy, heavier GPU & lower FPS — worth trying for small
#   cubes (~22–30 mm) if your GPU can sustain your target FPS at HD2K.
# - NEURAL_LIGHT: faster, less detail — OK for coarse avoidance, weaker for metrology.
# - Legacy stereo (ZED 2 / older SDK): QUALITY or ULTRA often give crisp edges on textured mats;
#   try ULTRA if NEURAL feels “soft” on cube boundaries.
# This wrapper does not force a mode unless you pass ``depth_mode=`` — otherwise the SDK /
# ZED Explorer default applies (what you’re seeing as NEURAL).


class ZedCamera:

    def __init__(
        self,
        resolution=sl.RESOLUTION.HD2K,
        fps=15,
        exposure=15,
        *,
        depth_mode=None,
        depth_minimum_distance_m=None,
        depth_maximum_distance_m=None,
        confidence_threshold=None,
        texture_confidence_threshold=None,
    ):
        """
        Parameters
        ----------
        depth_mode
            e.g. ``sl.DEPTH_MODE.NEURAL_PLUS`` for finer detail (more GPU), or ``sl.DEPTH_MODE.NEURAL``
            to match typical defaults. ``None`` = do not set (SDK / Explorer default).
        depth_minimum_distance_m, depth_maximum_distance_m
            Clip depth search to the workspace (meters). Narrowing max distance often cleans
            tabletop clouds (e.g. ``depth_maximum_distance_m=1.8``). ``None`` = SDK default.
        confidence_threshold
            Per-pixel depth confidence 0–100; **higher** = stricter (fewer outliers, sparser cloud).
            ``None`` = SDK default.
        texture_confidence_threshold
            Texture-based filtering; ``None`` = SDK default.
        """
        # Initialize ZED Camera
        self._zed = sl.Camera()
        init_params = sl.InitParameters()
        init_params.enable_image_validity_check = True
        init_params.camera_resolution = resolution
        init_params.camera_fps = fps

        if depth_mode is not None:
            try:
                init_params.depth_mode = depth_mode
            except Exception:
                print("Warning: could not set depth_mode; using SDK default.")
        if depth_minimum_distance_m is not None:
            try:
                init_params.depth_minimum_distance = float(depth_minimum_distance_m)
            except Exception:
                pass
        if depth_maximum_distance_m is not None:
            try:
                init_params.depth_maximum_distance = float(depth_maximum_distance_m)
            except Exception:
                pass

        self._runtime_parameters = sl.RuntimeParameters()
        if confidence_threshold is not None:
            try:
                self._runtime_parameters.confidence_threshold = int(confidence_threshold)
            except Exception:
                pass
        if texture_confidence_threshold is not None:
            try:
                self._runtime_parameters.texture_confidence_threshold = int(
                    texture_confidence_threshold
                )
            except Exception:
                pass

        # Open ZED Camera
        err = self._zed.open(init_params)
        if err > sl.ERROR_CODE.SUCCESS:
            print("Camera Open : "+repr(err)+". Exit program.")
            exit(-1)

        # Warmup ZED Camera
        self._zed.set_camera_settings(sl.VIDEO_SETTINGS.AEC_AGC, 1)
        self._zed.set_camera_settings(sl.VIDEO_SETTINGS.WHITEBALANCE_AUTO, 1)
        for _ in range(int(ZED_WARMUP_GRABS)):
            self._zed.grab(self._runtime_parameters)

        # Setup Explosure for Better Image Quality
        # self._zed.set_camera_settings(sl.VIDEO_SETTINGS.EXPOSURE, exposure)

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