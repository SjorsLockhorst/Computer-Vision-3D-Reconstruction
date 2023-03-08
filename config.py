"""Configuration"""
import os

import numpy as np

ACTIVE_CONF = "voxel"


class VoxelConfig():
    DATA_DIR = "data"
    CAM_DIR = "cam{}"
    CAMERAS = [1, 2, 3, 4]
    NUM_CAMS = len(CAMERAS)
    CAM_RES = (488, 644)  # Resolution of all cameras

    STRIDE_LEN = 115
    CHESS_DIMS = (8, 6)

# Intrinsic calibration tunable parameters
    CALIB_INTR = False
    N_SAMPLE = None  # If none, use existing frames, if int, sample this many frames
    N_CALIB = 25  # Amount of images to use for intrinsic calibration
    MANUAL_INTERPOLATE_INTR = False  # Whether to use manual interpolate to fix images
    SHOW_LIVE = True  # Whether to show progress live

# Extrinsic calibration tunable parameters
    CALIB_EXTR = True
    PLOT_AXES = True

    def get_calib_dir(self, num):
        return self.get_some_cam_dir(num, "calibration")

    def get_some_cam_dir(self, num, end_path):
        return os.path.join(self.get_cam_dir(num), end_path)

    def get_cam_dir(self, num):
        if num > self.NUM_CAMS:
            raise Exception("Max number of cams is 4")
        return os.path.abspath(os.path.join(self.DATA_DIR, self.CAM_DIR.format(num)))

    def get_frame_dir(self, num):
        return self.get_some_cam_dir(num, "frames")

    def load_intr_calib(self, num):
        calib_dir = self.get_calib_dir(num)
        mtx = np.load(os.path.join(calib_dir, "mtx.npy"), allow_pickle=True)
        dist = np.load(os.path.join(calib_dir, "dist.npy"), allow_pickle=True)
        return mtx, dist

    def load_extr_calib(self, num):
        base_path = self.get_some_cam_dir(num, "calibration")
        rvec = np.load(os.path.join(base_path, "rvec.npy"), allow_pickle=True)
        tvec = np.load(os.path.join(base_path, "tvec.npy"), allow_pickle=True)
        return rvec, tvec

    def extr_calib_vid_path(self, num):
        return os.path.join(self.get_cam_dir(num), "checkerboard.avi")

    def background_vid_path(self, num):
        return os.path.join(self.get_cam_dir(num), "background.avi")

    def main_vid_path(self, num):
        return os.path.join(self.get_cam_dir(num), "video.avi")


class ClusteringConf():
    DATA_DIR = "clustering_data"
    CAM_DIR = "cam{}"
    CAMERAS = [1, 2, 3, 4]
    NUM_CAMS = len(CAMERAS)
    CAM_RES = (488, 644)  # Resolution of all cameras

    STRIDE_LEN = 115
    CHESS_DIMS = (8, 6)

# Intrinsic calibration tunable parameters
    CALIB_INTR = False
    N_SAMPLE = None  # If none, use existing frames, if int, sample this many frames
    N_CALIB = 25  # Amount of images to use for intrinsic calibration
    MANUAL_INTERPOLATE_INTR = False  # Whether to use manual interpolate to fix images
    SHOW_LIVE = False  # Whether to show progress live

# Extrinsic calibration tunable parameters
    CALIB_EXTR = True
    PLOT_AXES = True

    def get_calib_dir(self, num):
        return os.path.abspath("calibration")

    def load_intr_calib(self, num):
        calib_dir = self.get_calib_dir(num)
        mtx = np.load(os.path.join(
            calib_dir, f"mtx_{num}.npy"), allow_pickle=True)
        dist = np.load(os.path.join(
            calib_dir, f"dist_{num}.npy"), allow_pickle=True)
        return mtx, dist

    def load_extr_calib(self, num):
        calib_path = self.get_calib_dir(num)
        rvec = np.load(os.path.join(
            calib_path, f"rvec_{num}.npy"), allow_pickle=True)
        tvec = np.load(os.path.join(
            calib_path, f"tvec_{num}.npy"), allow_pickle=True)
        return rvec, tvec

    def _get_vid(self, num, dirname):
        path = os.path.join(self.DATA_DIR, dirname)
        file = os.listdir(path)[num - 1]
        return os.path.join(path, file)


    def extr_calib_vid_path(self, num):
        DIRNAME = "extrinsics"
        return self._get_vid(num, DIRNAME)

    def background_vid_path(self, num):
        DIRNAME = "background"
        return self._get_vid(num, DIRNAME)

    def main_vid_path(self, num):
        DIRNAME = "video"
        return self._get_vid(num, DIRNAME)


if ACTIVE_CONF == "voxel":
    conf = VoxelConfig()
elif ACTIVE_CONF == "clustering":
    conf = ClusteringConf()
