"""Configuration"""
import os

DATA_DIR = "data"
CAM_DIR = "cam{}"
CAMERAS = [1, 2, 3, 4]
NUM_CAMS = len(CAMERAS)
CAM_RES = (488,644)  # Resolution of all cameras

STRIDE_LEN = 115
CHESS_DIMS = (8, 6)

# Intrinsic calibration tunable parameters
CALIB_INTR = True
N_SAMPLE = None  # If none, use existing frames, if int, sample this many frames
N_CALIB = 25  # Amount of images to use for intrinsic calibration
MANUAL_INTERPOLATE_INTR = False  # Whether to use manual interpolate to fix images
SHOW_LIVE = True  # Whether to show progress live

# Extrinsic calibration tunable parameters
CALIB_EXTR = True
PLOT_AXES = True


def get_cam_dir(num):
    if num > NUM_CAMS:
        raise Exception("Max number of cams is 4")
    return os.path.abspath(os.path.join(DATA_DIR, CAM_DIR.format(num)))
