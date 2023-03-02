"""Configuration"""
import os

DATA_DIR = "data"
CAM_DIR = "cam{}"
NUM_CAMS = 4

STRIDE_LEN = 115
CHESS_DIMS = (8, 6)

def get_cam_dir(num):
    if num > NUM_CAMS:
        raise Exception("Max number of cams is 4")
    return os.path.abspath(os.path.join(DATA_DIR, CAM_DIR.format(num)))
