"""
Module that does calibration.
Inspired by https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html
"""
import os
from typing import List

import cv2 as cv
import numpy as np

from util import interpolate_chessboard, get_pictures
from config import STRIDE_LEN, CHESS_DIMS, CALIB_DIR


def create_dataset_1() -> List[str]:
    """Use 20 detected, 5 undetected images."""
    return get_pictures(20, 5)


def create_dataset_2() -> List[str]:
    """Use 10 detected images."""
    return get_pictures(10, 0)


def create_dataset_3() -> List[str]:
    """Use 5 images from dataset 2."""
    return get_pictures(5, 0)


def calibrate_with(experiment, **kwargs):
    """Calibrate with a specific set of detected and undetected edges."""
    experiment_map = {
        1: create_dataset_1,
        2: create_dataset_2,
        3: create_dataset_3
    }
    dataset_creator = experiment_map[experiment]
    paths = dataset_creator()
    ret, mtx, dist, rvecs, tvecs = calibrate(paths, CHESS_DIMS, STRIDE_LEN, **kwargs)
    if ret:
        np.save(os.path.join(CALIB_DIR, f"mtx_{experiment}"), mtx)
        np.save(os.path.join(CALIB_DIR, f"dist_{experiment}"), dist)
        np.save(os.path.join(CALIB_DIR, f"rvecs_{experiment}"), rvecs)
        np.save(os.path.join(CALIB_DIR, f"tvecs_{experiment}"), tvecs)


def calibrate(image_paths, pattern_size, checker_size_in_mm, improve_corner_points=True, improve_interp_points = True):
    """Calibrate from images at certain paths."""
    imgpoints = []
    objectpoints = []

    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0],
                           0:pattern_size[1]].T.reshape(-1, 2)
    objp = objp * checker_size_in_mm

    for path in image_paths:
        img = cv.imread(path)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        ret, corners = cv.findChessboardCorners(gray, pattern_size, None)
        if ret:
            imgpoints.append(corners)
        else:
            ret, corners = interpolate_chessboard(
                img,
                pattern_size,
                improve_corner_points=improve_corner_points,
                improve_interp_points=improve_interp_points
            )
            imgpoints.append(corners)
        objectpoints.append(objp)

        cv.drawChessboardCorners(img, pattern_size, corners, ret)
        cv.imshow('img', img)
        cv.waitKey(500)
        cv.destroyAllWindows()

    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(
        objectpoints, imgpoints, gray.shape[::-1], cv.CALIB_USE_INTRINSIC_GUESS, None)
    return ret, mtx, dist, rvecs, tvecs


if __name__ == "__main__":
    # calibrate_with(1, improve_interp_points=True)
    calibrate_with(2)
    calibrate_with(3)
