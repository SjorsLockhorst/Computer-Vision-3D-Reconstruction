"""
Module that does calibration.
Inspired by https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html
"""
import os

import cv2 as cv
import numpy as np

from config import CHESS_DIMS
from util import interpolate_chessboard, sample_files
from online import draw

def get_frame(video, frame_id):
    video.set(cv.CAP_PROP_POS_FRAMES, frame_id)
    return video.read()
    
def frames(video_path, n_samples):
    video = cv.VideoCapture(video_path)
    length = video.get(cv.CAP_PROP_FRAME_COUNT)
    sampled_frames = np.linspace(
        0, length, num=n_samples, endpoint=False, dtype=int)
    for frame in sampled_frames:
        ret, frame = get_frame(video, frame)
        if ret:
            yield frame



def detect_and_save_frames(video_path, pattern_size, n_samples, manual_interpolate, output_dirname="frames") -> None:
    """Samples frame from video, save to disk."""
    video_dir = os.path.dirname(video_path)
    frame_dir = os.path.join(video_dir, output_dirname)
    if not os.path.exists(frame_dir):
        os.mkdir(frame_dir)

    correct = 0
    for idx, frame in enumerate(frames(video_path, n_samples)):
        output_frame = frame.copy()
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        ret, corners = cv.findChessboardCorners(gray, pattern_size, None)
        if ret:
            filename = os.path.join(frame_dir, f"frame_{idx}.jpg")
            cv.imwrite(filename, output_frame)
            correct += 1
        else:
            if manual_interpolate:
                ret, corners = interpolate_chessboard(frame, pattern_size)
        cv.drawChessboardCorners(frame, pattern_size, corners, ret)
        cv.imshow("Frame", frame)
        cv.waitKey(5)
    print(correct)


def calibrate_camera(num, n_images, pattern_size, **kwargs):
    # Open correct camera name folder
    # Call calibrate with image paths, pattern_size=pattern_size, checker_size = 1
    # Save calibration
    file_paths = os.path.join("data", f"cam{num}", "frames")

    files = sample_files(file_paths, n_images)
    return calibrate(files, pattern_size, 1, **kwargs)


def calibrate(image_paths, pattern_size, checker_size, show_live = False):
    """Calibrate from images at certain paths."""
    imgpoints = []
    objectpoints = []

    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0],
                           0:pattern_size[1]].T.reshape(-1, 2)
    objp = objp * checker_size

    for path in image_paths:
        img = cv.imread(path)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        ret, corners = cv.findChessboardCorners(gray, pattern_size, None)
        if ret:
            imgpoints.append(corners)
            objectpoints.append(objp)

        if show_live:
            cv.drawChessboardCorners(img, pattern_size, corners, ret)
            cv.imshow('img', img)
            cv.waitKey(10)

    cv.destroyAllWindows()

    return cv.calibrateCameraExtended(
        objectpoints, imgpoints, gray.shape[::-1], cv.CALIB_USE_INTRINSIC_GUESS, None)


if __name__ == "__main__":
    res = calibrate_camera(1, 30, CHESS_DIMS, show_live=True)
    mtx, dist = res[1:3]
    chess_frame = [frame for frame in frames(os.path.join("data", "cam1", "checkerboard.avi"), 1)][0]
    _, corners = interpolate_chessboard(chess_frame, CHESS_DIMS, improve_interp_points=False, improve_corner_points=False)
    img, error = draw(chess_frame, mtx, dist, CHESS_DIMS, 1, corners=corners, include_error=False)
    cv.drawChessboardCorners(chess_frame, CHESS_DIMS, corners, True)
    # chess_frame = [frame for frame in frames(os.path.join("data", "cam1", "intrinsics.avi"), 200)]
    # for frame in chess_frame:
    #     ret, corners = cv.findChessboardCorners(frame, CHESS_DIMS)
    #     if ret:
    #         img, error = draw(frame, mtx, dist, CHESS_DIMS, 1, corners=corners, include_error=False)
    #         cv.imshow('img', img)
    #         cv.waitKey(0)
