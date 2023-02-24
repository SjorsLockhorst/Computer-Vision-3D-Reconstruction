"""
Module that does calibration.
Inspired by https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html
"""
import os

import cv2 as cv
import numpy as np

from config import get_cam_dir, CHESS_DIMS, STRIDE_LEN
from util import interpolate_chessboard, sample_files, interpolate_chessboard_with_perspective
from online import load_internal_calibrations, draw


def get_frame(video, frame_id):
    """Get a specific frame from a video."""
    video.set(cv.CAP_PROP_POS_FRAMES, frame_id)
    return video.read()


def frames(video, n_samples):
    """Load a video and iterate over it's frames."""
    length = video.get(cv.CAP_PROP_FRAME_COUNT)
    sampled_frames = np.linspace(
        0, length, num=n_samples, endpoint=False, dtype=int)
    for frame in sampled_frames:
        ret, frame = get_frame(video, frame)
        if ret:
            yield frame


def detect_and_save_frames(video_path, pattern_size, n_samples, manual_interpolate, output_dirname="frames", show_live=False) -> None:
    """Samples frame from video, save to disk."""
    video_dir = os.path.dirname(video_path)
    video = cv.VideoCapture((os.path.join(video_dir, "intrinsics.avi")))
    frame_dir = os.path.join(video_dir, output_dirname)
    if not os.path.exists(frame_dir):
        os.mkdir(frame_dir)

    correct = 0
    for idx, frame in enumerate(frames(video, n_samples)):
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
        if show_live:
            cv.drawChessboardCorners(frame, pattern_size, corners, ret)
            cv.imshow("Frame", frame)
            cv.waitKey(5)


def sample_video_and_calibrate(num, n_samples, n_calibration, pattern_size, manual_interpolate, frame_dirname="frames", **kwargs):
    cam_dir = get_cam_dir(num)
    vid_path = os.path.join(cam_dir, "intrinsics.avi")
    detect_and_save_frames(vid_path, pattern_size, n_samples,
                           manual_interpolate, output_dirname=frame_dirname, **kwargs)
    mtx, dist = calibrate_camera(
        num, n_calibration, pattern_size, frame_dirname=frame_dirname, **kwargs)
    return mtx, dist


def calibrate_camera(num, n_images, pattern_size, frame_dirname="frames", **kwargs):
    """Calibrate a specific camera."""
    file_paths = os.path.join("data", f"cam{num}", frame_dirname)
    cam_dir = get_cam_dir(num)
    files = sample_files(file_paths, n_images)
    _, mtx, dist, _, _ = calibrate(files, pattern_size, STRIDE_LEN, **kwargs)
    calib_path = os.path.join(cam_dir, "calibration")
    if not os.path.exists(calib_path):
        os.mkdir(calib_path)
    np.save(os.path.join(calib_path, "mtx"), mtx)
    np.save(os.path.join(calib_path, "dist"), dist)
    return mtx, dist


def calibrate(image_paths, pattern_size, checker_size, show_live=False):
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

    return cv.calibrateCamera(
        objectpoints, imgpoints, gray.shape[::-1], cv.CALIB_USE_INTRINSIC_GUESS, None)


def calibrate_all(n_calibration, pattern_size, manual_interpolate, show_live, frame_dirname="frames", n_sample=None):
    CAMERAS = [1, 2, 3, 4]
    for camera_num in CAMERAS:
        if n_sample is not None:
            sample_video_and_calibrate(camera_num, n_sample, n_calibration,
                                       pattern_size, manual_interpolate, show_live=show_live)
        else:
            calibrate_camera(camera_num, n_calibration, pattern_size, frame_dirname=frame_dirname, show_live=show_live)


def get_extrinsics(img, mtx, dist, pattern_size, stride_len, corners):

    objp = np.zeros((pattern_size[0]*pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0],
                           0:pattern_size[1]].T.reshape(-1, 2) * stride_len

    # Find the rotation and translation vectors.
    ret, rvec, tvec = cv.solvePnP(objp, corners, mtx, dist, useExtrinsicGuess=False)
    if ret:

        return rvec, tvec

def save_extrinsics(num, rvec, tvec):
    cam_dir = get_cam_dir(num)
    calib_path = os.path.join(cam_dir, "calibration")
    if not os.path.exists(calib_path):
        os.mkdir(calib_path)
    np.save(os.path.join(calib_path, "rvec"), rvec)
    np.save(os.path.join(calib_path, "tvec"), tvec)


def draw_axes(img, mtx, dist, rvec, tvec, corners, stride_len):
    axes = np.float32([[1, 0, 0], [0, 1, 0], [0, 0, -1]]
                      ).reshape(-1, 3) * stride_len * 4
    axes_points, jac2 = cv.projectPoints(axes, rvec, tvec, mtx, dist)

    corners = np.int32(corners).reshape(-1, 2)
    imgpts = np.int32(axes_points).reshape(-1, 2)

    corner = tuple(corners[0].ravel())

    img = cv.line(img, corner, tuple(imgpts[0].ravel()), (0, 0, 255), 2)
    img = cv.line(img, corner, tuple(imgpts[1].ravel()), (0, 255, 0), 2)
    img = cv.line(img, corner, tuple(imgpts[2].ravel()), (255, 0, 0), 2)
    return img





if __name__ == "__main__":
    # calibrate_all(25, CHESS_DIMS, False, True)
    # # mtx, dist = res[1:3]
    for cam_num in [1, 2, 3, 4]:
        video = cv.VideoCapture(os.path.join(
            "data", f"cam{cam_num}", "checkerboard.avi"))
        img = [frame for frame in frames(video, 1)][0]
        res, corners = interpolate_chessboard_with_perspective(img, CHESS_DIMS, window_size=(2, 2))

        mtx, dist = load_internal_calibrations(cam_num)
        rvec, tvec = get_extrinsics(img, mtx, dist, CHESS_DIMS, STRIDE_LEN, corners)
        img_w_axes = draw_axes(img, mtx, dist, rvec, tvec, corners, STRIDE_LEN)
        cv.imshow("chessboard with axes", img_w_axes)
        cv.waitKey(0)
