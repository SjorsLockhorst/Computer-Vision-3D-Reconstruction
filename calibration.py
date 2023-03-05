"""
Module that does calibration.
Inspired by https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html
"""
import os

import cv2 as cv
import numpy as np

import config

from util import (
    interpolate_chessboard,
    interpolate_chessboard_with_perspective,
    sample_files,
)


def get_frame(video, frame_id):
    """Get a specific frame from a video."""
    video.set(cv.CAP_PROP_POS_FRAMES, frame_id)
    ret, frame = video.read()
    if ret:
        return frame


def frames(video, n_frames):
    """Uniformly sample n frames from a video."""
    length = video.get(cv.CAP_PROP_FRAME_COUNT)
    sampled_frames = np.linspace(
        0, length, num=n_frames, endpoint=False, dtype=int)
    for frame in sampled_frames:
        ret, frame = get_frame(video, frame)
        if ret:
            yield frame


def detect_and_save_frames(
    video_path,
    pattern_size,
    n_samples,
    manual_interpolate,
    output_dirname="frames",
    show_live=False
):
    """Samples frame from video, only save to disk if chessboard edges are found."""
    video_dir = os.path.dirname(video_path)
    video = cv.VideoCapture((os.path.join(video_dir, "intrinsics.avi")))
    frame_dir = os.path.join(video_dir, output_dirname)
    if not os.path.exists(frame_dir):  # Create directory for frames if not exists
        os.mkdir(frame_dir)

    correct = 0

    # Uniformly sample n frames from a video and iterate over them
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
                correct += 1
        if show_live:
            cv.drawChessboardCorners(frame, pattern_size, corners, ret)
            cv.imshow("Frame", frame)
            cv.waitKey(5)


def sample_video_and_calibrate_intr(
        num,
        n_samples,
        n_calibration,
        pattern_size,
        manual_interpolate,
        frame_dirname="frames",
        **kwargs
):
    """
    Draw n frames uniformly from intrinsics video, calibrate intrinsics of camera 
    with subset using those samples.
    """
    cam_dir = config.get_cam_dir(num)
    vid_path = os.path.join(cam_dir, "intrinsics.avi")
    detect_and_save_frames(vid_path, pattern_size, n_samples,
                           manual_interpolate, output_dirname=frame_dirname, **kwargs)
    mtx, dist = calibrate_camera_intr(
        num, n_calibration, pattern_size, frame_dirname=frame_dirname, **kwargs)
    return mtx, dist


def calibrate_camera_intr(
        num,
        n_images,
        pattern_size,
        frame_dirname="frames",
        **kwargs
):
    """Calibrate intrinsics for a specific camera."""
    cam_dir = config.get_cam_dir(num)
    frame_directory = os.path.join(cam_dir, frame_dirname)
    files = sample_files(frame_directory, n_images)

    # Get only camera matrix and distortion from intrinsic calibration
    _, mtx, dist, _, _ = calibrate_intrinsics(
        files, pattern_size, config.STRIDE_LEN, **kwargs)

    # Save intrinsic calibration in folder calibration
    calib_path = os.path.join(cam_dir, "calibration")
    if not os.path.exists(calib_path):
        os.mkdir(calib_path)
    np.save(os.path.join(calib_path, "mtx"), mtx)
    np.save(os.path.join(calib_path, "dist"), dist)
    return mtx, dist


def calibrate_intrinsics(image_paths, pattern_size, checker_size, show_live=False):
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


def calibrate_and_save_cam_calib_intr(
        cam_num,
        n_calibration,
        pattern_size,
        manual_interpolate,
        show_live,
        frame_dirname="frames",
        n_sample=None
):
    """
    Calibrate all camera intrinsics, sample n frames from intrinisc video if 
    specified, otherwise assume sampling has been performed already.
    """
    if n_sample is not None:
        sample_video_and_calibrate_intr(
            cam_num,
            n_sample,
            n_calibration,
            pattern_size,
            manual_interpolate,
            show_live=show_live
        )
    else:
        calibrate_camera_intr(
            cam_num,
            n_calibration,
            pattern_size,
            frame_dirname=frame_dirname,
            show_live=show_live
        )


def get_chessboard_obj_points(pattern_size, stride_len):
    """Returns object points of chessboard relative to pattern_size and stride_len."""
    objp = np.zeros((pattern_size[0]*pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0],
                           0:pattern_size[1]].T.reshape(-1, 2) * stride_len
    return objp


def calibrate_extr(img, mtx, dist, pattern_size, stride_len, corners):
    """Calibrate extrinic paramters based on internal calibration parameters."""
    objp = get_chessboard_obj_points(pattern_size, stride_len)
    # Find the rotation and translation vectors.
    ret, rvec, tvec = cv.solvePnP(
        objp, corners, mtx, dist, useExtrinsicGuess=False)
    if ret:
        return rvec, tvec


def calibrate_camera_extr(num, pattern_size, stride_len):
    """
    Calibrate externals for a given camera number, using perspective transform and 
    manual interpolation.
    """
    WINDOW_SIZE = (2, 2)  # Set window size for subpix function

    mtx, dist = load_intr_calibration(num)
    img = get_extr_calibration_img(num)
    res, corners = interpolate_chessboard_with_perspective(
        img, pattern_size, window_size=WINDOW_SIZE)
    rvec, tvec = calibrate_extr(
        img, mtx, dist, pattern_size, stride_len, corners)
    return rvec, tvec


def save_extrinsics(cam_num, rvec, tvec):
    """Write extrinsic parameters to disk."""
    cam_dir = config.get_cam_dir(cam_num)
    calib_path = os.path.join(cam_dir, "calibration")
    if not os.path.exists(calib_path):
        os.mkdir(calib_path)
    np.save(os.path.join(calib_path, "rvec"), rvec)
    np.save(os.path.join(calib_path, "tvec"), tvec)


def load_all_calibration(cam_num):
    """Load both internal and external calibrations for a given camera"""
    cam_dir = config.get_cam_dir(cam_num)
    calib_path = os.path.join(cam_dir, "calibration")
    mtx = np.load(os.path.join(calib_path, "mtx.npy"), allow_pickle=True)
    dist = np.load(os.path.join(calib_path, "dist.npy"), allow_pickle=True)
    rvec = np.load(os.path.join(calib_path, "rvec.npy"), allow_pickle=True)
    tvec = np.load(os.path.join(calib_path, "tvec.npy"), allow_pickle=True)
    return mtx, dist, rvec, tvec

def load_intr_calibration(num):
    """Loads internal calibrations based on camera number."""
    load_path = os.path.join(config.get_cam_dir(num), "calibration")

    mtx = np.load(os.path.join(
        load_path, "mtx.npy"), allow_pickle=True)
    dist = np.load(os.path.join(
        load_path, "dist.npy"), allow_pickle=True)
    
    return mtx, dist

def load_extr_calibration(num):
    """Loads external calibrations based on camera number."""
    load_path = os.path.join(config.get_cam_dir(num), "calibration")
    rvec = np.load(os.path.join(
        load_path, "rvec.npy"), allow_pickle=True)
    tvec = np.load(os.path.join(
        load_path, "tvec.npy"), allow_pickle=True)
    return rvec, tvec


def calibrate_and_save_extr(num, pattern_size, stride_len):
    """First calibrate, then save extrinsics."""
    rvec, tvec = calibrate_camera_extr(
        num, pattern_size, stride_len)
    save_extrinsics(num, rvec, tvec)
    return rvec, tvec


def get_extr_calibration_img(cam_num):
    """Gives frame number 1 from checkerboard.avi for a given camera number."""
    FRAME_TO_SELECT = 1
    video = cv.VideoCapture(os.path.join(
        "data", f"cam{cam_num}", "checkerboard.avi"))
    return get_frame(video, FRAME_TO_SELECT)


def calibrate_intr_and_extr(
        num,
        n_images,
        pattern_size,
        stride_len,
        frame_dirname="frames",
        n_samples=None
):
    """Calibrate all intr and extr in one go."""
    WINDOW_SIZE = (2, 2)  # Set window size for subpix function

    if n_samples is not None:
        cam_dir = config.get_cam_dir(num)
        vid_path = os.path.abspath(os.path.join(cam_dir, "intrinsics.avi"))
        detect_and_save_frames(vid_path, pattern_size,
                               n_samples, False, output_dirname=frame_dirname)

    mtx, dist = calibrate_camera_intr(
        num, n_images, pattern_size, frame_dirname=frame_dirname)
    img = get_extr_calibration_img(num)
    res, corners = interpolate_chessboard_with_perspective(
        img, pattern_size, window_size=WINDOW_SIZE)
    rvec, tvec = calibrate_extr(
        img, mtx, dist, pattern_size, stride_len, corners)
    save_extrinsics(num, rvec, tvec)
    return mtx, dist, rvec, tvec


def draw_axes(img, stride_len, mtx, dist, rvec, tvec, corners):
    """Draw axes on a chessboard, given calibration and corners."""
    drawn_img = img.copy()
    axes = np.float32([[1, 0, 0], [0, 1, 0], [0, 0, -1]]
                      ).reshape(-1, 3) * stride_len * 4
    axes_points, jac2 = cv.projectPoints(axes, rvec, tvec, mtx, dist)

    corners = np.int32(corners).reshape(-1, 2)
    imgpts = np.int32(axes_points).reshape(-1, 2)

    corner = tuple(corners[0].ravel())

    img = cv.line(drawn_img, corner, tuple(imgpts[0].ravel()), (0, 0, 255), 2)
    img = cv.line(img, corner, tuple(imgpts[1].ravel()), (0, 255, 0), 2)
    img = cv.line(img, corner, tuple(imgpts[2].ravel()), (255, 0, 0), 2)
    return img


def draw_axes_from_zero(img, stride_len, mtx, dist, rvec, tvec, origin):
    """Draw axis on chessboard given calibration and origin"""
    drawn_img = img.copy()
    axes = np.float32([[1, 0, 0], [0, 1, 0], [0, 0, -1]]
                      ).reshape(-1, 3) * stride_len * 4
    axes_points, jac2 = cv.projectPoints(axes, rvec, tvec, mtx, dist)

    imgpts = np.int32(axes_points).reshape(-1, 2)

    img = cv.line(drawn_img, origin, tuple(imgpts[0].ravel()), (0, 0, 255), 2)
    img = cv.line(img, origin, tuple(imgpts[1].ravel()), (0, 255, 0), 2)
    img = cv.line(img, origin, tuple(imgpts[2].ravel()), (255, 0, 0), 2)
    return img


def calibrate():
    """Run calibration with configuration from config"""
    for cam_num in config.CAMERAS:
        if config.CALIB_INTR:
            calibrate_and_save_cam_calib_intr(
                cam_num,
                config.N_CALIB,
                config.CHESS_DIMS,
                config.MANUAL_INTERPOLATE_INTR,
                config.SHOW_LIVE,
                n_sample=config.N_SAMPLE
            )

        if config.CALIB_EXTR:
            mtx, dist = load_intr_calibration(cam_num)
            rvec, tvec = calibrate_camera_extr(
                cam_num, config.CHESS_DIMS, config.STRIDE_LEN)
            save_extrinsics(cam_num, rvec, tvec)
            if config.PLOT_AXES:
                img = get_extr_calibration_img(cam_num)
                objpoints = get_chessboard_obj_points(
                    config.CHESS_DIMS, config.STRIDE_LEN)
                corners, _ = cv.projectPoints(objpoints, rvec, tvec, mtx, dist)
                drawn = draw_axes(img, config.STRIDE_LEN, mtx,
                                  dist, rvec, tvec, corners)
                cv.imshow("With axes", drawn)
                cv.waitKey(0)


if __name__ == "__main__":
    calibrate()
