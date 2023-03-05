""" module that drops the background by comparing to a background model """

import os

import cv2 as cv
import numpy as np
from tqdm import tqdm
from numba import njit

import itertools

from calibration import get_frame
from config import get_cam_dir


def create_background_model(cam_num):
    cam_dir = get_cam_dir(cam_num)
    video_path = os.path.abspath(os.path.join(cam_dir, "background.avi"))

    # Load in background video for camera
    video = cv.VideoCapture(video_path)
    # Check if video opened successfully
    if not video.isOpened():
        print("Error opening video file")

    frames_list = []

    # Transform each frame to hsv and append to list
    while True:
        ret, frame = video.read()
        if not ret:
            break
        # Set to HSV
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        frames_list.append(hsv)

    frames_list = np.array(frames_list)

    gmm = cv.createBackgroundSubtractorMOG2()

    # train on background
    for hsv_frame in frames_list:
        gmm.apply(hsv_frame)

    return gmm

if __name__ == "__main__":
    # THIS IS OPENCV BACKGROUND MODEL BRANCH

    for cam in [1, 2, 3, 4]:
        gmm = create_background_model(cam)
        vid = cv.VideoCapture(os.path.abspath(
            os.path.join(get_cam_dir(cam), "video.avi")))
        img = get_frame(vid, 0)
        hsv_frame = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        mask = gmm.apply(hsv_frame)
        std_dev = np.std(mask)
        if std_dev > 90:
            mask = cv.threshold(mask, 128, 255, cv.THRESH_BINARY)[1]
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        background_removed = gray * mask
        cv.imshow("cleaned", background_removed)
        cv.waitKey(0)
