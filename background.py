""" module that drops the background by comparing to a background model """

import os

import cv2 as cv
import numpy as np

from calibration import frames, get_frame
from config import get_cam_dir


def create_background_model(cam_num):
    cam_dir = get_cam_dir(cam_num)
    video_path = os.path.abspath(os.path.join(cam_dir, "background.avi"))
    calib_dir = os.path.abspath(os.path.join(cam_dir, "calibration"))
    
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

    mean_background_hsv = np.mean(frames_list, axis=0)
    std_background_hsv = np.std(frames_list, axis=0)
    
    gaussian = np.stack((mean_background_hsv, std_background_hsv), axis=2)

    np.save(os.path.join(calib_dir, "background"), gaussian)
    return gaussian


def load_background_model(cam_num):
    cam_dir = get_cam_dir(cam_num)
    backgound_model_path = os.path.join(
        cam_dir, "calibration", "background.npy")
    return np.load(backgound_model_path, allow_pickle=True)


# Create version of this that uses std as theshold instead of hard limit
def substract_background(
    background_model,
    img,
    thresh_h,
    thresh_v,
    thresh_s,
    dilate=False,
    erode=False
):

    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    
    mean_background_model = background_model[:, :, 0, :]
    std_background_model = background_model[:, :, 1, :]

    std = abs(hsv - mean_background_model) / std_background_model

    mask = np.zeros(hsv.shape)

    mask[:, :, 0] = np.uint8(np.where(std[:, :, 0] > thresh_h, 1, 0))
    mask[:, :, 1] = np.uint8(np.where(std[:, :, 1] > thresh_s, 1, 0))
    mask[:, :, 2] = np.uint8(np.where(std[:, :, 2] > thresh_v, 1, 0))

    full_mask = np.uint8(mask.all(axis=2))

    # Dilate mask
    if erode:
        kernel = np.ones((1, 2), np.uint8)
        full_mask = cv.erode(full_mask, kernel)
        kernel = np.ones((2, 1), np.uint8)
        full_mask = cv.erode(full_mask, kernel)
    if dilate:
        kernel = np.ones((2, 8), np.uint8)
        full_mask = cv.dilate(full_mask, kernel)
        kernel = np.ones((8, 2), np.uint8)
        full_mask = cv.dilate(full_mask, kernel)

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    background_removed = np.uint8(full_mask * gray)

    return background_removed, full_mask


if __name__ == "__main__":
    # THIS IS IMPROVEMENT BRANCH
    H = 50 # 2
    S = 50 # 8
    V = 50 # 13
    for cam in [1, 2, 3, 4]:
        create_background_model(cam)
        load = load_background_model(cam)
        vid = cv.VideoCapture(os.path.abspath(
            os.path.join(get_cam_dir(cam), "video.avi")))
        img = get_frame(vid, 0)
        background_removed, mask = substract_background(
            load, img, H, S, V, dilate=True, erode=True)
        cv.imshow("cleaned", background_removed)
        cv.waitKey(0)
