""" module that drops the background by comparing to a background model """

import cv2 as cv
import numpy as np
import os
from calibration import get_frame
from config import get_cam_dir

def frames(video_path):
    video = cv.VideoCapture(video_path)
    length = video.get(cv.CAP_PROP_FRAME_COUNT)
    for frame in range(int(length)):
        ret, frame = get_frame(video, frame)
        if ret:
            yield frame

def get_first_foreground_frame(video_path):
    for frame in frames(video_path):
        return frame

def create_background_model(cam_num):
    cam_dir = get_cam_dir(cam_num)
    video_path = os.path.abspath(os.path.join(cam_dir, "background.avi"))
    calib_dir = os.path.abspath(os.path.join(cam_dir, "calibration"))
    frames_list = []
    for frame in frames(video_path):
        # Set to HSV
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        frames_list.append(hsv)
    frames_list = np.array(frames_list)
    mean_background_hsv = np.mean(frames_list, axis=0)
    np.save(os.path.join(calib_dir, "background"), mean_background_hsv)
    return mean_background_hsv

def load_background_model(cam_num):
    cam_dir = get_cam_dir(cam_num)
    backgound_model_path = os.path.join(cam_dir, "calibration", "background.npy")
    return np.load(backgound_model_path, allow_pickle=True)

def substract_background(background_model, img, thresh_h, thresh_v, thresh_s, dilate = False, erode = False):

    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    hsv_diff = abs(hsv - background_model)

    # Create mask 

    # NOW IMPLEMENT HSV THRESHOLDS

    mask = np.zeros(hsv.shape)


    mask[:,:,0] = np.uint8(np.where(hsv_diff[:,:,0] > thresh_h, 1, 0))
    mask[:,:,1] = np.uint8(np.where(hsv_diff[:,:,1] > thresh_s, 1, 0))
    mask[:,:,2] = np.uint8(np.where(hsv_diff[:,:,2] > thresh_v, 1, 0))

    full_mask = np.uint8(mask.all(axis=2)) * 255

    # Dilate mask
    if erode:
        kernel = np.ones((1,2), np.uint8)
        full_mask = cv.erode(full_mask, kernel)
        kernel = np.ones((2,1), np.uint8)
        full_mask = cv.erode(full_mask, kernel)
    if dilate:
        kernel = np.ones((2,8), np.uint8)
        full_mask = cv.dilate(full_mask, kernel)
        kernel = np.ones((8,2), np.uint8)
        full_mask = cv.dilate(full_mask, kernel)
        # kernel = np.ones((2,2), np.uint8)
        # full_mask = cv.erode(full_mask, kernel)

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    background_removed = np.uint8(full_mask * gray)

    return background_removed, full_mask

# CHANGE THE CODE BELOW THIS

def gaussian_background_model_hsv(background_path, foreground_path):
    # Load the background and foreground videos
    bg_cap = cv.VideoCapture(background_path)
    fg_cap = cv.VideoCapture(foreground_path)
    # Initialize variables
    alpha = 0.05  # Controls the learning rate
    background_model = None
    while True:
        # Read the next frames of the background and foreground videos
        bg_ret, bg_frame = bg_cap.read()
        fg_ret, fg_frame = fg_cap.read()
        if not bg_ret or not fg_ret:
            break 
        # Convert the frames to HSV color space
        bg_hsv = cv.cvtColor(bg_frame, cv.COLOR_BGR2HSV)
        fg_hsv = cv.cvtColor(fg_frame, cv.COLOR_BGR2HSV)
        # Initialize the background model if it doesn't exist
        if background_model is None:
            background_model = bg_hsv.astype(float)
            continue
        # Update the background model using a Gaussian distribution
        background_model = alpha * bg_hsv.astype(float) + (1 - alpha) * background_model
        # Calculate the difference between the foreground and background
        diff = cv.absdiff(fg_hsv, background_model.astype(np.uint8))
        # Threshold the difference image based on standard deviation
        mean, stddev = cv.meanStdDev(diff)
        threshold = mean + 3*stddev  # Change the factor to adjust the threshold
        # Threshold each channel of the diff image separately
        mask = np.zeros(diff.shape[:2], dtype=np.uint8)
        for i in range(diff.shape[2]):
            channel_mask = np.zeros(diff.shape[:2], dtype=np.uint8)
            channel_mask[diff[:,:,i] > threshold[i]] = 255
            mask |= channel_mask
        # Apply dilation and erosion to the mask
        kernel = np.ones((3, 3), dtype=np.uint8)
        mask = cv.dilate(mask, kernel)
        mask = cv.erode(mask, kernel)
        # Apply opening to the mask
        mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
        # Display the current frame, the background model, and the mask
        cv.imshow('Foreground', fg_hsv)
        cv.imshow('Background Model', background_model.astype(np.uint8))
        cv.imshow('Mask', mask)
        k = cv.waitKey(30) & 0xff
        if k == 27:
            break
    # Release the video and close all windows
    bg_cap.release()
    fg_cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    H = 2
    S = 8
    V = 13
    for cam in [1, 2, 3, 4]:
        load = load_background_model(cam)
        vid = cv.VideoCapture(os.path.abspath(os.path.join(get_cam_dir(cam), "video.avi")))
        ret, img = get_frame(vid, 0)
        background_removed, mask = substract_background(load, img, 2, 8, 13, dilate=True, erode=True)
        cv.imshow("cleaned", background_removed)
        cv.waitKey(0)
    # gaussian_background_model_hsv(os.path.abspath(os.path.join(get_cam_dir(3), "background.avi")), os.path.abspath(os.path.join(get_cam_dir(3), "video.avi")))
