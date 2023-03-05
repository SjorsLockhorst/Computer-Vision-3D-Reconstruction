""" module that drops the background by comparing to a background model """

import os

import cv2 as cv
import numpy as np
from tqdm import tqdm
from numba import njit

import itertools

from calibration import get_frame
from config import get_cam_dir, CAMERAS


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


def threshold_difference(diff, thresholds):

    thresh_h, thresh_s, thresh_v = thresholds

    mask = np.zeros(diff.shape)
    mask[:, :, 0] = np.uint8(np.where(diff[:, :, 0] > thresh_h, 1, 0))
    mask[:, :, 1] = np.uint8(np.where(diff[:, :, 1] > thresh_s, 1, 0))
    mask[:, :, 2] = np.uint8(np.where(diff[:, :, 2] > thresh_v, 1, 0))

    full_mask = np.uint8(mask.all(axis=2))

    return full_mask


def substract_background(background_model, img, thresholds):
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    mean_background_model = background_model[:, :, 0, :]
    std_background_model = background_model[:, :, 1, :]

    std_diff = abs(hsv - mean_background_model) / std_background_model
    h, s, v = thresholds
    full_mask = threshold_difference(std_diff, thresholds)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # kernel = np.ones((1, 1), np.uint8)
    # full_mask = cv.erode(full_mask, kernel)
    kernel = np.ones((4, 1), np.uint8)
    full_mask = cv.dilate(full_mask, kernel)
    full_mask = cv.GaussianBlur(full_mask, (5, 5), 0)
    # kernel = np.ones((4, 1), np.uint8)
    # full_mask = cv.dilate(full_mask, kernel)
    # kernel = np.ones((1, 2), np.uint8)
    # full_mask = cv.erode(full_mask, kernel)
    # kernel = np.ones((4, 1), np.uint8)
    # full_mask = cv.dilate(full_mask, kernel)
    background_removed = np.uint8(full_mask * gray)

    return background_removed, full_mask


@njit(fastmath=True)
def calculate_error(full_mask, cutout, img):
    return np.absolute(((full_mask * img - cutout * img)**2).sum())


def find_optimal_background_substraction(
    frame_id,
    h_range,
    s_range,
    v_range,
):

    models = []
    images = []
    gold_labels = []
    for cam in CAMERAS:
        create_background_model(cam)
        bg_model = load_background_model(cam)
        vid = cv.VideoCapture(os.path.abspath(
            os.path.join(get_cam_dir(cam), "video.avi")))
        img = get_frame(vid, frame_id)
        images.append(img)
        cam_dir = get_cam_dir(cam)
        cutout_path = os.path.abspath(os.path.join(cam_dir, f"image{cam}.png"))
        cutout = cv.imread(cutout_path)

        hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

        mean_background_model = bg_model[:, :, 0, :]
        std_background_model = bg_model[:, :, 1, :]

        std_diff = abs(hsv - mean_background_model) / std_background_model
        models.append(std_diff)

        cutout = cv.cvtColor(cutout, cv.COLOR_BGR2GRAY)
        cutout = np.where(cutout == 255, 1, 0)
        gold_labels.append(cutout)

    print("Created all models, start optimizing")
    best_masks = []
    best_error = float('inf')
    best_h = 0
    best_s = 0
    best_v = 0

    combinations = list(itertools.product(h_range, s_range, v_range))
    for h, s, v in tqdm(combinations):

        error = 0
        masks = []
        for cam_num in CAMERAS:
            std_diff = models[cam_num - 1]
            cutout = gold_labels[cam_num - 1]

            full_mask = threshold_difference(
                std_diff, (h, s, v))
            masks.append(full_mask)

            # if erode:
            #     kernel = np.ones((1, 2), np.uint8)
            #     full_mask = cv.erode(full_mask, kernel)
            #     kernel = np.ones((2, 1), np.uint8)
            #     full_mask = cv.erode(full_mask, kernel)
            # if dilate:
            #     kernel = np.ones((2, 8), np.uint8)
            #     full_mask = cv.dilate(full_mask, kernel)
            #     kernel = np.ones((8, 2), np.uint8)
            #     full_mask = cv.dilate(full_mask, kernel)

            # error = calculate_error(full_mask[:, :, None], cutout[:, :, None], hsv)
            xor = np.logical_xor(full_mask, cutout).astype("uint8")
            contours, hierarchy = cv.findContours(
                xor, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
            error = len(contours) * xor.sum()

        if error < best_error:
            best_masks = masks
            best_error = error
            best_h = h
            best_s = s
            best_v = v

    for img, best_mask in zip(images, best_masks):
        cv.imshow("Best mask", img * best_mask[:, :, None])
        cv.waitKey(0)

    return best_masks, (best_h, best_s, best_v)


if __name__ == "__main__":
    # THIS IS IMPROVEMENT BRANCH
    # h_range = range(8, 25)
    # s_range = range(8, 25)
    # v_range = range(60, 85)
    # background_masks, best_hsv = find_optimal_background_substraction(
    #     1, h_range, s_range, v_range)
    # background_removed,_ = substract_background(load, img, (100, 100, 200))
    # print(best_hsv)
    for cam in CAMERAS:
        bg_model = load_background_model(cam)
        vid = cv.VideoCapture(os.path.abspath(
            os.path.join(get_cam_dir(cam), "video.avi")))
        img = get_frame(vid, 1)
        background_removed = substract_background(bg_model, img, (18, 10, 61))[0]
        cv.imshow("test", background_removed)
        cv.waitKey(0)

