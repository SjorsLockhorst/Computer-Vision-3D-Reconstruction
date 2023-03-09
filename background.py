""" module that drops the background by comparing to a background model """

import os

import cv2 as cv
import numpy as np
from tqdm import tqdm

import itertools

from calibration import get_frame
from config import conf


def create_background_model(cam_num):
    """Create a background model for a given camera and save it to disk."""
    background_path = conf.background_vid_path(cam_num)
    print(background_path)
    calib_path = conf.get_calib_dir(cam_num)

    # Load in background video for camera
    video = cv.VideoCapture(background_path)
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

    save_path = os.path.join(calib_path, f"background_{cam_num}")
    print(save_path)
    np.save(save_path, gaussian)
    return gaussian


def load_background_model(cam_num):
    """Load background model from disk for a certain camera."""
    return conf.load_bg_model(cam_num)


def threshold_difference(diff, thresholds):
    """Apply a threshold to a difference in HSV values"""
    thresh_h, thresh_s, thresh_v = thresholds

    mask = np.zeros(diff.shape)
    mask[:, :, 0] = np.uint8(np.where(diff[:, :, 0] > thresh_h, 1, 0))
    mask[:, :, 1] = np.uint8(np.where(diff[:, :, 1] > thresh_s, 1, 0))
    mask[:, :, 2] = np.uint8(np.where(diff[:, :, 2] > thresh_v, 1, 0))

    full_mask = np.uint8(mask.all(axis=2))

    return full_mask


def substract_background(
    background_model,
    img,
    thresholds,
    erode_kernel=(3, 3),
    dilate_kernel=(2,4),
    gaussian_kernel=(3,3),
    n_biggest=1,
):
    """Substract background from an image given a background model and parameters."""

    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    mean_background_model = background_model[:, :, 0, :]
    std_background_model = background_model[:, :, 1, :]

    std_diff = abs(hsv - mean_background_model) / std_background_model

    h, s, v = thresholds
    full_mask = threshold_difference(std_diff, thresholds)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    #
    full_mask = cv.GaussianBlur(full_mask, gaussian_kernel, 0)

    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (4, 4))
    full_mask = cv.dilate(full_mask, kernel, iterations=1)
    #
    new_mask = np.zeros(gray.shape)
    contours, _ = cv.findContours(
        full_mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    # biggest = sorted(contours, key=cv.contourArea, reverse=True)[:n_biggest]
    biggest = [contour for contour in contours if cv.contourArea(contour) > 5000]

    # cv.drawContours(new_mask, biggest, -1, 1, -1)
    full_mask = cv.fillPoly(new_mask, biggest, 1)
    full_mask = new_mask

    background_removed = np.uint8(full_mask * gray)

    return background_removed, full_mask


def opt_substract_background(std_diff, img, thresholds, erode_kernel=(2, 2), dilate_kernel=(4, 1), gaussian_kernel=(5, 5)):
    """Alternative helper used when background diff is already available."""
    h, s, v = thresholds
    full_mask = threshold_difference(std_diff, thresholds)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    kernel = np.ones(dilate_kernel, np.uint8)
    full_mask = cv.dilate(full_mask, kernel)

    kernel = np.ones(erode_kernel, np.uint8)
    full_mask = cv.erode(full_mask, kernel)

    full_mask = cv.GaussianBlur(full_mask, gaussian_kernel, 0)

    new_mask = np.zeros(gray.shape)
    contours, _ = cv.findContours(
        full_mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    biggest = sorted(contours, key=cv.contourArea, reverse=True)[0]

    cv.drawContours(new_mask, [biggest], 0, 1, -1)
    full_mask = new_mask

    background_removed = np.uint8(full_mask * gray)

    return background_removed, full_mask


def find_optimal_background_substraction(
    frame_id,
    h_range,
    s_range,
    v_range,
    erode_combs,
    dilate_combs,
    gaussian_combs
):
    """
    Loop over parameters and find background substraction that is closest to a manual 
    cutout.
    """

    models = []
    images = []
    gold_labels = []
    for cam in conf.CAMERAS:
        create_background_model(cam)
        bg_model = load_background_model(cam)
        vid_path = conf.main_vid_path(cam)
        vid = cv.VideoCapture(vid_path)
        img = get_frame(vid, frame_id)
        images.append(img)
        cutout_path = conf.cutout_img_path(cam)
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

    combinations = list(itertools.product(
        h_range, s_range, v_range, erode_combs, dilate_combs, gaussian_combs))
    best_erode = None
    best_dilate = None
    best_guassian = None
    for h, s, v, erode_kern, dilate_kern, gaussian_kern in tqdm(combinations):

        error = 0
        masks = []
        for cam_num in conf.CAMERAS:
            std_diff = models[cam_num - 1]
            cutout = gold_labels[cam_num - 1]
            img = images[cam_num - 1]

            full_mask = opt_substract_background(
                std_diff, img, (h, s, v), erode_kernel=erode_kern, dilate_kernel=dilate_kern, gaussian_kernel=gaussian_kern)[1]
            masks.append(full_mask)

            xor = np.logical_xor(full_mask, cutout).astype("uint8")
            error = xor.sum()

        if error < best_error:
            best_masks = masks
            best_error = error
            best_h = h
            best_s = s
            best_v = v
            best_erode = erode_kern
            best_dilate = dilate_kern
            best_guassian = gaussian_kern

    for img, best_mask in zip(images, best_masks):
        cv.imshow("Best mask", img * best_mask[:, :, None])
        cv.waitKey(0)

    return best_masks, (best_h, best_s, best_v), (best_erode, best_dilate, best_guassian)


def show_background_substraction(cam):
    """Shows final background substraction."""
    bg_model = load_background_model(cam)
    vid_path = conf.main_vid_path(cam)
    vid = cv.VideoCapture(vid_path)
    length = vid.get(cv.CAP_PROP_FRAME_COUNT)
    for frame_id in range(int(length)):
        img = get_frame(vid, frame_id)
        background_removed = substract_background(
            bg_model, img, (5, 18, 18), n_biggest=4)[0]
        cv.imshow("test", background_removed)
        cv.waitKey(1)


if __name__ == "__main__":
    # h_range = range(0, 25)
    # s_range = range(0, 25)
    # v_range = range(0, 85)
    #
    # erode_combs = itertools.product(range(1, 4), repeat=2)
    # dilate_combs = itertools.product(range(1, 8), repeat=2)
    # gaussian_combs = [(3, 3), (5, 5), (7, 7)]
    #
    # erode_combs = itertools.product(range(1, 4), repeat=2)
    # dilate_combs = itertools.product(range(1, 8), repeat=2)
    # gaussian_combs = [(3, 3)]
    #
    # background_masks, best_hsv, best_kernels = find_optimal_background_substraction(
    #     1, h_range, s_range, v_range, erode_combs, dilate_combs, gaussian_combs)
    # print(best_hsv)
    # print(best_kernels)
    for cam in conf.CAMERAS:
        create_background_model(cam)
    show_background_substraction(1)
