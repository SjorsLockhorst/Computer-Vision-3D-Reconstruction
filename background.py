""" module that drops the background by comparing to a background model """

import cv2 as cv
import numpy as np
import os
from offline import get_frame
from config import get_cam_dir

def frames(video_path):
    video = cv.VideoCapture(video_path)
    length = video.get(cv.CAP_PROP_FRAME_COUNT)
    for frame in range(int(length)):
        ret, frame = get_frame(video, frame)
        if ret:
            yield frame

def get_foreground_image(video_path):
    frames_list = []

    for frame in frames(video_path):
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        frames_list.append(hsv)

    frames_list = np.array(frames_list)

    img = frames_list[0]

    cv.imshow('img', img)
    cv.waitKey(0)

    return frames_list[0]

def create_background_model(video_path):
    frames_list = []

    for frame in frames(video_path):
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        frames_list.append(hsv)

    frames_list = np.array(frames_list)

    mean_background_hsv = np.mean(frames_list, axis=0)

    return mean_background_hsv

def substract_background(mean_background_hsv, img):
    
    new_img = abs(img - mean_background_hsv)

    final_img = np.uint8(np.where(new_img > 20, 255, 0))

    kernel = np.ones((5,5), np.uint8) 

    img_dilation = cv.dilate(img, kernel, iterations=1)
    
    # mask = np.where(new_img > 20, 1, 0)

    # final_img = np.uint8(mask * img)

    cv.imshow('final_img', final_img)
    cv.waitKey(0)

    return final_img

if __name__ == "__main__":
    mean_background_hsv = create_background_model(os.path.abspath(os.path.join(get_cam_dir(1), "background.avi")))
    img = get_foreground_image(os.path.abspath(os.path.join(get_cam_dir(1), "video.avi")))
    new_img = substract_background(mean_background_hsv, img)