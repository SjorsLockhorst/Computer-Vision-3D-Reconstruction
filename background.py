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

    print(frames_list.shape)

    mean_background_hsv = np.mean(frames_list, axis=0)

    return mean_background_hsv

def substract_background(mean_background_hsv, img):
    
    new_img = abs(img - mean_background_hsv)

    cv.imshow('new_img', new_img)
    cv.waitKey(0)

if __name__ == "__main__":
    mean_background_hsv = create_background_model(os.path.abspath(os.path.join(get_cam_dir(1), "background.avi")))
    print(mean_background_hsv.shape)
    img = get_foreground_image(os.path.abspath(os.path.join(get_cam_dir(1), "video.avi")))
    substract_background(mean_background_hsv, img)