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


def create_gaussian(video_path):
    frames_list = []

    for frame in frames(video_path):
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        frames_list.append(hsv)


    means = np.mean(frames_list)
    stds = np.std(frames_list)

    return means, stds

if __name__ == "__main__":
    means, stds = create_gaussian(os.path.abspath(os.path.join(get_cam_dir(1), "background.avi")))
    print(means, stds)