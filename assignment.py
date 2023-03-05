import os

import cv2 as cv
import glm
import numpy as np

from background import load_background_model, substract_background
from calibration import (
    draw_axes_from_zero,
    get_frame,
    load_extr_calibration,
    load_intr_calibration,
)
from config import STRIDE_LEN, get_cam_dir

block_size = 1.0
scale = 4


def generate_grid(width, depth):
    # Generates the floor grid locations
    # You don't need to edit this function
    data = []
    for x in range(width):
        for z in range(depth):
            data.append([x*block_size - width/2, -
                        block_size, z*block_size - depth/2])
    return data


def create_lookup_table(cam_num):
    mtx, dist = load_intr_calibration(cam_num)
    rvec, tvec = load_extr_calibration(cam_num)
    size_x = 32
    size_y = 32
    size_z = 64
    scale_factor = STRIDE_LEN / scale 
    voxel_block = np.zeros((size_x * size_y * size_z, 3))

    counter = 0
    for x in range(size_x):
        for y in range(size_y):
            for z in range(size_z):
                voxel_block[counter] = [x, y, -z]
                counter += 1

    scaled_voxels = voxel_block * scale_factor

    coords = cv.projectPoints(scaled_voxels, rvec, tvec, mtx, dist)[0]
    lookup_table = {}
    for voxel_coord, img_coord in zip(scaled_voxels, np.squeeze(coords)):
        scaled_back_voxel = voxel_coord / scale_factor
        lookup_table[tuple(scaled_back_voxel.astype(int))] = img_coord.astype(int)
    return lookup_table


def draw_axes_on_image(cam_num, frame_id):
    point = (0,0,0)
    mtx, dist = load_intr_calibration(cam_num)
    rvec, tvec = load_extr_calibration(cam_num)
    coords = cv.projectPoints(point, rvec, tvec, mtx, dist)[0].astype(int)
    vid = cv.VideoCapture(os.path.abspath(os.path.join(get_cam_dir(cam_num), "video.avi")))
    img = get_frame(vid, frame_id)
    x_img, y_img = coords[0][0]
    return draw_axes_from_zero(img, STRIDE_LEN, mtx, dist, rvec, tvec, (x_img, y_img))


def plot_projection(cam_num, point):
    rescaled_point = tuple(map(lambda x: x * STRIDE_LEN / scale, point))
    mtx, dist = load_intr_calibration(cam_num)
    rvec, tvec = load_extr_calibration(cam_num)
    coords = cv.projectPoints(rescaled_point, rvec, tvec, mtx, dist)[0].astype(int)
    vid = cv.VideoCapture(os.path.abspath(os.path.join(get_cam_dir(cam_num), "video.avi")))
    img = get_frame(vid, 2)
    x_img, y_img = coords[0][0]
    img = draw_axes_on_image(cam_num, 2)
    img[y_img: y_img+ 10,x_img: x_img+ 10 ] = 255
    cv.imshow("test", img)
    cv.waitKey(0)


# TODO: Find intersection while creating voxels
def set_voxel_positions(width, height, depth):
    cams = [1, 2, 3, 4]
    H = 2
    S = 8
    V = 13
    voxels_in_mask = []
    for cam in cams:
        lookup_table = create_lookup_table(cam)

        bg_model = load_background_model(cam)
        vid = cv.VideoCapture(os.path.abspath(os.path.join(get_cam_dir(cam), "video.avi")))
        img = get_frame(vid, 2)
        mask = substract_background(bg_model, img, H, S, V, dilate=True, erode=True)[1]
        is_in_mask = in_mask(lookup_table.values(), mask)
        voxels_in_mask.append(is_in_mask)

    voxels_to_draw = find_intersection_masks(*voxels_in_mask)
    selected = np.array(list(lookup_table.keys()))[voxels_to_draw]
    return [(x, -1 * z, y) for x, y, z in list(selected)]


def get_cam_positions():
    # Generates dummy camera locations at the 4 corners of the room

    cams = [1, 2, 3, 4]
    cam_positions = []
    for cam in cams:
        rvec, tvec = load_extr_calibration(cam)
        tvec /= STRIDE_LEN / scale
        # Calculate rotation matrix and camera position
        rot_mat = cv.Rodrigues(rvec)[0]
        cam_pos = -np.matrix(rot_mat).T * np.matrix(tvec)
        cam_pos *= block_size
        cam_positions.append(cam_pos)

    return [[cam_pos[0], -1 * cam_pos[2], cam_pos[1]] for cam_pos in cam_positions]


def get_cam_rotation_matrices():

    cams = [1, 2, 3, 4]
    cam_angles = []
    for i in cams:
        rvec, tvec = load_extr_calibration(i)
        rot_m = cv.Rodrigues(rvec)[0]
        with_zeros = np.pad(rot_m, ((0, 1), (0, 1)))
        with_zeros[with_zeros.shape[0] - 1, with_zeros.shape[1] - 1] = 1

        rotation = glm.mat4(with_zeros)
        for idx, val in enumerate(np.squeeze(tvec)):
            rotation[idx, 3] = val

        rotation = glm.rotate(rotation, glm.radians(90), (0, 0, 1))
        cam_angles.append(rotation)

    return cam_angles

def in_mask(coordinates, mask):
    coordinates_in_mask = []
    for x,y in coordinates:
        try:
            coordinates_in_mask.append(mask[y][x] == 1)
        except IndexError:
            coordinates_in_mask.append(False)
    coordinates_in_mask = np.array(coordinates_in_mask)
    return coordinates_in_mask

def find_intersection_masks(mask1, mask2, mask3, mask4):
    return mask1 & mask2 & mask3 & mask4


if __name__ == "__main__":
    for i in [1, 2, 3, 4]:
        plot_projection(i, (0, 0, -10))
