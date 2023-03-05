import os
from collections import defaultdict

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
from config import STRIDE_LEN, get_cam_dir, CAMERAS

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


def create_all_voxels():
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

    return voxel_block * scale_factor

def create_all_voxels_set():
    size_x = 32
    size_y = 32
    size_z = 64
    scale_factor = STRIDE_LEN / scale 
    voxel_block = set()

    for x in range(size_x):
        for y in range(size_y):
            for z in range(size_z):
                voxel_block.add((x * scale_factor, y * scale_factor, -z * scale_factor))

    return voxel_block


def create_lookup_tables(dims):
    max_y, max_x = dims
    size_x = 32
    size_y = 32
    size_z = 64
    total_voxels = size_x * size_y * size_z 
    to_include = np.ones(total_voxels).astype(bool)
    included_voxels = create_all_voxels_set()
    lookup_table = defaultdict(dict)

    for cam_num in CAMERAS:
        mtx, dist = load_intr_calibration(cam_num)
        rvec, tvec = load_extr_calibration(cam_num)

        scaled_voxels = create_all_voxels()

        coords = cv.projectPoints(scaled_voxels, rvec, tvec, mtx, dist)[0]
        coords = np.squeeze(coords)
        to_include = to_include & (coords[:, 0] < max_y) & ( coords[:, 1] < max_x)
        voxels_to_include = scaled_voxels[to_include]
        coords_to_include = coords[to_include]

        tuple_voxels = [tuple(voxel) for voxel in voxels_to_include]
        for voxel, coords in zip(tuple_voxels, coords_to_include):
            if voxel in included_voxels:
                lookup_table[voxel][cam_num] = tuple(coords.astype(int))
        included_voxels = included_voxels & set(tuple_voxels)

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
    vid = cv.VideoCapture(os.path.abspath(os.path.join(get_cam_dir(1), "video.avi")))
    img = get_frame(vid, 2)
    lookup_table = create_lookup_tables((img.shape[0],img.shape[1]))
    voxels_to_draw = set(lookup_table.keys())

    for cam in cams:

        bg_model = load_background_model(cam)
        vid = cv.VideoCapture(os.path.abspath(os.path.join(get_cam_dir(cam), "video.avi")))
        img = get_frame(vid, 2)
        mask = substract_background(bg_model, img, H, S, V, dilate=True, erode=True)[1]
        is_in_mask = in_mask(lookup_table, cam, mask)
        voxels_to_draw = voxels_to_draw & is_in_mask

    # voxels_to_draw = find_intersection_masks(*voxels_in_mask)
    # selected = np.array(list(lookup_table.keys()))[voxels_to_draw]
    scale_factor = STRIDE_LEN / scale 
    return [(x / scale_factor, -1 * z / scale_factor, y / scale_factor) for x, y, z in voxels_to_draw]


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

def in_mask(lookup_table, cam_num, mask):
    coordinates_in_mask = set()
    for voxel, cam_map in lookup_table.items():
        x, y = cam_map[cam_num]
        if mask[y][x] == 1:
            coordinates_in_mask.add(voxel)
    return coordinates_in_mask

def find_intersection_masks(mask1, mask2, mask3, mask4):
    return mask1 & mask2 & mask3 & mask4


if __name__ == "__main__":
    # for i in [1, 2, 3, 4]:
    #     plot_projection(i, (0, 0, -10))
    cam = 1
    vid = cv.VideoCapture(os.path.abspath(os.path.join(get_cam_dir(cam), "video.avi")))
    img = get_frame(vid, 2)
    test = create_lookup_tables((img.shape[0], img.shape[1]))
