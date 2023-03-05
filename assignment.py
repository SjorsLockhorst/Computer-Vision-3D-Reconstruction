import os
from collections import defaultdict
import pickle

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
from config import STRIDE_LEN, get_cam_dir, CAMERAS, CAM_RES

block_size = 1.0
scale = 4
frame = 1


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


def create_lookup_table(reload=False):
    """
    Create lookup table with mapping of voxel world position to image coordinates per 
    camera.
    """

    lookup_table_path = os.path.abspath(os.path.join("data", "lookup_table.pickle"))
    if os.path.exists(lookup_table_path) and not reload:
        with open(lookup_table_path, "rb") as file:
            return pickle.load(file)

    size_x = 32
    size_y = 32
    size_z = 64

    max_y, max_x = CAM_RES
    total_voxels = size_x * size_y * size_z 

    to_include = np.ones(total_voxels).astype(bool)
    included_voxels = create_all_voxels_set()
    lookup_table = defaultdict(dict)

    for cam_num in CAMERAS:
        mtx, dist = load_intr_calibration(cam_num)
        rvec, tvec = load_extr_calibration(cam_num)

        scaled_voxels = np.array(list(included_voxels))

        scale_factor = STRIDE_LEN / scale 
        middle_of_voxels = scaled_voxels - scale_factor / 2
        coords = cv.projectPoints(middle_of_voxels, rvec, tvec, mtx, dist)[0]

        coords = np.squeeze(coords)
        to_include = to_include & (coords[:, 0] < max_y) & ( coords[:, 1] < max_x)
        voxels_to_include = scaled_voxels[to_include]
        coords_to_include = coords[to_include]

        tuple_voxels = [tuple(voxel) for voxel in voxels_to_include]
        for voxel, coords in zip(tuple_voxels, coords_to_include):
            if voxel in included_voxels:
                lookup_table[voxel][cam_num] = tuple(coords.astype(int))
        included_voxels = included_voxels & set(tuple_voxels)

    with open(lookup_table_path, "wb") as file:
        pickle.dump(lookup_table, file)
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
    # Hard coded hsv values
    H = 2
    S = 8
    V = 13

    global frame
    lookup_table = create_lookup_table()
    voxels_to_draw = set(lookup_table.keys())

    for cam in CAMERAS:
        bg_model = load_background_model(cam)
        vid = cv.VideoCapture(os.path.abspath(os.path.join(get_cam_dir(cam), "video.avi")))
        length = vid.get(cv.CAP_PROP_FRAME_COUNT)

        img = get_frame(vid, frame % length)
        mask = substract_background(bg_model, img, H, S, V, dilate=True, erode=True)[1]
        is_in_mask = in_mask(lookup_table, cam, mask)
        voxels_to_draw = voxels_to_draw & is_in_mask

    frame += 1

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


def get_cam_rotation_matrices(verbose=False):

    cams = [1, 2, 3, 4]
    cam_angles = []
    for i in cams:
        rvec, tvec = load_extr_calibration(i)
        rot_m = cv.Rodrigues(rvec)[0]
        if verbose:
            print(f"Cam: {i}")
            print("rvec")
            print(rvec)
            print("tvec")
            print(tvec)
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
