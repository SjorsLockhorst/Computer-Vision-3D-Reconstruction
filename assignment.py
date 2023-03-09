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
from config import conf

block_size = 1.0

scale = 4  # Factor by which everything is scaled up
frame = 1  # Frame to initialise


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
    """Creates entire voxel block of fixed size"""
    size_x = 32
    size_y = 32
    size_z = 64
    scale_factor = conf.STRIDE_LEN / scale 
    voxel_block = np.zeros((size_x * size_y * size_z, 3))

    counter = 0
    for x in range(size_x):
        for y in range(size_y):
            for z in range(size_z):
                voxel_block[counter] = [x, y, -z]
                counter += 1

    return voxel_block * scale_factor

def create_all_voxels_set():
    """Create all voxels as a set"""
    size_x = 32
    size_y = 32
    size_z = 64
    scale_factor = conf.STRIDE_LEN / scale 
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

    # Load lookup table if it exists and we don't want to reload
    lookup_table_path = os.path.abspath(os.path.join("data", "lookup_table.pickle"))
    if os.path.exists(lookup_table_path) and not reload:
        with open(lookup_table_path, "rb") as file:
            return pickle.load(file)

    size_x = 32
    size_y = 32
    size_z = 64

    max_y, max_x = conf.CAM_RES
    total_voxels = size_x * size_y * size_z 

    to_include = np.ones(total_voxels).astype(bool)

    # Init so that first camera has access to all voxels
    included_voxels = create_all_voxels_set()
    lookup_table = defaultdict(dict)

    for cam_num in conf.CAMERAS:
        mtx, dist = load_intr_calibration(cam_num)
        rvec, tvec = load_extr_calibration(cam_num)

        scaled_voxels = np.array(list(included_voxels))

        scale_factor = conf.STRIDE_LEN / scale 

        # Make sure projection is done from the middle of voxel cube
        middle_of_voxels = scaled_voxels - scale_factor / 2
        coords = cv.projectPoints(middle_of_voxels, rvec, tvec, mtx, dist)[0]

        coords = np.squeeze(coords)

        # Only add if already included and within current image
        to_include = to_include & (coords[:, 0] < max_y) & ( coords[:, 1] < max_x)

        voxels_to_include = scaled_voxels[to_include]
        coords_to_include = coords[to_include]

        tuple_voxels = [tuple(voxel) for voxel in voxels_to_include]
        for voxel, coords in zip(tuple_voxels, coords_to_include):

            # Only add voxels that are already selected
            if voxel in included_voxels:
                lookup_table[voxel][cam_num] = tuple(coords.astype(int))
        # Make sure only voxels that fall within image plain are added
        included_voxels = included_voxels & set(tuple_voxels)

    # Save lookup table
    with open(lookup_table_path, "wb") as file:
        pickle.dump(lookup_table, file)
    return lookup_table


def draw_axes_on_image(cam_num, frame_id):
    """Helper function for testing, draws axes in 2d image."""
    point = (0,0,0)
    mtx, dist = load_intr_calibration(cam_num)
    rvec, tvec = load_extr_calibration(cam_num)
    coords = cv.projectPoints(point, rvec, tvec, mtx, dist)[0].astype(int)
    vid_dir = conf.main_vid_path(cam_num)
    vid = cv.VideoCapture(vid_dir)
    img = get_frame(vid, frame_id)
    x_img, y_img = coords[0][0]
    return draw_axes_from_zero(img, conf.STRIDE_LEN, mtx, dist, rvec, tvec, (x_img, y_img))


def plot_projection(cam_num, point):
    """Helper function for testing, draws a given 3d point in 2d image."""
    rescaled_point = tuple(map(lambda x: x * conf.STRIDE_LEN / scale, point))
    mtx, dist = load_intr_calibration(cam_num)
    rvec, tvec = load_extr_calibration(cam_num)
    coords = cv.projectPoints(rescaled_point, rvec, tvec, mtx, dist)[0].astype(int)
    vid_dir = conf.main_vid_path(cam_num)
    vid = cv.VideoCapture(vid_dir)
    img = get_frame(vid, 2)
    x_img, y_img = coords[0][0]
    img = draw_axes_on_image(cam_num, 2)
    img[y_img: y_img+ 10,x_img: x_img+ 10 ] = 255
    cv.imshow("test", img)
    cv.waitKey(0)


def set_voxel_positions(width, height, depth):
    """Calculate final voxel array"""
    global frame  # Frame which to show

    H = 11
    S = 13
    V = 12

    lookup_table = create_lookup_table()
    voxels_to_draw = set(lookup_table.keys())

    for cam in conf.CAMERAS:
        bg_model = load_background_model(cam)
        vid_dir = conf.main_vid_path(cam)
        vid = cv.VideoCapture(vid_dir)
        length = vid.get(cv.CAP_PROP_FRAME_COUNT)

        img = get_frame(vid, frame % length)
        mask = substract_background(bg_model, img, (H, S, V))[1]
        is_in_mask = in_mask(lookup_table, cam, mask)

        # Make sure that only voxels that are already in all previous masks are added
        voxels_to_draw = voxels_to_draw & is_in_mask

    frame += 1

    # Put voxels in right scale and rearange axes for openGL
    scale_factor = conf.STRIDE_LEN / scale 
    return [(x / scale_factor, -1 * z / scale_factor, y / scale_factor) for x, y, z in voxels_to_draw]

def get_cam_positions():
    """Generates camera positions."""

    cam_positions = []
    for cam in conf.CAMERAS:
        rvec, tvec = load_extr_calibration(cam)
        tvec /= conf.STRIDE_LEN / scale
        # Calculate rotation matrix and camera position
        rot_mat = cv.Rodrigues(rvec)[0]
        cam_pos = -np.matrix(rot_mat).T * np.matrix(tvec)
        cam_pos *= block_size
        cam_positions.append(cam_pos)

    return [[cam_pos[0], -1 * cam_pos[2], cam_pos[1]] for cam_pos in cam_positions]


def get_cam_rotation_matrices(verbose=False):
    """Rotates each camera appropriatly"""

    cam_angles = []
    for i in conf.CAMERAS:
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
    """Check if all coordinates in a lookup table are in mask of a certain camera."""
    coordinates_in_mask = set()
    for voxel, cam_map in lookup_table.items():
        x, y = cam_map[cam_num]
        if mask[y][x] == 1:
            coordinates_in_mask.add(voxel)
    return coordinates_in_mask
