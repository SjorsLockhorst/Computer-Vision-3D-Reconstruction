import glm
import numpy as np
import itertools

import cv2 as cv

from online import load_internal_calibrations, load_external_calibrations

block_size = 1.0
scale = 2


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
    mtx, dist = load_internal_calibrations(cam_num)
    rvec, tvec = load_external_calibrations(cam_num)
    size_x = 32
    size_y = 32
    size_z = 64
    scale_factor = 115 / scale 
    voxel_block = np.zeros((32 * 32 * 64, 3))

    counter = 0
    for x in range(size_x):
        for y in range(size_y):
            for z in range(size_z):
                voxel_block[counter] = [x, y, z]
                counter += 1

    scaled_voxels = voxel_block * scale_factor

    coords = cv.projectPoints(scaled_voxels, rvec, tvec, mtx, dist)[0]
    lookup_table = {}
    for voxel_coord, img_coord in zip(scaled_voxels, np.squeeze(coords)):
        scaled_back_voxel = voxel_coord / scale_factor
        lookup_table[tuple(scaled_back_voxel.astype(int))] = img_coord.astype(int)
    return lookup_table

    #     break


def set_voxel_positions(width, height, depth):
    # Generates random voxel locations
    # TODO: You need to calculate proper voxel arrays instead of random ones.
    size_x = 32
    size_y = 32
    size_z = 64
    voxel_block = list(itertools.product([x for x in range(size_x)], [
                z for z in range(size_z)], [y for y in range(size_y)]))
    # for x in range(width):
    #     for y in range(height):
    #         for z in range(depth):
    #             if random.randint(0, 1000) < 5:
    #                 data.append([x*block_size - width/2, y*block_size, z*block_size - depth/2])
    print(data)
    return data


def get_cam_positions():
    # Generates dummy camera locations at the 4 corners of the room

    cams = [1, 2, 3, 4]
    cam_positions = []
    for cam in cams:
        rvec, tvec = load_external_calibrations(cam)
        tvec /= 115 / scale
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
        rvec, tvec = load_external_calibrations(i)
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
        coordinates_in_mask.append(mask[x][y] == 1)
    coordinates_in_mask = np.array(coordinates_in_mask)
    return coordinates_in_mask

def find_intersection_masks(mask1, mask2, mask3, mask4):
    return mask1 == mask2 == mask3 == mask4


if __name__ == "__main__":
    lookup_table = create_lookup_table(2)
    print(lookup_table[(1, 2, 3)])
    # get_cam_rotation_matrices()
