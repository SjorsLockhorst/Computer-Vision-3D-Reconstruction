import glm
import random
import numpy as np

import cv2 as cv

from online import load_internal_calibrations

block_size = 1.0


def generate_grid(width, depth):
    # Generates the floor grid locations
    # You don't need to edit this function
    data = []
    for x in range(width):
        for z in range(depth):
            data.append([x*block_size - width/2, -block_size, z*block_size - depth/2])
    return data


def set_voxel_positions(width, height, depth):
    # Generates random voxel locations
    # TODO: You need to calculate proper voxel arrays instead of random ones.
    data = []
    for x in range(width):
        for y in range(height):
            for z in range(depth):
                if random.randint(0, 1000) < 5:
                    data.append([x*block_size - width/2, y*block_size, z*block_size - depth/2])
    return data


def get_cam_positions():
    # Generates dummy camera locations at the 4 corners of the room
    
    # TODO: You need to input the estimated locations of the 4 cameras in the world coordinates.
    mtx1, dist1, rvec1, tvec1 = load_internal_calibrations(1)
    mtx2, dist2, rvec2, tvec2 = load_internal_calibrations(2)
    mtx3, dist3, rvec3, tvec3 = load_internal_calibrations(3)
    mtx4, dist4, rvec4, tvec4 = load_internal_calibrations(4)
    tvec1 = tvec1 / 115
    tvec2 = tvec2 / 115
    tvec3 = tvec3 / 115
    tvec4 = tvec4 / 115
    
    # Calculate rotation matrix and camera position
    rotM1 = cv.Rodrigues(rvec1)[0]
    cameraPosition1 = -np.matrix(rotM1).T * np.matrix(tvec1)
    
    rotM2 = cv.Rodrigues(rvec2)[0]
    cameraPosition2 = -np.matrix(rotM2).T * np.matrix(tvec2)
    
    rotM3 = cv.Rodrigues(rvec3)[0]
    cameraPosition3 = -np.matrix(rotM3).T * np.matrix(tvec3)
    
    rotM4 = cv.Rodrigues(rvec4)[0]
    cameraPosition4 = -np.matrix(rotM4).T * np.matrix(tvec4)

    return [[cameraPosition1[0] * block_size, -1 * cameraPosition1[2] * block_size, cameraPosition1[1] * block_size],
            [cameraPosition2[0] * block_size, -1 * cameraPosition2[2] * block_size, cameraPosition2[1] * block_size],
            [cameraPosition3[0] * block_size, -1 * cameraPosition3[2] * block_size, cameraPosition3[1] * block_size],
            [cameraPosition4[0] * block_size, -1 * cameraPosition4[2] * block_size, cameraPosition4[1] * block_size]]


def get_cam_rotation_matrices():
    # Generates dummy camera rotation matrices, looking down 45 degrees towards the center of the room
    # TODO: You need to input the estimated camera rotation matrices (4x4) of the 4 cameras in the world coordinates.
    # cam_angles = [[0, 45, -45], [0, 135, -45], [0, 225, -45], [0, 315, -45]]
    # cam_rotations = [glm.mat4(1), glm.mat4(1), glm.mat4(1), glm.mat4(1)]
    # for c in range(len(cam_rotations)):
    #     cam_rotations[c] = glm.rotate(cam_rotations[c], cam_angles[c][0] * np.pi / 180, [1, 0, 0])
    #     cam_rotations[c] = glm.rotate(cam_rotations[c], cam_angles[c][1] * np.pi / 180, [0, 1, 0])
    #     cam_rotations[c] = glm.rotate(cam_rotations[c], cam_angles[c][2] * np.pi / 180, [0, 0, 1])
    # print(cam_rotations)
    # return cam_rotations

    cams = [1, 2, 3, 4]
    cam_angles = []
    for i in cams:
        _, _, rvec, tvec = load_internal_calibrations(i)
        rot_m = cv.Rodrigues(rvec)[0]
        with_zeros = np.pad(rot_m, ((0, 1), (0, 1)))
        with_zeros[with_zeros.shape[0] - 1, with_zeros.shape[1] - 1] = 1
        

        rotation = glm.mat4(with_zeros)
        for idx, val in enumerate(np.squeeze(tvec)):
            rotation[idx, 3] = val

        rotation = glm.rotate(rotation, glm.radians(90), (0, 0, 1))
        cam_angles.append(rotation)

    print(cam_angles)
    return cam_angles

if __name__ == "__main__":
    get_cam_rotation_matrices()
