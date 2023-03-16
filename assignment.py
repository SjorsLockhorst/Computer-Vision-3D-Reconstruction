import os
from collections import defaultdict
import pickle

import cv2 as cv
import glm
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from background import substract_background_new, load_all_background_models, create_new_bg_model
from calibration import (
    draw_axes_from_zero,
    get_frame,
    load_extr_calibration,
    load_intr_calibration,
)
from config import conf
from clustering import find_and_classify_people, cluster_and_create_color_model, test_colors

block_size = 1.0

scale = 2  # Factor by which everything is scaled up
frame = 1  # Frame to initialise


def generate_grid(width, depth):
    # Generates the floor grid locations
    # You don't need to edit this function
    data, colors = [], []
    for x in range(width):
        for z in range(depth):
            data.append([x*block_size - width/2, -
                        block_size, z*block_size - depth/2])
            colors.append([1.0, 1.0, 1.0] if (x+z) % 2 == 0 else [0, 0, 0])
    return data, colors


def create_clustering_voxel_set():
    """Create all voxels as a set"""
    scale_factor = conf.STRIDE_LEN / scale
    voxel_block = set()

    for x in range(-conf.VOXEL_X // 2, conf.VOXEL_X // 2):
        for y in range(-conf.VOXEL_Y // 2, conf.VOXEL_Y):
            for z in range(conf.VOXEL_Z):
                voxel_block.add(
                    (x * scale_factor, y * scale_factor, -z * scale_factor))

    return voxel_block


def create_all_voxels_set():
    """Create all voxels as a set"""
    scale_factor = conf.STRIDE_LEN / scale
    voxel_block = set()

    for x in range(conf.VOXEL_X):
        for y in range(conf.VOXEL_X):
            for z in range(conf.VOXEL_Z):
                voxel_block.add(
                    (x * scale_factor, y * scale_factor, -z * scale_factor))

    return voxel_block


def create_lookup_table(reload=False, optim=False, voxel_generator=create_all_voxels_set):
    """
    Create lookup table with mapping of voxel world position to image coordinates per 
    camera.
    """

    # Load lookup table if it exists and we don't want to reload
    lookup_table_path = conf.lookup_table_path()
    if os.path.exists(lookup_table_path) and not reload:
        with open(lookup_table_path, "rb") as file:
            return pickle.load(file)

    max_y, max_x = conf.CAM_RES

    # to_include = np.ones(total_voxels).astype(bool)

    # Init so that first camera has access to all voxels
    included_voxels = voxel_generator()
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
        to_include = (coords[:, 0] < max_x) & (coords[:, 1] < max_y)

        voxels_to_include = scaled_voxels[to_include]
        coords_to_include = coords[to_include]

        tuple_voxels = [tuple(voxel) for voxel in voxels_to_include]
        for voxel, coords in tqdm(zip(tuple_voxels, coords_to_include)):

            # Only add voxels that are already selected
            if voxel in included_voxels:
                lookup_table[voxel][cam_num] = tuple(coords.astype(int))
            else:
                lookup_table[voxel][cam_num] = tuple(coords.astype(int))

        # Make sure only voxels that fall within image plain are added
        if optim:
            included_voxels = included_voxels & set(tuple_voxels)
    # Save lookup table
    with open(lookup_table_path, "wb") as file:
        pickle.dump(lookup_table, file)
    return lookup_table


def draw_axes_on_image(cam_num, frame_id):
    """Helper function for testing, draws axes in 2d image."""
    point = (0, 0, 0)
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
    coords = cv.projectPoints(rescaled_point, rvec, tvec, mtx, dist)[
        0].astype(int)
    vid_dir = conf.main_vid_path(cam_num)
    vid = cv.VideoCapture(vid_dir)
    img = get_frame(vid, 2)
    x_img, y_img = coords[0][0]
    img = draw_axes_on_image(cam_num, 2)
    img[y_img: y_img + 10, x_img: x_img + 10] = 255
    cv.imshow("test", img)
    cv.waitKey(0)


def generate_voxels(frame, bg_models, verbose=False):
    lookup_table = create_lookup_table()
    voxels_to_draw = set(lookup_table.keys())

    masks = []
    contours = []
    for cam, bg_model in zip(conf.CAMERAS, bg_models):
        vid_dir = conf.main_vid_path(cam)
        vid = cv.VideoCapture(vid_dir)

        img = get_frame(vid, frame)
        mask, biggest = substract_background_new(img, bg_model)
        masks.append(masks)
        contours.append(biggest)
        if verbose:
            print(f"Cam: {cam}, {len(biggest)} contours.")
            cv.imshow(f"Mask {cam}", mask)
        is_in_mask = in_mask(lookup_table, cam, mask)

        # Make sure that only voxels that are already in all previous masks are added
        voxels_to_draw = voxels_to_draw & is_in_mask

    return list(voxels_to_draw), masks, contours


def get_voxels_in_world_coods(width, height, depth, frame, bg_models, verbose=False):
    voxels = generate_voxels(width, height, depth,
                             frame, bg_models, verbose=verbose)
    return np.array(voxels)


def set_voxel_positions(width, height, depth, opengl=True):
    """Calculate final voxel array"""
    global frame

    bg_models = load_all_background_models()
    color_models = []
    for cam in conf.CAMERAS:
        color_models.append(conf.load_color_model(cam))

    voxels, _, _ = generate_voxels(width, height, depth, frame, bg_models)
    clusters, centers, cluster_colors, preds = find_and_classify_people(
        voxels, frame, bg_models, color_models)

    # Put voxels in right scale and rearange axes for openGL
    scale_factor = conf.STRIDE_LEN / scale

    colors = [(255, 255, 0), (255, 0, 255), (0, 0, 255), (0, 255, 0)]
    frame += 100
    flipped_voxels = [
        (x / scale_factor, -1 * z / scale_factor, y / scale_factor)
        for x, y, z in voxels
    ]
    colors = [(255, 0, 0)]*len(flipped_voxels)
    return flipped_voxels, colors


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

    return [[cam_pos[0], -1 * cam_pos[2], cam_pos[1]] for cam_pos in cam_positions], [[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0], [1.0, 1.0, 0]]


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
        if cam_num in cam_map:
            x, y = cam_map[cam_num]
            if mask[y][x] == 1:
                coordinates_in_mask.add(voxel)

    return coordinates_in_mask


def plot_clusters(clusters, centers):
    x_y_clusters = [cluster[:, 0:2] for cluster in clusters]
    plt.clf()
    for cluster in x_y_clusters:
        plt.scatter(cluster[:, 0], cluster[:, 1])
    plt.scatter(centers[:, 0], centers[:, 1], s=80, c='y', marker='s')
    plt.xlabel('x'), plt.ylabel('y')
    ax = plt.gca()
    ax.set_xlim((-2000, 2000))
    ax.set_ylim((-2000, 5000))
    plt.pause(0.05)



def find_trajectory(verbose=False, show_cluster_plot=False, show_reference_frame=False, show_colors=False):
    CAM = 2
    HALFWAY_Z = -800

    mtx, dist = conf.load_intr_calib(CAM)
    rvec, tvec = conf.load_extr_calib(CAM)
    vid_path = conf.main_vid_path(CAM)
    vid = cv.VideoCapture(vid_path)
    length = int(vid.get(cv.CAP_PROP_FRAME_COUNT))

    bg_models = []
    color_models = []

    for cam in conf.CAMERAS:
        bg_models.append(create_new_bg_model(cam))
        color_models.append(conf.load_color_model(cam))

    steps = [[], [], [], []]

    for frame_id in tqdm(range(0, length, 24)):
        img = get_frame(vid, frame_id)
        if img is None:
            break
        voxels, _, _ = generate_voxels(frame_id, bg_models, verbose=verbose)
        clusters, centers, colors, preds = find_and_classify_people(
                voxels, frame_id, color_models, verbose=verbose)

        # if len(centers) != 4:
        #     continue

        for center, pred in zip(centers, preds):
            steps[pred].append(center)

        if show_cluster_plot:
            plot_clusters(clusters, centers)

        if show_colors:
            _, biggest = substract_background_new(img, bg_models[CAM - 1])

            new_mask = np.zeros(img.shape, dtype="uint8")
            three_d_centers = np.zeros(
                (centers.shape[0], centers.shape[1] + 1))
            three_d_centers[:, :2] = centers
            three_d_centers[:, 2] = HALFWAY_Z
            projected_centers, _ = cv.projectPoints(
                three_d_centers, rvec, tvec, mtx, dist)
            projected_centers = np.squeeze(projected_centers)
            for center, color in zip(projected_centers, colors):
                x, y = map(int, center)
                for contour in biggest:
                    if cv.pointPolygonTest(contour, center, False) > 0:
                        cv.fillPoly(new_mask, [contour], color.mean(axis=0))

            cv.imshow("new_mask", new_mask)

        if show_reference_frame:
            img = get_frame(vid, frame_id)
            cv.imshow("Frame of reference cam", img)
        if show_colors or show_reference_frame or verbose:
            if cv.waitKey(1) == ord("q"):
                cv.destroyAllWindows()

        plt.clf()
        a, b, c, d = [np.array(step) for step in steps]
        print(len(a), len(b), len(c), len(d))
        plt.title("test")
        plt.scatter(a[:, 0], a[:, 1])
        plt.scatter(b[:, 0], b[:, 1], c='r')
        plt.scatter(c[:, 0], c[:, 1], c='g')
        plt.scatter(d[:, 0], d[:, 1], c='b')
        plt.xlabel('x'), plt.ylabel('y')
        ax = plt.gca()
        ax.set_xlim((-2000, 2000))
        ax.set_ylim((-1000, 3000))
        plt.pause(0.05)



def find_good_frame(cam):
    vid_path = conf.main_vid_path(cam)
    vid = cv.VideoCapture(vid_path)
    length = int(vid.get(cv.CAP_PROP_FRAME_COUNT))
    bg_model = create_new_bg_model(cam)


    usable_frames = {}
    for frame_id in tqdm(range(length - 1)):
        frame = get_frame(vid, frame_id)
        full_mask, contours = substract_background_new(frame, bg_model)
        if len(contours) == 4:
            usable_frames[frame_id] = frame
            print(frame_id)
            cv.imshow("Frame", frame)
            cv.waitKey(100)





if __name__ == "__main__":
    # create_lookup_table(reload=True, voxel_generator=create_clustering_voxel_set)
    # bg_models = []
    # for cam in conf.CAMERAS:
    #     bg_models.append(create_new_bg_model(cam))
    # #
    # # find_trajectory()
    # frame_cam_map = {1:49, 2: 480, 4:511}
    # # #
    # voxels, _, _ = generate_voxels(1, bg_models)
    # base_model = cluster_and_create_color_model(voxels, None, frame=1, base_cam=3)
    #
    # for cam, frame in frame_cam_map.items():
    #     voxels, _, _ = generate_voxels(frame, bg_models)
    #     cluster_and_create_color_model(voxels, base_model, frame=frame, base_cam=cam)
    #     print(cam, frame)

    find_trajectory(show_cluster_plot=True)
    # find_good_frame(2)
    # for i in range(523, 540):
    #     voxels, _, _ = generate_voxels(i, bg_models)
    #     test_colors(voxels, i, 4, bg_models)
