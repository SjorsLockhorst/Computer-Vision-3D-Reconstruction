import pickle
import enum

import numpy as np
import cv2 as cv

from config import conf
from calibration import get_frame
from background import substract_background_new
from color_model import fit_color_model, predict_color_model


class EliminationMode(enum.Enum):

    BEST_ROW = enum.auto()
    WEIGHTED_AVG = enum.auto()
    BEST_MAT = enum.auto()


class ClusteringException(Exception):
    pass


def iterative_elimination(table, masks, contours, mode=EliminationMode.BEST_ROW):

    if mode == EliminationMode.BEST_ROW:
        best_scores = np.zeros(table[0].shape)
        best_scores[:, :] = -np.infty

        cam_cluster_map = {}
        for i in range(table.shape[0]):
            scores = table[i]
            for j in range(table.shape[1]):
                if scores[j].max() > best_scores[j].max():
                    best_scores[j] = scores[j]
                    cam_cluster_map[j] = i + 1

        # print("Mapping between camera and cluster:")
        # print(cam_cluster_map)

        new_table = best_scores

    elif mode == EliminationMode.WEIGHTED_AVG:
        mask_weights = np.array([mask.sum() for mask in masks])
        mask_weights_z = mask_weights / mask_weights.mean()

        contour_weights = np.array([len(contour) for contour in contours])
        contour_weights_z = contour_weights / contour_weights.mean()

        new_table = np.average(
            table, axis=0, weights=mask_weights_z + contour_weights_z)

    elif mode == EliminationMode.BEST_MAT:
        list_mean_max_row_cams = []

        for cam in table:
            max_row = np.max(cam, axis=1)
            mean_max_row = np.mean(max_row)
            list_mean_max_row_cams.append(mean_max_row)

        index_cam = np.argmax(list_mean_max_row_cams)
        # print(f"Using camera {index_cam + 1}")

        new_table = table[index_cam].astype(float)

    mapping = np.zeros(table.shape[1], dtype=int)

    for i in range(table.shape[1]):
        max_index = np.unravel_index(new_table.argmax(), new_table.shape)
        row, column = max_index
        mapping[row] = column
        new_table[row, :] = -np.inf
        new_table[:, column] = -np.inf

    return mapping


def kmeans_clustering(voxel_arr, n_clusters=4, clean_outliers=True):
    """Preform k-means clustering on voxels."""

    # Select only x y points for voxel
    x_y_points = voxel_arr[:, :2].astype("float32")

    # Set criteria for algorithm
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)

    # Run k-means
    ret, labels, centers = cv.kmeans(
        x_y_points, n_clusters, None, criteria, 10, cv.KMEANS_PP_CENTERS)

    # If couldn't cluster, raise exception
    if not ret:
        raise Exception("Couldn't cluster!!!")

    # Use labels to seperate voxels into found clusters
    flattened_labels = labels.ravel()
    cluster_masks = [flattened_labels == i for i in range(n_clusters)]
    clusters = [voxel_arr[mask] for mask in cluster_masks]

    if clean_outliers:
        clean_clusters = []
        for cluster, center in zip(clusters, centers):
            distances = np.linalg.norm(cluster[:, :2] - center, axis=1)
            mean_distance, std_distance = distances.mean(), distances.std()
            abs_deviance = abs(mean_distance - distances)
            z_scores = abs_deviance / std_distance
            clean_clusters.append(cluster[z_scores < 2])

        return clean_clusters, centers

    # Plot the data
    return clusters, centers


# Function that thresholds clusters, now at 0 for test
def threshold_clusters(clusters, centers):
    THRESHOLD = 200
    valid_clusters, valid_centers = [], []
    for cluster, center in zip(clusters, centers):
        if len(cluster) > THRESHOLD:
            valid_clusters.append(cluster)
            valid_centers.append(center)

    return valid_clusters, valid_centers


def cluster_voxels(voxels, frame, verbose=False):
    """Cluster voxels given a certain frame."""
    voxel_arr = np.array(voxels)
    clusters, centers = kmeans_clustering(voxel_arr, n_clusters=4)
    last_k = 4
    valid_clusters, valid_centers = threshold_clusters(clusters, centers)
    while len(valid_clusters) != last_k:
        if len(valid_clusters) == 0:
            raise ClusteringException
        last_k -= 1
        clusters, centers = kmeans_clustering(
            np.concatenate(valid_clusters), n_clusters=last_k)
        valid_clusters, valid_centers = threshold_clusters(clusters, centers)

    return valid_clusters, valid_centers


def get_voxel_colors(voxels, frame, base_cam=3, show_cluster=False):
    """Get colors of a certain cluster"""

    above_z_ratio = 5/10
    below_z_ratio = 2/10

    above_x_ratio = 3/10
    below_x_ratio = 3/10

    # Load image from video
    vid_path = conf.main_vid_path(base_cam)
    vid = cv.VideoCapture(vid_path)
    img = get_frame(vid, frame)
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    # Load calibration
    mtx, dist = conf.load_intr_calib(base_cam)
    rvec, tvec = conf.load_extr_calib(base_cam)

    min_z = voxels[:, 2].min(axis=0)
    max_z = voxels[:, 2].max(axis=0)
    dz = abs(min_z - max_z)

    min_x = voxels[:, 0].min(axis=0)
    max_x = voxels[:, 0].max(axis=0)
    dx = abs(min_x - max_x)

    pants_cutof = voxels[:, 2] < min_z + dz * above_z_ratio
    head_cutof = voxels[:, 2] > min_z + dz * below_z_ratio

    left_cutof = voxels[:, 0] > min_x + dx * above_x_ratio
    right_cutof = voxels[:, 0] < max_x - dx * below_x_ratio

    voxels = voxels[pants_cutof & head_cutof & left_cutof & right_cutof]

    projected_points = cv.projectPoints(voxels, rvec, tvec, mtx, dist)[0]

    # Remove aribtrary dimension
    squeezed_points = np.squeeze(projected_points)

    # Array to store each color
    colors = np.zeros((squeezed_points.shape[0], 3))

    # Find color of each pixel
    mask = np.zeros((hsv.shape[0], hsv.shape[1]))
    for idx, point in enumerate(squeezed_points):
        x, y = point.astype(int)
        hsv_pix = hsv[y][x]
        if hsv_pix[2] > 30:
            colors[idx] = hsv_pix
            mask[y][x] = 1

    if show_cluster:
        cv.imshow("Cluster", mask)
        cv.waitKey(0)
    return colors


def test_colors(voxels, frame, cam, bg_models):
    clusters, centers = cluster_voxels(voxels, frame, bg_models)
    cluster_colors = [get_voxel_colors(
        cluster, frame, base_cam=cam, show_cluster=True) for cluster in clusters]

    vid_path = conf.main_vid_path(cam)
    vid = cv.VideoCapture(vid_path)
    img = get_frame(vid, frame)

    bg_rem, _ = substract_background_new(img, bg_models[cam - 1])
    cv.imshow("no bg", bg_rem)
    cv.waitKey(0)
    for idx, colors in enumerate(cluster_colors):
        blank_image = np.zeros((800, 800, 3))
        blank_image[:, :, :] = colors.mean(axis=0)
        # bgr = cv.cvtColor(blank_image, cv.COLOR_HSV2BGR)
        cv.imshow(f"Average color cluster {idx}", blank_image.astype('uint8'))
        cv.waitKey(0)


def cluster_and_create_color_model(voxels, base_model, frame=1, base_cam=3):

    clusters, centers = cluster_voxels(voxels, frame)
    cluster_colors = [get_voxel_colors(
        cluster, frame, base_cam=base_cam) for cluster in clusters]

    gm = fit_color_model(cluster_colors, base_model)

    color_model_path = conf.color_model_path(base_cam)
    with open(color_model_path, "wb") as file:
        pickle.dump(gm, file)
    return gm


def find_and_classify_people(voxels, frame_id, color_models, masks, contours, verbose=False):

    clusters, centers = cluster_voxels(
        voxels, frame_id, verbose=verbose)

    preds = []
    cluster_colors = []
    all_scores = []

    for idx, cam in enumerate(conf.CAMERAS):
        cam_scores = []
        for cluster, center in zip(clusters, centers):
            colors = get_voxel_colors(cluster, frame_id, base_cam=cam)
            scores = predict_color_model(color_models[cam - 1], colors)
            cluster_colors.append(colors)
            cam_scores.append(scores)
        all_scores.append(cam_scores)

    preds = iterative_elimination(
        np.array(all_scores), masks, contours, mode=EliminationMode.BEST_ROW)

    if verbose:
        print(f"Predicted classes: {preds}")

    return clusters, centers, cluster_colors, preds


if __name__ == "__main__":
    pass
