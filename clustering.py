import pickle

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from tqdm import tqdm

from config import conf
from calibration import get_frame
from background import create_new_bg_model, substract_background_new
from color_model import fit_color_model, predict_color_model


def create_best_scores(table, use_best_row = True):

    best_scores = np.zeros(table[0].shape)
    best_scores[:, :] = -np.infty

    for i in range(table.shape[0]):
        scores = table[i]
        for j in range(table.shape[1]):
            if scores[j].max() > best_scores[j].max():
                best_scores[j] = scores[j]

    list_mean_max_row_cams = []
     
    for cam in table:
        max_row = np.max(cam, axis=1)
        mean_max_row = np.mean(max_row)
        list_mean_max_row_cams.append(mean_max_row)
   
    index_cam = np.argmax(list_mean_max_row_cams)
    
    new_table = table[index_cam].astype(float)
    if use_best_row:
        new_table = best_scores

    mapping = np.zeros(table.shape[1], dtype=int)
    
    for i in range(table.shape[1]):
        max_index = np.unravel_index(new_table.argmax(), new_table.shape)
        row, column = max_index
        mapping[row] = column
        new_table[row, :] = -np.inf
        new_table[:, column] = -np.inf 

    
    return mapping

def kmeans_clustering(voxel_arr, n_clusters = 4):
    """Preform k-means clustering on voxels."""

    # Select only x y points for voxel
    x_y_points = voxel_arr[:, :2].astype("float32")

    # Set criteria for algorithm
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)

    # Run k-means
    ret, labels, centers = cv.kmeans(
        x_y_points, n_clusters, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)

    # If couldn't cluster, raise exception
    if not ret:
        raise Exception("Couldn't cluster!!!")

    # Use labels to seperate voxels into found clusters
    flattened_labels = labels.ravel()
    cluster_masks = [flattened_labels == i for i in range(n_clusters)]
    clusters = [voxel_arr[mask] for mask in cluster_masks]

    # Plot the data
    return clusters, centers


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

# Function that thresholds clusters, now at 0 for test
def threshold_clusters(clusters):
    n_valid_clusters = 0
    for cluster in clusters:
        print("CLUSTER SIZE")
        print(len(cluster))
        if len(cluster) > 500:
            n_valid_clusters += 1
    return n_valid_clusters

def cluster_voxels(voxels, frame, bg_models, verbose=False):
    """Cluster voxels given a certain frame."""
    voxel_arr = np.array(voxels)
    # camera_contour_lengths = np.array([len(contour) for contour in contours])
    # return kmeans_clustering(voxel_arr, n_clusters=camera_contour_lengths.max())
    clusters, centers = kmeans_clustering(voxel_arr, n_clusters=4)
    last_k = 4
    n_valid_clusters = threshold_clusters(clusters)
    while n_valid_clusters != last_k:
        last_k -= 1
        clusters, centers = kmeans_clustering(voxel_arr, n_clusters=last_k)
        n_valid_clusters = threshold_clusters(clusters)
    return clusters, centers
   
    # return (*kmeans_clustering(voxel_arr, n_clusters=4), camera_contour_lengths)
    # Implementation outside of voxel
    # Call kmeans with k = 4 DONE
    # last_k = 4 DONE
    # call function that thresholds clusters, store in n_valid_clusters DONE
    # While n_valid_cluster != last_k: DONE
    # last_k -= 1 DONE
    # clusters = kmeans_clustering(valid_voxels, n_clusters = last_k) DONE
    # n_valid_clusters = threshold(cluster) DONE


def get_voxel_colors(voxels, frame, base_cam=3, show_cluster=False, above_z_ratio=1/2, below_z_ratio=1/10):
    """Get colors of a certain cluster"""

    # Load image from video
    vid_path = conf.main_vid_path(base_cam)
    vid = cv.VideoCapture(vid_path)
    img = get_frame(vid, frame)
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    # Load calibration
    mtx, dist = conf.load_intr_calib(base_cam)
    rvec, tvec = conf.load_extr_calib(base_cam)

    min_z = voxels[:, 2].min(axis=0)

    pants_cutof = voxels[:, 2] < min_z * above_z_ratio
    head_cutof = voxels[:, 2] > min_z - min_z * below_z_ratio

    voxels = voxels[pants_cutof & head_cutof]

    projected_points = cv.projectPoints(voxels, rvec, tvec, mtx, dist)[0]

    # Remove aribtrary dimension
    squeezed_points = np.squeeze(projected_points)

    # Array to store each color
    colors = np.zeros((squeezed_points.shape[0], 3))

    # Find color of each pixel
    mask = np.zeros((hsv.shape[0], hsv.shape[1]))
    for idx, point in enumerate(squeezed_points):
        x, y = point.astype(int)
        bgr = hsv[y][x]
        colors[idx] = bgr
        mask[y][x] = 1

    if show_cluster:
        cv.imshow("Cluster", mask)
        cv.waitKey(0)
    return colors


def test_colors(bg_models):
    frame = 1
    clusters, centers, _ = cluster_voxels(frame, bg_models)
    plot_clusters(clusters, centers)
    plt.show()
    CAM_NUM = 3
    cluster_colors = [get_voxel_colors(
        cluster, frame, base_cam=CAM_NUM, show_cluster=True) for cluster in clusters]

    vid_path = conf.main_vid_path(CAM_NUM)
    vid = cv.VideoCapture(vid_path)
    img = get_frame(vid, frame)

    bg_rem = substract_background_new(img, bg_models[CAM_NUM - 1])
    cv.imshow("no bg", bg_rem)
    cv.waitKey(0)
    for idx, colors in enumerate(cluster_colors):
        blank_image = np.zeros((800, 800, 3))
        blank_image[:, :, :] = colors.mean(axis=0)
        cv.imshow(f"Average color cluster {idx}", blank_image.astype("uint8"))
        cv.waitKey(0)


def cluster_and_create_color_model(bg_models, frame=1, base_cam=3):

    clusters, centers, _ = cluster_voxels(frame, bg_models)
    cluster_colors = [get_voxel_colors(
        cluster, frame, base_cam=base_cam) for cluster in clusters]

    gm = fit_color_model(cluster_colors)

    color_model_path = conf.color_model_path(base_cam)
    with open(color_model_path, "wb") as file:
        pickle.dump(gm, file)
    return gm


# Make sure we actually use different color models for each camera
def find_and_classify_people(voxels, frame_id, bg_models, color_models, verbose=False):
    
    clusters, centers = cluster_voxels(                         # TOOK CONTOUR LENS OUT
        voxels, frame_id, bg_models, verbose=verbose)

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

    preds = create_best_scores(np.array(all_scores))

    if verbose:
        print(f"Predicted classes: {preds}")
    sorted_index = np.array(preds).argsort()

    return np.array(clusters)[sorted_index], centers[sorted_index], np.array(cluster_colors)[sorted_index], preds


# TODO: Make use of bg subtraction to know if the camera sees 4 seperate entities or not
# TODO: Make a very verbose version that shows for each frame:
# 1. Clustering of voxels
# 2. Average color of each voxel
# 3. Predicted color based on color model

def find_trajectory(verbose=False, show_cluster_plot=False, show_reference_frame=False, show_colors = False):
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

    for frame_id in tqdm(range(0, length, 4)):
        img = get_frame(vid, frame_id)
        if img is None:
            break
        clusters, centers, colors, preds = find_and_classify_people(
            frame_id, bg_models, color_models, CAM, verbose=verbose)

        for idx, center in enumerate(centers):
            steps[idx].append(center)

        if show_cluster_plot:
            plot_clusters(clusters, centers)

        if show_colors:
            _, biggest = substract_background_new(img, bg_models[CAM - 1])

            new_mask = np.zeros(img.shape, dtype="uint8")
            three_d_centers = np.zeros((centers.shape[0], centers.shape[1] + 1))
            three_d_centers[:, :2] = centers
            three_d_centers[:, 2] = HALFWAY_Z
            projected_centers, _ = cv.projectPoints(three_d_centers, rvec, tvec, mtx, dist)
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
    plt.scatter(a[:, 0], a[:, 1])
    plt.scatter(b[:, 0], b[:, 1], c='r')
    plt.scatter(c[:, 0], c[:, 1], c='g')
    plt.scatter(d[:, 0], d[:, 1], c='b')
    plt.xlabel('x'), plt.ylabel('y')
    ax = plt.gca()
    ax.set_xlim((-2000, 2000))
    ax.set_ylim((-1000, 3000))
    plt.show()
    #
if __name__ == "__main__":
    bg_models = []
    for cam in conf.CAMERAS:
        bg_models.append(create_new_bg_model(cam))
    test_colors(bg_models)

    # find_trajectory()
    frame_cam_map = {1:1, 2:480, 3:96, 4:528}

    for cam, frame in frame_cam_map.items():
        cluster_and_create_color_model(bg_models, frame=frame, base_cam=cam)
