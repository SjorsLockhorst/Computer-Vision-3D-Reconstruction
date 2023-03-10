import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

from assignment import get_voxels_in_world_coods
from config import conf
from calibration import get_frame
from background import substract_background


def kmeans_clustering(voxel_arr, plot=False):
    """Preform k-means clustering on voxels."""

    # Select only x y points for voxel
    x_y_points = voxel_arr[:, :2].astype("float32")

    # Set criteria for algorithm
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)

    # Run k-means
    ret, labels, centers = cv.kmeans(
        x_y_points, 4, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)

    # If couldn't cluster, raise exception
    if not ret:
        raise Exception("Couldn't cluster!!!")

    # Use labels to seperate voxels into found clusters
    flattened_labels = labels.ravel()
    cluster_masks = [flattened_labels == i for i in range(4)]
    clusters = [voxel_arr[mask] for mask in cluster_masks]

    # Plot the data
    if plot:
        a, b, c, d = [cluster[:, 0:2] for cluster in clusters]
        plt.scatter(a[:, 0], a[:, 1])
        plt.scatter(b[:, 0], b[:, 1], c='r')
        plt.scatter(c[:, 0], c[:, 1], c='g')
        plt.scatter(d[:, 0], d[:, 1], c='b')
        plt.scatter(centers[:, 0], centers[:, 1], s=80, c='y', marker='s')
        plt.xlabel('x'), plt.ylabel('y')
        ax = plt.gca()
        ax.set_xlim((-2000, 2000))
        ax.set_ylim((-200, 3000))
        plt.pause(0.05)

    return clusters, centers


def cluster_voxels(frame, plot=False):
    """Cluster voxels given a certain frame."""
    points = get_voxels_in_world_coods(512, 256, 512, frame)
    voxel_arr = np.array(points)
    return kmeans_clustering(voxel_arr, plot=plot)


def get_voxel_colors(voxels, frame, base_cam=1, show_cluster=False, above_z_ratio=1/2):
    """Get colors of a certain cluster"""

    # Load image from video
    vid_path = conf.main_vid_path(base_cam)
    vid = cv.VideoCapture(vid_path)
    img = get_frame(vid, frame)
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    # Load calibration
    mtx, dist = conf.load_intr_calib(base_cam)
    rvec, tvec = conf.load_extr_calib(base_cam)

    voxels = voxels[voxels[:, 2] < voxels[:, 2].min(axis=0) * above_z_ratio]
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


def test_colors():
    frame = 1
    clusters, centers = cluster_voxels(frame, plot=True)
    plt.show()
    CAM_NUM = 3
    cluster_colors = [get_voxel_colors(
        cluster, frame, base_cam=CAM_NUM, show_cluster=True) for cluster in clusters]

    vid_path = conf.main_vid_path(CAM_NUM)
    vid = cv.VideoCapture(vid_path)
    img = get_frame(vid, frame)

    bg_model = conf.load_bg_model(CAM_NUM)

    bg_rem, _ = substract_background(
        bg_model, img, (conf.H_THRESH, conf.S_THRESH, conf.V_THRESH), n_biggest=4)
    cv.imshow("no bg", bg_rem)
    cv.waitKey(0)
    for idx, colors in enumerate(cluster_colors):
        blank_image = np.zeros((800, 800, 3))
        blank_image[:, :, :] = colors.mean(axis=0)
        print(blank_image.shape)
        cv.imshow(f"Average color cluster {idx}", blank_image.astype("uint8"))
        cv.waitKey(0)


if __name__ == "__main__":
    frame = 1
    clusters, centers = cluster_voxels(frame)
    CAM_NUM = 3
    cluster_colors = [get_voxel_colors(
        cluster, frame, base_cam=CAM_NUM) for cluster in clusters]

    init_means = [cluster.mean(axis=0) for cluster in cluster_colors]

    all_clusters = np.concatenate(cluster_colors)

    gmm = GaussianMixture(n_components=4, covariance_type='spherical',
                          max_iter=100_000, tol=1e-6, means_init=init_means)
    gmm.fit(all_clusters)
    print(gmm.means_)
