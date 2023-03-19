from config import conf

import numpy as np
from sklearn.mixture import GaussianMixture
import cv2 as cv

# input is a list of HSV values for each pixel


def fit_color_model(cluster_hsv, base_model):
    """Fit a color model based on clusters HSV, order by a base model."""
    cluster_hsv = np.array(cluster_hsv)
    if base_model is not None:
        preds = []
        for hsv in cluster_hsv:
            pred = predict_color_model(base_model, hsv.mean(axis=0).reshape(1, -1))
            preds.append(pred)

        ordering = eliminate(np.array(preds))
        cluster_hsv = cluster_hsv[np.argsort(ordering)]

    gms = []
    for idx, hsv in enumerate(cluster_hsv):
        gm = GaussianMixture(
            n_components=1, covariance_type='spherical', random_state=conf.RANDOM_STATE)
        gm.fit(hsv)
        gms.append(gm)
        to_display = np.zeros((400, 400, 3), dtype=np.uint8)
        to_display[:, :] = hsv.mean(axis=0)
        cv.imshow(f"Cluster {idx}", to_display)
        cv.waitKey(100)

    return gms


def predict_color_model(gms, hsv_arr ):
    """Get log likelihoods from a color model, based on an array of HSV values."""
    scores = []
    for gm in gms:
        score = gm.score(hsv_arr.mean(axis=0).reshape(1, -1))
        scores.append(score)

    return np.array(scores)


def eliminate(scores):
    """Select best scores iteratively."""
    mapping = np.zeros(scores.shape[0], dtype=int)
    
    for i in range(scores.shape[0]):
        max_index = np.unravel_index(scores.argmax(), scores.shape)
        row, column = max_index
        mapping[row] = column
        scores[row, :] = -np.inf
        scores[:, column] = -np.inf 
    
    return mapping
