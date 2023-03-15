from config import conf

import numpy as np
from sklearn.mixture import GaussianMixture

# input is a list of HSV values for each pixel


def fit_color_model(cluster_hsv, prev_model):
    gms = []

    for hsv in cluster_hsv:
        gm = GaussianMixture(n_components=1, covariance_type='spherical', random_state=conf.RANDOM_STATE)
        gm.fit(hsv)
        gms.append(gm)

    return gms

# TODO: Make sure this takes into account multiple clusters, use some sort of softmax


def predict_color_model(gms, hsv_arr):
    scores = []
    for gm in gms:
        score = gm.score(hsv_arr)
        scores.append(score)

    return np.array(scores)


if __name__ == "__main__":
    hsv_array = np.random.rand(100, 100, 3)
    hsv_array[..., 0] *= 360
    n_pixels = hsv_array.shape[0] * hsv_array.shape[1]
    hsv_array_2d = hsv_array.reshape(n_pixels, 3)
    gmm = fit_color_model(hsv_array_2d)
    labels = gmm.predict_proba(hsv_array_2d)
    print(labels.min())

    labels = gmm.predict(hsv_array_2d * 100_000)
    print((labels == 0).sum() / len(labels))
