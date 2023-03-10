import numpy as np
from sklearn.mixture import GaussianMixture

def fit_gaussian(HSV): # input is a list of HSV values for each pixel in shirt
    gmm = GaussianMixture(n_components=2, covariance_type='full')
    gmm.fit(HSV)
    return gmm


if __name__ == "__main__":
    hsv_array = np.random.rand(100, 100, 3)
    hsv_array[..., 0] *= 360
    n_pixels = hsv_array.shape[0] * hsv_array.shape[1]
    hsv_array_2d = hsv_array.reshape(n_pixels, 3)
    gmm = fit_gaussian(hsv_array_2d)
    labels = gmm.predict_proba(hsv_array_2d)
    print(labels.min())

    labels = gmm.predict(hsv_array_2d * 100_000)
    print((labels == 0).sum() / len(labels))
