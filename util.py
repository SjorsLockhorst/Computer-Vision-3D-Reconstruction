"""Module containing several utility functions."""

import os
import re

import cv2 as cv
import numpy as np

from config import DATA_DIR


def load_image(path):
    """Load image from path."""
    img = cv.imread(cv.samples.findFile(path))
    if img is None:
        raise OSError(f"Could not find file {path}")
    return img


def find_chessboard_corners(img, pattern_size, improve = True):
    """
    Find the chessboard corners and display them.
    Optionally improve them using cornerSubPix.
    """
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, corners = cv.findChessboardCorners(gray, pattern_size, None)
    if ret and improve:
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        ret, corners =  cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return ret, corners


def draw_rectangle(img, rect_heigth=100, rect_width=100):
    """Draw rectangle in the center of an image."""
    WHITE_PIXEL = (255, 255, 255)
    RECT_HEIGTH = 100
    RECT_WIDTH = 100

    heigth, width, _ = img.shape

    heigth_first_pixel = heigth // 2 - RECT_HEIGTH // 2
    width_first_pixel = width // 2 - RECT_WIDTH // 2

    heigth_last_pixel = heigth_first_pixel + RECT_HEIGTH
    width_last_pixel = width_first_pixel + RECT_WIDTH

    img[heigth_first_pixel:heigth_last_pixel,
        width_first_pixel:width_last_pixel] = WHITE_PIXEL
    return img


def draw_rect_img(img_path):
    """Draw a white rectangle on a picture of a cute cat."""
    CAT_PATH = os.path.join(DATA_DIR, img_path)
    cat_img = load_image(CAT_PATH)
    cat_rect_img = draw_rectangle(cat_img)
    cv.imshow("Cute cat", cat_rect_img)
    cv.waitKey(0)


def select_corners_event(event, x, y, flags, params):
    """Allow user to click corners of a chessboard, return coordinates."""

    # Get name for window, the image, and found corners from params
    window_name, img, corners  = params

    # Only on left mouse button
    if event == cv.EVENT_LBUTTONDOWN:
        # Specify order of clicking
        corner_texts = ["Top left", "Top right", "Bottom left", "Bottom right"]
        font = cv.FONT_HERSHEY_PLAIN

        # Show the point the user picked in the image
        PADDING = 5
        cv.putText(
            img,
            f"{corner_texts[len(corners)]}: ({x}, {y})",
            (x, y), font, 1,
            (0, 0, 255), 2)


        img[y-PADDING:y+PADDING, x-PADDING:x+PADDING] = (0, 0, 255)
        cv.imshow(window_name, img)

        # Add corner to selection
        corners.append((x, y))

        # If all four corners are selected, close all windows
        if len(corners) == 4:
            cv.destroyAllWindows()


def click_corners(img):
    """Click 4 corners and select their coordinates"""

    window_name = "Select corners"
    ui_img = img.copy()
    cv.imshow(window_name, ui_img)

    corners = []

    # Set mouse callback, will allow user to select 4 corners
    cv.setMouseCallback("Select corners", select_corners_event,
                        (window_name, ui_img, corners))

    # Wait forever, untill mousecallback destorys all windows.
    cv.waitKey(0)

    return corners

def perspective_transform(img):
    corners1 = click_corners(img)
    corners1  =  np.float32(corners1)
    max_x, max_y = corners1.max(axis=0).astype(int)
    pts2 = np.float32([[0, 0], [max_x, 0],
                       [0, max_y], [max_x, max_y]])
    mtx = cv.getPerspectiveTransform(corners1, pts2)
    return cv.warpPerspective(img, mtx, (img.shape[0], img.shape[1])), mtx

def interpolate_chessboard_with_perspective(img, pattern_size, **kwargs):
    res, mtx = perspective_transform(img)
    ret, corners = find_chessboard_corners(img, pattern_size, improve=False)
    if not ret:
        ret, corners = interpolate_chessboard(res, pattern_size, **kwargs)

    cv.drawChessboardCorners(res, pattern_size, corners, True)
    cv.imshow("transformed", res)
    cv.waitKey(0)
    mtx_inv = np.linalg.pinv(mtx)
    corners2 = cv.perspectiveTransform(corners, mtx_inv)
    return True, corners2



def interpolate(coords, axis, dims):
    """
    Interpolate between two coordinates.

    Parameters
    ----------
    coords : list of tuples (x, y)
        Two coordinates between which to interpolate.
    axis : int
        Whether to interpolate along axis 0 (x-axis) or 1 (y-axis).
    dims : tuple
        Amount of points to interpolate per axis.

    Returns
    -------
    list of tuples (x, y)
        List of interpolated points between coordinates, always in correct orientation
        as the original axis system.
    """
    # Order the two coordinates because np.interp needs increasing values of x
    sorted_coords = sorted(coords, key=lambda x: x[0])

    # Split coordinates into xp (x values) and fp (y values)
    xp = [corner[0] for corner in sorted_coords]
    fp = [corner[1] for corner in sorted_coords]

    # Generate some amount of equially spaced points based on the dims of a certain axis
    # This can be done because all squares in the chessboard are of same size
    # Float32 because that is what drawChessboardCorners expects
    est_xs = np.linspace(xp[0], xp[1], dims[axis], dtype="float32")

    # Interpolate the corresponding y values
    # Float32 because that is what drawChessboardCorners expects
    est_ys = np.interp(est_xs, xp, fp).astype("float32")

    interp_coords = list(zip(est_xs, est_ys))
    if sorted_coords[0] != coords[0]:
        interp_coords.reverse()
    return interp_coords


def interpolate_from_corners(img, clicked_corners, pattern_size):
    """Interpolate a grid from corners."""
    # Set axes
    X_AXIS = 0
    Y_AXIS = 1

    # List to store interpolated points
    interp_corners = []

    # Split out different corners
    top_left, top_right, bottom_left, bottom_right = clicked_corners

    # Interpolate vertically between given corners
    left_col = interpolate((top_left, bottom_left), Y_AXIS, pattern_size)
    right_col = interpolate((top_right, bottom_right), Y_AXIS, pattern_size)

    # # Interpolate vertically between bottom and top row of horizontally interpolated points
    for horizontal_corners in zip(left_col, right_col):
        interp_corners += interpolate(horizontal_corners, X_AXIS, pattern_size)

    # Expand dimensions to make it the expected shape for drawChessboardCorners
    return np.expand_dims(interp_corners, axis=1)


def interpolate_chessboard(img, pattern_size, window_size = (11, 11), improve_corner_points=True, improve_interp_points=True):
    """Start interpolation of chessboard"""
    # Obtain corners from UI
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    clicked_corners = click_corners(img.copy())
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    if improve_corner_points:
        clicked_corners = np.expand_dims(
            np.array(clicked_corners, dtype="float32"), axis=1)
        clicked_corners = cv.cornerSubPix(
            gray, clicked_corners, (11, 11), (-1, -1), criteria)
        clicked_corners = [tuple(row[0].astype(int))for row in clicked_corners]

    # Interpolate based on those corners
    interp_corners = interpolate_from_corners(
        img, clicked_corners, pattern_size=pattern_size)

    if improve_interp_points:
        return True, cv.cornerSubPix(gray, interp_corners, window_size, (-1, -1), criteria)

    # Return ret=True, and the interpolated corners, just like findChessboardCorners
    return True, interp_corners


def sample_files(dir_path, n, ext="jpg"):
    """Sample files from a directory."""

    # Make sure only files with correct extension are included
    filterd_files = [file for file in os.listdir(
        dir_path) if re.match(rf".*\.{ext}$", file)]

    # If sample size is too big, just return all files
    if n > len(filterd_files):
        selected_files = filterd_files
    # If sample size is lower than amount of files
    else:
        # Subsample first n files
        selected_files = list(np.take(filterd_files, np.linspace(0, n, dtype=int)))

    # Join each file to full path and return
    return [os.path.join(dir_path, file) for file in selected_files]

# if __name__ == "__main__":
#     from config import CHESS_DIMS
#
#     img = cv.imread("data/cam1/frames/frame_108.jpg")
#     res, mtx = perspective_transform(img)
#     _, corners = interpolate_chessboard(res, CHESS_DIMS, window_size=(2, 2))
#
#     cv.drawChessboardCorners(res, CHESS_DIMS, corners, True)
#     cv.imshow("transformed", res)
#     cv.waitKey(0)
#
#     mtx_inv = np.linalg.pinv(mtx)
#     corners2 = cv.perspectiveTransform(corners, mtx_inv)
#     cv.drawChessboardCorners(img, CHESS_DIMS, corners2, True)
#     #
#     cv.imshow("original", img)
#     cv.waitKey(0)
