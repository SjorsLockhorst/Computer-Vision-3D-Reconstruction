"""Draw a cube on a chessboard using a calibrated camera."""
import os
import time

import cv2 as cv
import numpy as np

from config import DATA_DIR, CHESS_DIMS, STRIDE_LEN, get_cam_dir


def load_internal_calibrations(num):
    """Loads internal calibrations based on experiment number."""
    load_path = os.path.join(get_cam_dir(num), "calibration")

    mtx = np.load(os.path.join(
        load_path, "mtx.npy"), allow_pickle=True)
    dist = np.load(os.path.join(
        load_path, "dist.npy"), allow_pickle=True)
    rvec = np.load(os.path.join(
        load_path, "rvec.npy"), allow_pickle=True)
    tvec = np.load(os.path.join(
        load_path, "tvec.npy"), allow_pickle=True)
    
    return mtx, dist, rvec, tvec


def draw_axis_lines(img, corners, imgpts):
    """Draws x,y,z axis lines on a chessboard."""
    corners = np.int32(corners).reshape(-1, 2)
    imgpts = np.int32(imgpts).reshape(-1, 2)

    corner = tuple(corners[0].ravel())

    img = cv.line(img, corner, tuple(imgpts[0].ravel()), (0, 0, 255), 2)
    img = cv.line(img, corner, tuple(imgpts[1].ravel()), (0, 255, 0), 2)
    img = cv.line(img, corner, tuple(imgpts[2].ravel()), (255, 0, 0), 2)
    return img


def draw_cube(img, corners, imgpts):
    """Draw a cube on a chessboard."""
    imgpts = np.int32(imgpts).reshape(-1, 2)
    # draw ground floor
    img = cv.drawContours(img, [imgpts[:4]], -1, (240, 230, 140), 2)
    # draw pillars
    for i, j in zip(range(4), range(4, 8)):
        img = cv.line(img, tuple(imgpts[i]), tuple(
            imgpts[j]), (240, 230, 140), 2)
    # draw top layer
    img = cv.drawContours(img, [imgpts[4:]], -1, (240, 230, 140), 2)
    return img


def draw(img, mtx, dist, pattern_size, stride_len, corners = None, include_error=True):
    """
    Draw axis lines and cube on chessboard.
    """
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    if corners is None:
        ret, corners = cv.findChessboardCorners(gray, pattern_size, None)
    else:
        ret = True
    error = None
    if not ret:
        return img, error

    # Initialize criteria for SubPix module
    # criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    AXIS_UNIT_SIZE = 5
    # Create axis and object points
    axes = np.float32([[1, 0, 0], [0, 1, 0], [0, 0, -1]]
                      ).reshape(-1, 3) * stride_len * AXIS_UNIT_SIZE

    CUBE_UNIT_SIZE = 2
    cube_lines = np.float32(
        [[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0],
         [0, 0, -1], [0, 1, -1], [1, 1, -1], [1, 0, -1]]
    ) * stride_len * CUBE_UNIT_SIZE

    objp = np.zeros((pattern_size[0]*pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0],
                           0:pattern_size[1]].T.reshape(-1, 2) * stride_len

    # Find better corners with SubPix module
    # corners = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

    # Find the rotation and translation vectors.
    ret, rvecs, tvecs = cv.solvePnP(objp, corners, mtx, dist, useExtrinsicGuess=False, flags=(cv.SOLVEPNP_ITERATIVE))

    # Project 3D points to image plane for cube
    imgpts, jac = cv.projectPoints(cube_lines, rvecs, tvecs, mtx, dist)

    # Project 3D points to image plane with longer axis
    imgpts2, jac2 = cv.projectPoints(axes, rvecs, tvecs, mtx, dist)

    # Draw in 2D
    img = draw_cube(img, corners, imgpts)
    img = draw_axis_lines(img, corners, imgpts2)
    # Optionally show the reprojection error, along with reprojected points
    if include_error:
        error, projected_corners = reprojection_error(
                objp,
                corners,
                rvecs,
                tvecs,
                mtx,
                dist
        )
        cv.drawChessboardCorners(img, pattern_size, projected_corners, True)
    return img, error


def draw_live(initial_calibration=1, include_error = True, average_error = False):
    """
    Open webcam stream and live draw x,y,z axes and cube.

    Calibrations 1,2,3 are available.
    1 contains 25 images with automated detection, 5 images that were done manually.
    2 contains 10 images with automated detection.
    3 contains 5 images with automated detection.
    """
    calibration_map = {
        1: load_internal_calibrations(1),
        2: load_internal_calibrations(2),
        3: load_internal_calibrations(3)
    }
    calibration_num = initial_calibration
    vid = cv.VideoCapture(0)
    font = cv.FONT_HERSHEY_PLAIN
    error = 0
    counter = 1

    while(True):
        ret, frame = vid.read()
        cv.putText(frame, f"Calibration: {calibration_num}",
                   (100, frame.shape[1] // 2), font, 2, (0, 0, 255), 1)
        mtx, dist = calibration_map[calibration_num]
        if include_error:
            # img, new_err = draw(frame, mtx, dist, include_error=include_error)
            img, new_err = draw(frame, mtx, dist, CHESS_DIMS, STRIDE_LEN, include_error=include_error)
            if new_err is not None:
                if average_error:
                    error += 1/counter * (new_err - error)
                    counter += 1
                    text = f"Reprojection error (Avg): {error:.3f}"
                else:
                    error = new_err
                    text = f"Reprojection error: {error:.3f}"
                cv.putText(img, text,
                           (img.shape[1] // 2, 100), font, 2, (0, 0, 255), 1)

        else:
            img = draw(frame, mtx, dist, CHESS_DIMS, STRIDE_LEN, include_error=include_error)
        cv.imshow('Chessboard with axes and cube.', img)
        pressed_key = cv.waitKey(1) & 0xFF
        if pressed_key == ord('q'):
            break
        elif pressed_key == ord('1'):
            calibration_num = 1
            mtx, dist = calibration_map[calibration_num]
            print(f"Calibration {calibration_num} camera matrix and distortion coefs.")
            print(mtx, dist)
            counter = 1
            error = 0
        elif pressed_key == ord('2'):
            calibration_num = 2
            mtx, dist = calibration_map[calibration_num]
            print(f"Calibration {calibration_num} camera matrix and distortion coefs.")
            print(mtx, dist)
            counter = 1
            error = 0
        elif pressed_key == ord('3'):
            calibration_num = 3
            mtx, dist = calibration_map[calibration_num]
            print(f"Calibration {calibration_num} camera matrix and distortion coefs.")
            print(mtx, dist)
            counter = 1
            error = 0
        elif pressed_key == ord(' '):
            # Time formatting to make files unique
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filename = f"screenshot_{timestamp}.jpg"
            # If edges were detected, write to detected_edges folder
            if ret:
                path = os.path.abspath((os.path.join(DATA_DIR, filename)))
                print(path)
            cv.imwrite(path, frame)

    vid.release()
    cv.destroyAllWindows()


def reprojection_error(objpoints, imgpoints, rvec, tvec, mtx, dist):
    """Calculate reprojection error, return projected points."""
    imgpoints2, _ = cv.projectPoints(objpoints, rvec, tvec, mtx, dist)
    return cv.norm(imgpoints, imgpoints2, cv.NORM_L2) / len(imgpoints2), imgpoints2


if __name__ == "__main__":
    draw_live(include_error=True, average_error=True)
