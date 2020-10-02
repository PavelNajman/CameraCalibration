import time
import numpy as np
import cv2
import pathlib


if __name__ == "__main__":
    WIDTH = 1280
    HEIGHT = 720

    CHECKERBOARD = (4, 5)
    CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    SIZE = 0.025

    OBJ_POINTS = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    OBJ_POINTS[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2) * SIZE

    image_points = []
    object_points = []
    for f in pathlib.Path.cwd().glob("*.jpg"):
        # read image
        image = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        # find chessboard corners
        found, corners = cv2.findChessboardCorners(image, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
        if found:
            # refine corners
            corners = cv2.cornerSubPix(image, corners, (11, 11),(-1, -1), CRITERIA)
            # collect object points
            object_points.append(OBJ_POINTS)
            # collect image points
            image_points.append(corners)
            # draw corners
            image = cv2.drawChessboardCorners(image, CHECKERBOARD, corners, found)
    camera_matrix_guess = np.array([[1250.0, 0, (WIDTH-1)/2.0], [0, 1250.0, (HEIGHT-1)/2.0], [0, 0, 1.0]])
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(object_points, image_points, image.shape[::-1], camera_matrix_guess, None, flags = (cv2.CALIB_USE_INTRINSIC_GUESS + cv2.CALIB_FIX_ASPECT_RATIO + cv2.CALIB_ZERO_TANGENT_DIST + cv2.CALIB_FIX_K2 + cv2.CALIB_FIX_K3 + cv2.CALIB_FIX_K4 + cv2.CALIB_FIX_K5 + cv2.CALIB_FIX_K6))

    print("Error: ", ret)        
    print("Camera matrix: ")
    print(mtx)
    print("Distortion coeffs: ")
    print(dist)
    print(rvecs)
    print(tvecs)
