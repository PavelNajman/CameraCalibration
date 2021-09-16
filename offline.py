import cv2
import sys
import pathlib
import argparse
import numpy as np

import common

def ParseCommandLineArguments():
    parser = argparse.ArgumentParser(description='Detects chessboard pattern and calibrates the camera.')
    parser.add_argument('-c', '--chessboard-columns', default=5, type=int)
    parser.add_argument('-r', '--chessboard-rows', default=4, type=int)
    parser.add_argument('-s', '--chessboard-field-size', default=0.0245, type=float)
    return parser.parse_args(sys.argv[1:])

def main():
    args = ParseCommandLineArguments()

    OBJ_POINTS = np.zeros((1, args.chessboard_rows * args.chessboard_columns, 3), np.float32)
    OBJ_POINTS[0,:,:2] = np.mgrid[0:args.chessboard_rows, 0:args.chessboard_columns].T.reshape(-1, 2) * args.chessboard_field_size

    CHESSBOARD = (args.chessboard_rows, args.chessboard_columns)

    shape = None
    image_points = []
    object_points = []
    for f in pathlib.Path.cwd().glob("*.jpg"):
        # read image
        image = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        # find chessboard corners
        found, corners = cv2.findChessboardCorners(image, CHESSBOARD, common.FIND_CHESSBOARD_FLAGS)
        if found:
            # refine corners
            corners = cv2.cornerSubPix(image, corners, (11, 11),(-1, -1), common.SUBPIX_CRITERIA)
            # collect object points
            object_points.append(OBJ_POINTS)
            # collect image points
            image_points.append(corners)
            # ensure that all processed images have the same shape
            if not shape:
                shape = image.shape
            elif shape != image.shape:
                print("ERROR: All images should have the same size")
                return -1
    height, width = shape
    camera_matrix_guess = np.array([[1250.0, 0, (width-1)/2.0], [0, 1250.0, (height-1)/2.0], [0, 0, 1.0]])
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(object_points, image_points, image.shape[::-1], camera_matrix_guess, None, flags = common.CALIBRATE_CAMERA_FLAGS)

    print("Error: ", ret)
    print("Camera matrix: ")
    print(mtx)
    print("Distortion coeffs: ")
    print(dist)
    print(rvecs)
    print(tvecs)

    common.DumpResults(width, height, mtx, dist)

    return 0

if __name__ == "__main__":
    sys.exit(main())

