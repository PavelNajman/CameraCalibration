import cv2
import sys
import time
import pickle
import picamera
import argparse
import numpy as np

import common

def DrawAxis(img, origin, imgpts):
    cv2.line(img, origin, tuple(imgpts[0].ravel()), (255,0,0), 5)
    cv2.line(img, origin, tuple(imgpts[1].ravel()), (0,255,0), 5)
    cv2.line(img, origin, tuple(imgpts[2].ravel()), (0,0,255), 5)

def parse_args():
    parser = argparse.ArgumentParser(description='Detects chessboard pattern and calibrates the camera.')
    parser.add_argument('-c', '--chessboard-columns', default=5, type=int)
    parser.add_argument('-r', '--chessboard-rows', default=4, type=int)
    parser.add_argument('-s', '--chessboard-field-size', default=0.0245, type=float)
    return parser.parse_args(sys.argv[1:])

if __name__ == "__main__":
    args = parse_args()
    width, height, camera_matrix, distortion_coeffs = common.LoadCalibration()

    AXIS = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3) * args.chessboard_field_size

    OBJ_POINTS = np.zeros((1, args.chessboard_rows * args.chessboard_columns, 3), np.float32)
    OBJ_POINTS[0,:,:2] = np.mgrid[0:args.chessboard_rows, 0:args.chessboard_columns].T.reshape(-1, 2) * args.chessboard_field_size

    FPS = 30
    CHANNELS = 3
    CHESSBOARD = (args.chessboard_rows, args.chessboard_columns)
    SCALE = 1.0

    with picamera.PiCamera() as camera:
        camera.resolution = (width, height)
        camera.framerate = FPS
        time.sleep(2)
        prev_frame_time = time.time()
        while cv2.waitKey(int(1000.0 / FPS)) != 27:
            # prepare image
            image = np.empty((width * height * CHANNELS,), dtype=np.uint8)
            # capture color image
            camera.capture(image, 'bgr', True)
            # convert it to grayscale
            image = cv2.cvtColor(image.reshape((height, width, CHANNELS)), cv2.COLOR_BGR2GRAY)
            # find chessboard in smaller image
            found, roi, corners = common.GetChessboardROI(image, CHESSBOARD, SCALE)
            # continue if the chessboard was not found
            if found:
                (x, y, w, h) = roi
                # find chessboard corners
                if SCALE != 1.0:
                    found, corners = cv2.findChessboardCorners(image[y:y + h, x:x + w], CHESSBOARD, common.FIND_CHESSBOARD_FLAGS)
                    # offset corners
                    for corner in corners:
                        corner[0][0] += x
                        corner[0][1] += y
                if found:
                    # refine corners
                    #image_points = cv2.cornerSubPix(image, corners, (11, 11),(-1, -1), common.SUBPIX_CRITERIA)
                    image_points = corners
                    ret, rvec, tvec =  cv2.solvePnP(OBJ_POINTS, image_points, camera_matrix, distortion_coeffs)
                    if ret:
                        current_frame_time = time.time()
                        print(rvec.T, tvec.T, 1.0/(current_frame_time - prev_frame_time))
                        prev_frame_time = current_frame_time
                        axis_pts, _ = cv2.projectPoints(AXIS, rvec, tvec, camera_matrix, distortion_coeffs)
                        DrawAxis(image, tuple(image_points[0].ravel()), axis_pts)
            cv2.imshow("IMAGE", image)
