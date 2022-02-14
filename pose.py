import os
import cv2
import sys
import time
import pickle
import argparse
import numpy as np

import common

def DrawAxis(img, origin, imgpts):
    print(origin, tuple(imgpts[0].ravel()))
    cv2.line(img, origin, [int(x) for x in imgpts[0].ravel()], (255,0,0), 5)
    cv2.line(img, origin, [int(x) for x in imgpts[1].ravel()], (0,255,0), 5)
    cv2.line(img, origin, [int(x) for x in imgpts[2].ravel()], (0,0,255), 5)

def focus(val):
    value = (val << 4) & 0x3ff0
    data1 = (value >> 8) & 0x3f
    data2 = value & 0xf0
    os.system("i2cset -y 6 0x0c %d %d" % (data1,data2))

def adjust_focus(key_code, focal_distance):
    if key_code == 114: # r (reset)
        focal_distance = 0
        print("Focus distance", focal_distance)
        focus(focal_distance)
    elif key_code == 43 or key_code == 171: # +
        focal_distance += 10
        if focal_distance > 1020:
            focal_distance = 1020
        print("Focus distance", focal_distance)
        focus(focal_distance)
    elif key_code == 45 or key_code == 173: # -
        focal_distance -= 10
        if focal_distance < 10:
            focal_distance = 0
        print("Focus distance", focal_distance)
        focus(focal_distance)
    return focal_distance

def parse_command_line_arguments():
    parser = argparse.ArgumentParser(description='Detects chessboard pattern and calibrates the camera.')
    parser.add_argument('-c', '--chessboard-columns', default=5, type=int)
    parser.add_argument('-r', '--chessboard-rows', default=4, type=int)
    parser.add_argument('-s', '--chessboard-field-size', default=0.0245, type=float)
    return parser.parse_args(sys.argv[1:])

if __name__ == "__main__":
    args = parse_command_line_arguments()
    width, height, camera_matrix, distortion_coeffs = common.LoadCalibration()

    cap = cv2.VideoCapture(common.gstreamer_pipeline(width, height, width, height, 30, 0), cv2.CAP_GSTREAMER)
    if cap.isOpened():
        CHANNELS = 3
        CHESSBOARD = (args.chessboard_rows, args.chessboard_columns)
        SCALE = 2.0

        AXIS = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3) * args.chessboard_field_size

        OBJ_POINTS = np.zeros((1, args.chessboard_rows * args.chessboard_columns, 3), np.float32)
        OBJ_POINTS[0,:,:2] = np.mgrid[0:args.chessboard_rows, 0:args.chessboard_columns].T.reshape(-1, 2) * args.chessboard_field_size

        focal_distance = adjust_focus(114, 0)   # reset focus

        cv2.namedWindow('CSI Camera', cv2.WINDOW_AUTOSIZE)

        prev_frame_time = time.time()
        while cv2.getWindowProperty('CSI Camera',0) >= 0:
            # capture image
            ret_val, image = cap.read()
            if not ret_val:
                continue

            # convert it to grayscale
            image = cv2.cvtColor(image.reshape((height, width, CHANNELS)), cv2.COLOR_BGR2GRAY)
            # find chessboard in smaller image
            found, roi, corners = common.GetChessboardROI(image, CHESSBOARD, SCALE)
            # continue if the chessboard was not found
            if found:
                (x, y, w, h) = roi
                # find chessboard corners
                if SCALE != 1.0:
                    found, corners = cv2.findChessboardCorners(image[y:y + h, x:x + w], CHESSBOARD, flags = common.FIND_CHESSBOARD_FLAGS)
                    if found:
                        # offset corners
                        for corner in corners:
                            corner[0][0] += x
                            corner[0][1] += y
                if found:
                    image_points = cv2.cornerSubPix(image, corners, (11, 11),(-1, -1), common.SUBPIX_CRITERIA)
                    ret, rvec, tvec =  cv2.solvePnP(OBJ_POINTS, image_points, camera_matrix, distortion_coeffs)
                    if ret:
                        current_frame_time = time.time()
                        print(rvec.T, tvec.T, 1.0/(current_frame_time - prev_frame_time))
                        prev_frame_time = current_frame_time
                        axis_pts, _ = cv2.projectPoints(AXIS, rvec, tvec, camera_matrix, distortion_coeffs)
                        DrawAxis(image, [int(x) for x in image_points[0].ravel()], axis_pts)
            cv2.imshow("CSI Camera", image)

            key_code = cv2.waitKey(int(1000 / 30)) & 0xff
            if key_code == 27: # ESC (quit)
                break
            else:
                focal_distance = adjust_focus(key_code, focal_distance)

        cap.release()
        cv2.destroyAllWindows()
    else:
        print('Unable to open camera')

