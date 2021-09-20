import os
import cv2
import sys
import time
import argparse
import numpy as np

import common

def parse_command_line_arguments():
    parser = argparse.ArgumentParser(description='Detects chessboard pattern and calibrates the camera.')
    parser.add_argument('-iw', '--image-width', default=640, type=int)
    parser.add_argument('-ih', '--image-height', default=480, type=int)
    parser.add_argument('-f', '--fps', default=30, type=int)
    parser.add_argument('-c', '--chessboard-columns', default=5, type=int)
    parser.add_argument('-r', '--chessboard-rows', default=4, type=int)
    parser.add_argument('-s', '--chessboard-field-size', default=0.0245, type=float)
    return parser.parse_args(sys.argv[1:])

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

def calibrate_camera(image_points, args):
    OBJ_POINTS = np.zeros((1, args.chessboard_rows * args.chessboard_columns, 3), np.float32)
    OBJ_POINTS[0,:,:2] = np.mgrid[0:args.chessboard_rows, 0:args.chessboard_columns].T.reshape(-1, 2) * args.chessboard_field_size

    camera_matrix_guess = np.array([[1250.0, 0, (args.image_width-1)/2.0], [0, 1250.0, (args.image_height-1)/2.0], [0, 0, 1.0]])
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera([OBJ_POINTS for _ in image_points], image_points, (args.image_width, args.image_height), camera_matrix_guess, None, flags = common.CALIBRATE_CAMERA_FLAGS)

    print("Error: ", ret)
    print("Camera matrix: ")
    print(mtx)
    print("Distortion coeffs: ")
    print(dist)
    print(rvecs)
    print(tvecs)

    common.DumpResults(args.image_width, args.image_height, mtx, dist)

if __name__ == "__main__":
    args = parse_command_line_arguments()

    cap = cv2.VideoCapture(common.gstreamer_pipeline(args.image_width, args.image_height, args.image_width, args.image_height, args.fps, 0), cv2.CAP_GSTREAMER)
    if cap.isOpened():
        window_handle = cv2.namedWindow('CSI Camera', cv2.WINDOW_AUTOSIZE)

        image_points = []

        CHANNELS = 3
        CHESSBOARD = (args.chessboard_rows, args.chessboard_columns)

        CAPTURE_INTERVAL = 3.0
        last_capture_time = time.time()
        time_to_capture = CAPTURE_INTERVAL

        focal_distance = adjust_focus(114, 0)   # reset focus

        while cv2.getWindowProperty('CSI Camera',0) >= 0:
            # capture image
            ret_val, image = cap.read()
            if not ret_val:
                continue

            # convert it to grayscale
            image = cv2.cvtColor(image.reshape((args.image_height, args.image_width, CHANNELS)), cv2.COLOR_BGR2GRAY)

            if time_to_capture == 0:
                last_capture_time = time.time()
                # find chessboard in smaller image
                found, roi, _ = common.GetChessboardROI(image, CHESSBOARD, 2)
                # continue if the chessboard was not found
                if found:
                    (x, y, w, h) = roi
                    # find chessboard corners
                    found, corners = cv2.findChessboardCorners(image[y:y + h, x:x + w], CHESSBOARD, common.FIND_CHESSBOARD_FLAGS)
                    if found:
                        # offset corners
                        for corner in corners:
                            corner[0][0] += x
                            corner[0][1] += y
                        # refine corners
                        corners = cv2.cornerSubPix(image, corners, (11, 11),(-1, -1), common.SUBPIX_CRITERIA)
                        # collect image points
                        image_points.append(corners)
                        # store image
                        cv2.imwrite("image_{}.jpg".format(len(image_points)), image)
                        # draw corners
                        image = cv2.drawChessboardCorners(image, CHESSBOARD, corners, found)

            time_to_capture = max(0, CAPTURE_INTERVAL - (time.time() - last_capture_time))
            cv2.putText(image, "Next capture in: {:.2f} s".format(time_to_capture), (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, "Number of found chessboards: {}".format(len(image_points)), (10, args.image_height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1, cv2.LINE_AA)

            cv2.imshow('CSI Camera',image)

            key_code = cv2.waitKey(int(1000 / args.fps)) & 0xff
            if key_code == 27: # ESC (quit)
                break
            else:
                focal_distance = adjust_focus(key_code, focal_distance)

        cap.release()
        cv2.destroyAllWindows()

        if len(image_points) > 0:
            calibrate_camera(image_points, args)
    else:
        print('Unable to open camera')

