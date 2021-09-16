import cv2
import sys
import time
import picamera
import argparse
import numpy as np

import common

def ParseCommandLineArguments():
    parser = argparse.ArgumentParser(description='Detects chessboard pattern and calibrates the camera.')
    parser.add_argument('-iw', '--image-width', default=640, type=int)
    parser.add_argument('-ih', '--image-height', default=480, type=int)
    parser.add_argument('-f', '--fps', default=30, type=int)
    parser.add_argument('-c', '--chessboard-columns', default=5, type=int)
    parser.add_argument('-r', '--chessboard-rows', default=4, type=int)
    parser.add_argument('-s', '--chessboard-field-size', default=0.0245, type=float)
    return parser.parse_args(sys.argv[1:])

if __name__ == "__main__":
    args = ParseCommandLineArguments()

    OBJ_POINTS = np.zeros((1, args.chessboard_rows * args.chessboard_columns, 3), np.float32)
    OBJ_POINTS[0,:,:2] = np.mgrid[0:args.chessboard_rows, 0:args.chessboard_columns].T.reshape(-1, 2) * args.chessboard_field_size

    with picamera.PiCamera() as camera:
        camera.resolution = (args.image_width, args.image_height)
        camera.framerate = args.fps

        time.sleep(2)

        image_points = []
        object_points = []

        CHANNELS = 3
        CHESSBOARD = (args.chessboard_rows, args.chessboard_columns)

        CAPTURE_INTERVAL = 3.0
        last_capture_time = time.time()
        time_to_capture = CAPTURE_INTERVAL

        num_captured = 0
        while cv2.waitKey(int(1000 / args.fps)) != 27:
            # prepare image
            image = np.empty((args.image_width * args.image_height * CHANNELS,), dtype=np.uint8)
            # capture color image
            camera.capture(image, 'bgr', True)
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
                        # collect object points
                        object_points.append(OBJ_POINTS)
                        # collect image points
                        image_points.append(corners)
                        # store image
                        num_captured += 1
                        cv2.imwrite("image_{}.jpg".format(num_captured), image)
                        # draw corners
                        image = cv2.drawChessboardCorners(image, CHESSBOARD, corners, found)

            time_to_capture = max(0, CAPTURE_INTERVAL - (time.time() - last_capture_time))
            cv2.putText(image, "Next capture in: {:.2f} s".format(time_to_capture), (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, "Number of found chessboards: {}".format(num_captured), (10, args.image_height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1, cv2.LINE_AA)

            # show image
            cv2.imshow("IMAGE", image)
        cv2.destroyAllWindows()

        if num_captured > 0:
            camera_matrix_guess = np.array([[1250.0, 0, (args.image_width-1)/2.0], [0, 1250.0, (args.image_height-1)/2.0], [0, 0, 1.0]])
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(object_points, image_points, image.shape[::-1], camera_matrix_guess, None, flags = common.CALIBRATE_CAMERA_FLAGS)

            print("Error: ", ret)
            print("Camera matrix: ")
            print(mtx)
            print("Distortion coeffs: ")
            print(dist)
            print(rvecs)
            print(tvecs)

            common.DumpResults(args.image_width, args.image_height, mtx, dist)
