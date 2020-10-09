import cv2
import time
import picamera
import numpy as np

import common

if __name__ == "__main__":
    OBJ_POINTS = np.zeros((1, common.CHESSBOARD[0] * common.CHESSBOARD[1], 3), np.float32)
    OBJ_POINTS[0,:,:2] = np.mgrid[0:common.CHESSBOARD[0], 0:common.CHESSBOARD[1]].T.reshape(-1, 2) * common.SIZE

    print(OBJ_POINTS)

    with picamera.PiCamera() as camera:
        camera.resolution = (common.WIDTH, common.HEIGHT)
        camera.framerate = common.FPS
        time.sleep(2)
        image_points = []
        object_points = []
        i = 1
        while cv2.waitKey(3000) != 27:
            # prepare image
            image = np.empty((common.WIDTH * common.HEIGHT * common.CHANNELS,), dtype=np.uint8)
            # capture color image
            camera.capture(image, 'bgr')
            # convert it to grayscale
            image = cv2.cvtColor(image.reshape((common.HEIGHT, common.WIDTH, common.CHANNELS)), cv2.COLOR_BGR2GRAY)
            # find chessboard in smaller image
            found, roi = common.GetChessboardROI(image, common.CHESSBOARD)
            # continue if the chessboard was not found
            if found:
                (x, y, w, h) = roi
                # find chessboard corners
                found, corners = cv2.findChessboardCorners(image[y:y + h, x:x + w], common.CHESSBOARD, common.FIND_CHESSBOARD_FLAGS)
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
                    cv2.imwrite("image_{}.jpg".format(i), image)
                    # draw corners
                    image = cv2.drawChessboardCorners(image, common.CHESSBOARD, corners, found)
                    i += 1
            # show image
            cv2.imshow("IMAGE", image)
        cv2.destroyAllWindows()
        camera_matrix_guess = np.array([[1250.0, 0, (common.WIDTH-1)/2.0], [0, 1250.0, (common.HEIGHT-1)/2.0],
            [0, 0, 1.0]])
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(object_points, image_points, image.shape[::-1], camera_matrix_guess, None, flags = common.CALIBRATE_CAMERA_FLAGS)

        print("Error: ", ret)
        print("Camera matrix: ")
        print(mtx)
        print("Distortion coeffs: ")
        print(dist)
        print(rvecs)
        print(tvecs)
