import cv2
import time
import pickle
import picamera
import numpy as np

import common

def DrawAxis(img, origin, imgpts):
    cv2.line(img, origin, tuple(imgpts[0].ravel()), (255,0,0), 5)
    cv2.line(img, origin, tuple(imgpts[1].ravel()), (0,255,0), 5)
    cv2.line(img, origin, tuple(imgpts[2].ravel()), (0,0,255), 5)

if __name__ == "__main__":
    camera_matrix, distortion_coeffs = common.LoadCalibration()
    
    AXIS = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3) * common.SIZE

    OBJ_POINTS = np.zeros((1, common.CHESSBOARD[0] * common.CHESSBOARD[1], 3), np.float32)
    OBJ_POINTS[0,:,:2] = np.mgrid[0:common.CHESSBOARD[0], 0:common.CHESSBOARD[1]].T.reshape(-1, 2) * common.SIZE

    with picamera.PiCamera() as camera:
        camera.resolution = (common.WIDTH, common.HEIGHT)
        camera.framerate = common.FPS
        time.sleep(2)
        while cv2.waitKey(int(1000.0 / common.FPS)) != 27:
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
                    image_points = cv2.cornerSubPix(image, corners, (11, 11),(-1, -1), common.SUBPIX_CRITERIA)
                    ret, rvec, tvec =  cv2.solvePnP(OBJ_POINTS, image_points, camera_matrix, distortion_coeffs)
                    if ret:
                        print(rvec.T, tvec.T)
                        axis_pts, _ = cv2.projectPoints(AXIS, rvec, tvec, camera_matrix, distortion_coeffs)
                        DrawAxis(image, tuple(image_points[0].ravel()), axis_pts)
            cv2.imshow("IMAGE", image)
