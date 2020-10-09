import cv2
import time
import picamera
import numpy as np

def FindChessboardCornersInScaledImage(img, chessboard, scale):
    width = int(img.shape[1] * scale)
    height = int(img.shape[0] * scale)
    resized = cv2.resize(img, (width, height), interpolation = cv2.INTER_LINEAR)
    return cv2.findChessboardCorners(resized, chessboard, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)

def GetChessboardROI(img, chessboard):
    SCALE = 4.0
    MARGIN = 0.50
    found, corners = FindChessboardCornersInScaledImage(img, chessboard, 1.0 / SCALE)
    if not found:
        return found, None

    min_x = min([p[0][0] for p in corners]) * SCALE
    max_x = max([p[0][0] for p in corners]) * SCALE
    min_y = min([p[0][1] for p in corners]) * SCALE
    max_y = max([p[0][1] for p in corners]) * SCALE

    x_margin = (max_x - min_x) * MARGIN
    y_margin = (max_y - min_y) * MARGIN

    min_x = min_x - x_margin
    max_x = max_x + x_margin
    min_y = min_y - y_margin
    max_y = max_y + y_margin

    if min_x < 0: min_x = 0
    if min_y < 0: min_y = 0
    if max_x >= img.shape[1]: max_x = img.shape[1] - 1
    if max_y >= img.shape[0]: max_y = img.shape[0] - 1

    return found, (int(min_x), int(min_y), int(max_x - min_x), int(max_y - min_y))

if __name__ == "__main__":
    WIDTH = 1280
    HEIGHT = 720
    CHANNELS = 3
    FPS = 60

    CHESSBOARD = (4, 5)
    CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    SIZE = 0.0245

    OBJ_POINTS = np.zeros((1, CHESSBOARD[0] * CHESSBOARD[1], 3), np.float32)
    OBJ_POINTS[0,:,:2] = np.mgrid[0:CHESSBOARD[0], 0:CHESSBOARD[1]].T.reshape(-1, 2) * SIZE

    print(OBJ_POINTS)

    with picamera.PiCamera() as camera:
        camera.resolution = (WIDTH, HEIGHT)
        camera.framerate = FPS
        time.sleep(2)
        image_points = []
        object_points = []
        i = 1
        while cv2.waitKey(3000) != 27:
            # prepare image
            image = np.empty((WIDTH * HEIGHT * CHANNELS,), dtype=np.uint8)
            # capture color image
            camera.capture(image, 'bgr')
            # convert it to grayscale
            image = cv2.cvtColor(image.reshape((HEIGHT, WIDTH, CHANNELS)), cv2.COLOR_BGR2GRAY)
            # find chessboard in smaller image
            found, roi = GetChessboardROI(image, CHESSBOARD)
            # continue if the chessboard was not found
            if found:
                (x, y, w, h) = roi
                # find chessboard corners
                found, corners = cv2.findChessboardCorners(image[y:y + h, x:x + w], CHESSBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
                if found:
                    # offset corners
                    for corner in corners:
                        corner[0][0] += x
                        corner[0][1] += y
                    # refine corners
                    corners = cv2.cornerSubPix(image, corners, (11, 11),(-1, -1), CRITERIA)
                    # collect object points
                    object_points.append(OBJ_POINTS)
                    # collect image points
                    image_points.append(corners)
                    # store image
                    cv2.imwrite("image_{}.jpg".format(i), image)
                    # draw corners
                    image = cv2.drawChessboardCorners(image, CHESSBOARD, corners, found)
                    i += 1
            # show image
            cv2.imshow("IMAGE", image)
        cv2.destroyAllWindows()
        camera_matrix_guess = np.array([[1250.0, 0, (WIDTH-1)/2.0], [0, 1250.0, (HEIGHT-1)/2.0],
            [0, 0, 1.0]])
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(object_points, image_points, image.shape[::-1], camera_matrix_guess, None, flags = (cv2.CALIB_USE_INTRINSIC_GUESS + cv2.CALIB_FIX_ASPECT_RATIO + cv2.CALIB_ZERO_TANGENT_DIST + cv2.CALIB_FIX_K2 + cv2.CALIB_FIX_K3 + cv2.CALIB_FIX_K4 + cv2.CALIB_FIX_K5 + cv2.CALIB_FIX_K6))

        print("Error: ", ret)
        print("Camera matrix: ")
        print(mtx)
        print("Distortion coeffs: ")
        print(dist)
        print(rvecs)
        print(tvecs)
