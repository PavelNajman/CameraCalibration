import cv2
import pickle

WIDTH = 1280
HEIGHT = 720
CHANNELS = 3
FPS = 60

CHESSBOARD = (4, 5)
SIZE = 0.0245
SUBPIX_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
FIND_CHESSBOARD_FLAGS = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
CALIBRATE_CAMERA_FLAGS = cv2.CALIB_USE_INTRINSIC_GUESS + cv2.CALIB_FIX_ASPECT_RATIO + cv2.CALIB_ZERO_TANGENT_DIST + cv2.CALIB_FIX_K2 + cv2.CALIB_FIX_K3 + cv2.CALIB_FIX_K4 + cv2.CALIB_FIX_K5 + cv2.CALIB_FIX_K6

def FindChessboardCornersInScaledImage(img, chessboard, scale):
    width = int(img.shape[1] * scale)
    height = int(img.shape[0] * scale)
    resized = cv2.resize(img, (width, height), interpolation = cv2.INTER_LINEAR)
    return cv2.findChessboardCorners(resized, chessboard, FIND_CHESSBOARD_FLAGS)

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

def DumpResults(mtx, dist):
    obj_pickle = {}
    obj_pickle["camera_matrix"] = mtx
    obj_pickle["distortion_coeffs"] = dist
    pickle_file = open("camera_calibration.p", "wb")
    pickle.dump(obj_pickle, pickle_file)
    pickle_file.close()
