import cv2
import numpy as np

# ArUco marker, 4x4 dictionary, ID 0
aruco = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
marker = cv2.aruco.generateImageMarker(aruco, 0, 600)
marker = cv2.copyMakeBorder(marker, 100, 100, 100, 100, cv2.BORDER_CONSTANT, value=255)
cv2.imwrite("calibration/marker_id0.png", marker)

# Checkerboard: 10x8 squares gives 9x7 inner corners
sq = 100
board = np.zeros((8 * sq, 10 * sq), np.uint8)
for r in range(8):
    for c in range(10):
        if (r + c) % 2 == 0:
            board[r * sq:(r + 1) * sq, c * sq:(c + 1) * sq] = 255
board = cv2.copyMakeBorder(board, 50, 50, 50, 50, cv2.BORDER_CONSTANT, value=255)
cv2.imwrite("calibration/checkerboard_9x7.png", board)