import glob
import sys
import cv2
import numpy as np

PATTERN = (9, 7)       # inner corners (columns, rows)
SQUARE_MM = 25.0       # REPLACE with your calliper measurement
MIN_IMAGES = 20

paths = sorted(glob.glob("calibration/images/*.jpg"))
if not paths:
    sys.exit("No images found in calibration/images/. Add your checkerboard photos first.")

objp = np.zeros((PATTERN[0] * PATTERN[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:PATTERN[0], 0:PATTERN[1]].T.reshape(-1, 2) * SQUARE_MM
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)

obj_pts, img_pts, used = [], [], []
size = None
for path in paths:
    gray = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)
    if size is None:
        size = gray.shape[::-1]
    elif gray.shape[::-1] != size:
        print("skipped (different image size):", path)
        continue
    found, corners = cv2.findChessboardCorners(gray, PATTERN)
    if not found:
        print("skipped (no board found):", path)
        continue
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    obj_pts.append(objp)
    img_pts.append(corners)
    used.append(path)

if len(used) < MIN_IMAGES:
    sys.exit(f"Only {len(used)} usable images; need at least {MIN_IMAGES}. Retake the skipped ones.")

rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(obj_pts, img_pts, size, None, None)

errors = []
for o, i, r, t in zip(obj_pts, img_pts, rvecs, tvecs):
    proj, _ = cv2.projectPoints(o, r, t, K, dist)
    diff = i.reshape(-1, 2).astype(np.float64) - proj.reshape(-1, 2)
    errors.append(float(np.sqrt(np.mean(np.sum(diff ** 2, axis=1)))))

print(f"image size: {size[0]}x{size[1]}")
print(f"images used: {len(used)} of {len(paths)}")
print(f"RMS reprojection error: {rms:.3f} px")
print(f"mean per-image error: {np.mean(errors):.3f} px, worst: {np.max(errors):.3f} px")
print("camera matrix:\n", K)
print("distortion:", dist.ravel())
np.savez("calibration/camera.npz", K=K, dist=dist, size=np.array(size))