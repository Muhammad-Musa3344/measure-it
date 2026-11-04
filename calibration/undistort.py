import glob
import os
import sys
import cv2
import numpy as np

src, dst = sys.argv[1], sys.argv[2]
os.makedirs(dst, exist_ok=True)
data = np.load("calibration/camera.npz")
K, dist = data["K"], data["dist"]
size = tuple(int(v) for v in data["size"])

paths = sorted(glob.glob(os.path.join(src, "*.jpg")))
done = 0
for path in paths:
    img = cv2.imread(path)
    if (img.shape[1], img.shape[0]) != size:
        print(f"skipped (size {img.shape[1]}x{img.shape[0]}, expected {size[0]}x{size[1]}):", path)
        continue
    cv2.imwrite(os.path.join(dst, os.path.basename(path)), cv2.undistort(img, K, dist))
    done += 1
print(f"undistorted {done} of {len(paths)} images")