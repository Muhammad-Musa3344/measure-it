import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    img_dir = ROOT / "dataset" / "images"
    check = ROOT / "dataset" / "label_check"
    check.mkdir(exist_ok=True)
    detector = cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), cv2.aruco.DetectorParameters())
    images, anns = [], []
    for path in sorted(img_dir.glob("img_*.jpg")):
        img = cv2.imread(str(path))
        h, w = img.shape[:2]
        corners, ids, _ = detector.detectMarkers(img)
        if ids is None or 0 not in ids.flatten():
            print(f"skip {path.name}: marker not found")
            continue
        poly = corners[list(ids.flatten()).index(0)].reshape(-1, 2).astype(np.int32)
        gray = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (9, 9), 0)
        _, bright = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        cnts, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        screen = max(cnts, key=cv2.contourArea)
        region = np.zeros((h, w), np.uint8)
        cv2.fillPoly(region, [cv2.convexHull(screen)], 255)
        obj = cv2.bitwise_and(region, cv2.bitwise_not(bright))
        zone = np.zeros((h, w), np.uint8)
        cv2.fillConvexPoly(zone, poly, 255)
        zone = cv2.dilate(zone, np.ones((41, 41), np.uint8))
        obj[zone > 0] = 0
        kernel = np.ones((7, 7), np.uint8)
        obj = cv2.morphologyEx(cv2.morphologyEx(obj, cv2.MORPH_OPEN, kernel), cv2.MORPH_CLOSE, kernel)
        cnts, _ = cv2.findContours(obj, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts or cv2.contourArea(max(cnts, key=cv2.contourArea)) < 2000:
            print(f"skip {path.name}: no object found")
            continue
        c = max(cnts, key=cv2.contourArea)
        c = cv2.approxPolyDP(c, 2.0, True).reshape(-1, 2)
        x, y, bw, bh = cv2.boundingRect(c)
        image_id = len(images) + 1
        images.append({"id": image_id, "file_name": path.name, "width": w, "height": h})
        anns.append({"id": image_id, "image_id": image_id, "category_id": 1,
                     "segmentation": [c.flatten().tolist()], "bbox": [x, y, bw, bh],
                     "area": float(cv2.contourArea(c)), "iscrowd": 0})
        shown = img.copy()
        cv2.polylines(shown, [c.astype(np.int32)], True, (0, 255, 0), 5)
        cv2.imwrite(str(check / path.name), shown)
    coco = {"images": images, "annotations": anns,
            "categories": [{"id": 1, "name": "object"}]}
    (ROOT / "dataset" / "annotations.json").write_text(json.dumps(coco), encoding="utf-8")
    print(f"labelled {len(images)} images; look at dataset\\label_check before training")


if __name__ == "__main__":
    main()
