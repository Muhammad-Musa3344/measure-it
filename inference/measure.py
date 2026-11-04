import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from models.common import CFG, ROOT, build_model, get_device

_cache = {}


def load_calibration():
    if "cal" not in _cache:
        data = np.load(ROOT / CFG["camera_file"])
        matrix = dist = None
        for key in data.files:
            arr = data[key]
            if arr.shape == (3, 3):
                matrix = arr
            elif arr.ndim <= 2 and arr.size == 5:
                dist = arr.reshape(-1)
        if matrix is None or dist is None:
            raise RuntimeError(f"camera matrix or distortion not found in {data.files}")
        _cache["cal"] = (matrix, dist)
    return _cache["cal"]


def load_model():
    if "model" not in _cache:
        device = get_device()
        model = build_model()
        model.load_state_dict(torch.load(ROOT / CFG["weights"], map_location=device))
        _cache["model"] = (model.to(device).eval(), device)
    return _cache["model"]


def get_detector():
    if "det" not in _cache:
        params = cv2.aruco.DetectorParameters()
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        _cache["det"] = cv2.aruco.ArucoDetector(dictionary, params)
    return _cache["det"]


def analyze(path, undistort=True, keep=False):
    """Undistort, find marker, segment object, convert to mm. Returns raw numbers, no refusal."""
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(path)
    h, w = img.shape[:2]
    if (w, h) != (CFG["image_w"], CFG["image_h"]):
        raise ValueError(f"image is {w}x{h}, calibration needs {CFG['image_w']}x{CFG['image_h']}")
    if undistort:
        matrix, dist = load_calibration()
        img = cv2.undistort(img, matrix, dist)

    res = {"obj": False, "ref": False, "score": 0.0, "skew": None, "edge": False,
           "width": None, "height": None}
    corners_out, mask_out = None, None

    corners, ids, _ = get_detector().detectMarkers(img)
    if ids is not None and CFG["marker_id"] in ids.flatten():
        k = list(ids.flatten()).index(CFG["marker_id"])
        c = corners[k].reshape(4, 2).astype(np.float32)
        sides = [np.linalg.norm(c[i] - c[(i + 1) % 4]) for i in range(4)]
        res["ref"], res["skew"], corners_out = True, max(sides) / min(sides), c

    model, device = load_model()
    tensor = torch.from_numpy(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float().div(255).to(device)
    with torch.no_grad():
        out = model([tensor])[0]
    if len(out["scores"]) > 0:
        k = int(out["scores"].argmax())
        mask = (out["masks"][k, 0] > 0.5).cpu().numpy()
        if mask.any():
            res["obj"], res["score"], mask_out = True, float(out["scores"][k]), mask
            res["edge"] = bool(mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any())

    if res["obj"] and res["ref"]:
        side = CFG["marker_mm"]
        target = np.float32([[0, 0], [side, 0], [side, side], [0, side]])
        homography = cv2.getPerspectiveTransform(corners_out, target)
        contours, _ = cv2.findContours(mask_out.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contour = max(contours, key=cv2.contourArea)
        pts = cv2.perspectiveTransform(contour.reshape(-1, 1, 2).astype(np.float32), homography)
        (_, (a, b), _) = cv2.minAreaRect(pts)
        res["width"], res["height"] = sorted((float(a), float(b)))
    if keep:
        res["img"], res["mask"], res["corners"] = img, mask_out, corners_out
    return res


def decide(res, score_thresh=None, skew_thresh=None):
    score_thresh = CFG["score_thresh"] if score_thresh is None else score_thresh
    skew_thresh = CFG["skew_thresh"] if skew_thresh is None else skew_thresh
    if not res["obj"]:
        return "OBJECT_NOT_FOUND"
    if not res["ref"]:
        return "REFERENCE_NOT_FOUND"
    if res["score"] < score_thresh:
        return "REFUSED_LOW_CONFIDENCE"
    if res["skew"] > skew_thresh:
        return "REFUSED_TILT"
    if res["edge"]:
        return "REFUSED_OBJECT_AT_EDGE"
    return "OK"


def draw_overlay(res, status):
    img = res["img"].copy()
    if res["mask"] is not None:
        img[res["mask"]] = (0.5 * img[res["mask"]] + 0.5 * np.array([0, 255, 0])).astype(np.uint8)
    if res["corners"] is not None:
        cv2.polylines(img, [res["corners"].astype(np.int32)], True, (0, 255, 255), 4)
    lines = [status, f"confidence {res['score']:.2f}"]
    if status == "OK":
        lines += [f"width {res['width']:.1f} mm", f"height {res['height']:.1f} mm"]
    for i, text in enumerate(lines):
        cv2.putText(img, text, (30, 60 + 60 * i), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 255), 4)
    return img


def measure(path, undistort=True, out_dir=None):
    start = time.perf_counter()
    res = analyze(path, undistort, keep=True)
    status = decide(res)
    out_dir = Path(out_dir) if out_dir else ROOT / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "" if undistort else "_raw"
    overlay = out_dir / f"{Path(path).stem}{suffix}_overlay.jpg"
    cv2.imwrite(str(overlay), draw_overlay(res, status))
    return {"status": status, "width_mm": res["width"], "height_mm": res["height"],
            "confidence": res["score"], "skew": res["skew"], "overlay": str(overlay),
            "time_s": time.perf_counter() - start}


def main():
    parser = argparse.ArgumentParser(description="Image to millimetres")
    parser.add_argument("image")
    parser.add_argument("--no-undistort", action="store_true")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    r = measure(args.image, undistort=not args.no_undistort, out_dir=args.out)
    print(f"status: {r['status']}")
    if r["status"] == "OK":
        print(f"width_mm: {r['width_mm']:.1f}")
        print(f"height_mm: {r['height_mm']:.1f}")
    print(f"confidence: {r['confidence']:.2f}")
    print(f"overlay: {r['overlay']}")
    sys.exit(0 if r["status"] == "OK" else 2)


if __name__ == "__main__":
    main()
