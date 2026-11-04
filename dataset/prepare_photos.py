import argparse
import csv
import shutil
import tempfile
import zipfile
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
W, H = 1200, 1600
CARD_W, CARD_H = 53.98, 85.60


def collect(src, tmp):
    src = Path(src)
    zips = [src] if src.suffix.lower() == ".zip" else list(src.glob("*.zip"))
    for z in zips:
        with zipfile.ZipFile(z) as zf:
            zf.extractall(tmp)
    found = list(Path(tmp).rglob("*"))
    if src.is_dir():
        found += list(src.glob("*"))
    files = [p for p in found if p.suffix.lower() in (".jpg", ".jpeg")]
    return sorted(files, key=lambda p: p.stat().st_mtime)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("src", help="folder or zip with new photos")
    parser.add_argument("kind", choices=["train", "accuracy", "tilt"])
    parser.add_argument("--deg", type=int, default=0, help="tilt angle, tilt kind only")
    args = parser.parse_args()

    detector = cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), cv2.aruco.DetectorParameters())
    raw = ROOT / "dataset" / "raw"
    acc = ROOT / "measurement" / "photos" / "accuracy"
    tilt = ROOT / "measurement" / "photos" / "tilt"
    for d in (raw, acc, tilt):
        d.mkdir(parents=True, exist_ok=True)

    good, bad = 0, 0
    with tempfile.TemporaryDirectory() as tmp:
        for path in collect(args.src, tmp):
            img = cv2.imread(str(path))
            if img is None or img.shape[:2] != (H, W):
                print(f"skip {path.name}: not {W}x{H}")
                bad += 1
                continue
            _, ids, _ = detector.detectMarkers(img)
            if ids is None or 0 not in ids.flatten():
                print(f"skip {path.name}: marker id 0 not found")
                bad += 1
                continue
            if args.kind == "train":
                n = len(list(raw.glob("img_*.jpg"))) + 1
                if n > 100:
                    print("already 100 training images")
                    break
                shutil.copyfile(path, raw / f"img_{n:03d}.jpg")
            elif args.kind == "accuracy":
                n = len(list(acc.glob("B*.jpg"))) + 1
                name = f"B{n:02d}.jpg"
                shutil.copyfile(path, acc / name)
                csv_path = ROOT / "measurement" / "photos" / "accuracy.csv"
                new = not csv_path.exists()
                with open(csv_path, "a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    if new:
                        writer.writerow(["file", "width_mm", "height_mm"])
                    writer.writerow([name, CARD_W, CARD_H])
            else:
                n = len(list(tilt.glob(f"tilt{args.deg:02d}_*.jpg"))) + 1
                shutil.copyfile(path, tilt / f"tilt{args.deg:02d}_{n:02d}.jpg")
            good += 1
    print(f"accepted {good}, skipped {bad}")


if __name__ == "__main__":
    main()
