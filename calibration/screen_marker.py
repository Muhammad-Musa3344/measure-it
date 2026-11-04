import argparse
import ctypes
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
W, H = 1920, 1080


def main():
    p = argparse.ArgumentParser(description="Show an ArUco marker at true size on the screen")
    p.add_argument("--screen_width_mm", type=float, required=True, help="physical width of the screen image, mm")
    p.add_argument("--marker_mm", type=float, default=80.0, help="wanted marker size, mm")
    a = p.parse_args()

    pitch = a.screen_width_mm / W
    px = 6 * round(a.marker_mm / pitch / 6)
    real_mm = px * pitch
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    marker = cv2.aruco.generateImageMarker(d, 0, px)
    canvas = np.full((H, W), 255, np.uint8)
    y0 = (H - px) // 2
    canvas[y0:y0 + px, 150:150 + px] = marker
    cv2.imwrite(str(ROOT / "calibration" / "screen_marker.png"), canvas)

    cfg = json.loads((ROOT / "config.json").read_text(encoding="utf-8-sig"))
    cfg["marker_mm"] = round(real_mm, 2)
    (ROOT / "config.json").write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    print(f"pixel pitch {pitch:.4f} mm, marker {px} px = {real_mm:.2f} mm, saved to config.json")
    print("Showing full screen. Press Esc to close.")

    ctypes.windll.user32.SetProcessDPIAware()
    cv2.namedWindow("marker", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("marker", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow("marker", canvas)
    while cv2.waitKey(100) != 27:
        pass
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
