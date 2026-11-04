import argparse
import csv
import platform
import statistics
import time

import torch

from inference.measure import analyze, decide, measure
from models.common import ROOT

PHOTOS = ROOT / "measurement" / "photos"
LINES = []


def say(text=""):
    print(text)
    LINES.append(text)


def save(name):
    (ROOT / "measurement" / name).write_text("\n".join(LINES) + "\n", encoding="utf-8")
    LINES.clear()


def load_accuracy():
    with open(PHOTOS / "accuracy.csv", newline="", encoding="utf-8-sig") as f:
        return [(PHOTOS / "accuracy" / r["file"], float(r["width_mm"]), float(r["height_mm"]))
                for r in csv.DictReader(f)]


def load_tilt(width, height):
    return [(p, int(p.stem[4:6]), width, height) for p in sorted((PHOTOS / "tilt").glob("tilt*.jpg"))]


def summary(label, errs, pcts):
    if not errs:
        say(f"{label}: no valid instances")
        return
    say(f"{label}: n={len(errs) // 2} instances, mean abs error {statistics.mean(errs):.2f} mm, "
        f"median {statistics.median(errs):.2f} mm, worst {max(errs):.2f} mm, "
        f"mean % error {statistics.mean(pcts):.2f}")


def cmd_accuracy():
    say("file | true_w true_h (mm) | RAW err_w err_h | UNDISTORTED err_w err_h | status")
    raw_e, und_e, raw_p, und_p = [], [], [], []
    for path, tw, th in load_accuracy():
        und, raw = analyze(path, True), analyze(path, False)
        s_und, s_raw = decide(und), decide(raw)
        if s_und == "OK" and s_raw == "OK":
            e = [raw["width"] - tw, raw["height"] - th, und["width"] - tw, und["height"] - th]
            say(f"{path.name} | {tw:.1f} {th:.1f} | {e[0]:+.2f} {e[1]:+.2f} | {e[2]:+.2f} {e[3]:+.2f} | OK")
            raw_e += [abs(e[0]), abs(e[1])]
            und_e += [abs(e[2]), abs(e[3])]
            raw_p += [abs(e[0]) / tw * 100, abs(e[1]) / th * 100]
            und_p += [abs(e[2]) / tw * 100, abs(e[3]) / th * 100]
        else:
            say(f"{path.name} | {tw:.1f} {th:.1f} | raw={s_raw} undistorted={s_und} (excluded)")
    say()
    summary("RAW (no undistortion)", raw_e, raw_p)
    summary("UNDISTORTED", und_e, und_p)
    save("accuracy_results.txt")


def cmd_tilt(width, height):
    say("file | tilt_deg | marker skew | measured_w measured_h | abs err (mm) | status")
    by_deg = {}
    for path, deg, tw, th in load_tilt(width, height):
        r = analyze(path, True)
        status = decide(r)
        if r["width"] is None:
            say(f"{path.name} | {deg} | - | no measurement | - | {status}")
            continue
        err = (abs(r["width"] - tw) + abs(r["height"] - th)) / 2
        say(f"{path.name} | {deg} | {r['skew']:.3f} | {r['width']:.1f} {r['height']:.1f} | {err:.2f} | {status}")
        by_deg.setdefault(deg, []).append((err, r["skew"], status))
    say()
    for deg in sorted(by_deg):
        rows = by_deg[deg]
        refused = sum(1 for _, _, s in rows if s != "OK")
        say(f"tilt {deg} deg: n={len(rows)}, mean abs error {statistics.mean(x[0] for x in rows):.2f} mm, "
            f"worst {max(x[0] for x in rows):.2f} mm, mean skew {statistics.mean(x[1] for x in rows):.3f}, "
            f"refused {refused}/{len(rows)}")
    save("tilt_results.txt")


def sweep(items, label, key, values):
    say(f"\nThreshold sweep: {label}")
    say("threshold | refused / total | mean abs err of kept (mm) | worst kept (mm)")
    for value in values:
        kept = []
        for r, tw, th in items:
            status = decide(r, **{key: value})
            if status == "OK":
                kept.append((abs(r["width"] - tw) + abs(r["height"] - th)) / 2)
        refused = len(items) - len(kept)
        mean = f"{statistics.mean(kept):.2f}" if kept else "-"
        worst = f"{max(kept):.2f}" if kept else "-"
        say(f"{value} | {refused} / {len(items)} | {mean} | {worst}")


def cmd_refusal(width, height):
    items = [(analyze(p, True), tw, th) for p, tw, th in load_accuracy()]
    items += [(analyze(p, True), tw, th) for p, _, tw, th in load_tilt(width, height)]
    sweep(items, "marker skew (tilt)", "skew_thresh", [1.02, 1.04, 1.06, 1.08, 1.10, 1.15, 1.20, 99])
    sweep(items, "mask confidence", "score_thresh", [0.5, 0.7, 0.8, 0.9, 0.95])
    save("refusal_results.txt")


def cmd_timing(image):
    measure(image)
    times = []
    for _ in range(5):
        start = time.perf_counter()
        measure(image)
        times.append(time.perf_counter() - start)
    say(f"hardware: {platform.processor()} | {platform.platform()} | cuda={torch.cuda.is_available()}")
    say(f"5 runs (s): {[round(t, 2) for t in times]}")
    say(f"median {statistics.median(times):.2f} s, min {min(times):.2f} s, max {max(times):.2f} s")
    save("timing_results.txt")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("accuracy")
    for name in ("tilt", "refusal"):
        p = sub.add_parser(name)
        p.add_argument("--width", type=float, required=True, help="true short side of the tilt object, mm")
        p.add_argument("--height", type=float, required=True, help="true long side of the tilt object, mm")
    p = sub.add_parser("timing")
    p.add_argument("image")
    args = parser.parse_args()
    if args.cmd == "accuracy":
        cmd_accuracy()
    elif args.cmd == "tilt":
        cmd_tilt(args.width, args.height)
    elif args.cmd == "refusal":
        cmd_refusal(args.width, args.height)
    else:
        cmd_timing(args.image)


if __name__ == "__main__":
    main()
