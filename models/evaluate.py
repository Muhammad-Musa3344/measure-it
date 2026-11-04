import hashlib

import numpy as np
import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from models.common import CFG, ROOT, SegDataset, build_model, get_device, mask_iou


def main():
    device = get_device()
    weights = ROOT / CFG["weights"]
    model = build_model()
    model.load_state_dict(torch.load(weights, map_location=device))
    model.to(device).eval()
    test_set = SegDataset("test")
    metric = MeanAveragePrecision(iou_type="segm")
    ious, tp, fp, fn = [], 0, 0, 0
    for image, target in test_set:
        with torch.no_grad():
            out = model([image.to(device)])[0]
        pred_masks = (out["masks"][:, 0] > 0.5).cpu()
        scores, labels = out["scores"].cpu(), out["labels"].cpu()
        gt_masks = target["masks"].bool()
        metric.update(
            [{"masks": pred_masks, "scores": scores, "labels": labels}],
            [{"masks": gt_masks, "labels": target["labels"]}],
        )
        gt_union = gt_masks.any(0).numpy()
        ious.append(mask_iou(pred_masks[0].numpy(), gt_union) if len(scores) else 0.0)
        keep = [k for k in range(len(scores)) if scores[k] >= 0.5]
        matched = set()
        for k in keep:
            cand = [(mask_iou(pred_masks[k].numpy(), gt_masks[j].numpy()), j)
                    for j in range(len(gt_masks)) if j not in matched]
            best = max(cand, default=(0.0, -1))
            if best[0] >= 0.5:
                tp += 1
                matched.add(best[1])
            else:
                fp += 1
        fn += len(gt_masks) - len(matched)
    result = metric.compute()
    sha = hashlib.sha256(weights.read_bytes()).hexdigest()
    lines = [
        f"test images: {len(test_set)}",
        f"mask IoU (mean, top prediction): {np.mean(ious):.4f}",
        f"mAP@0.5: {float(result['map_50']):.4f}",
        f"mAP@0.5:0.95: {float(result['map']):.4f}",
        f"precision (score>=0.5, IoU>=0.5): {tp / max(tp + fp, 1):.4f}",
        f"recall (score>=0.5, IoU>=0.5): {tp / max(tp + fn, 1):.4f}",
        f"weights sha256: {sha}",
    ]
    print("\n".join(lines))
    (ROOT / "measurement" / "test_metrics.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
