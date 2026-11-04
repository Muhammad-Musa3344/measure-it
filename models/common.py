import json
import random
from pathlib import Path

import numpy as np
import torch
from pycocotools.coco import COCO
from PIL import Image
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

ROOT = Path(__file__).resolve().parents[1]
CFG = json.loads((ROOT / "config.json").read_text(encoding="utf-8-sig"))


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model():
    """COCO-pretrained Mask R-CNN with new 2-class heads (background, object)."""
    model = maskrcnn_resnet50_fpn(
        weights="DEFAULT", min_size=CFG["min_size"], max_size=CFG["max_size"]
    )
    in_box = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_box, 2)
    in_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_mask, 256, 2)
    return model


def split_stems(split):
    text = (ROOT / "dataset" / "splits" / f"{split}.txt").read_text(encoding="utf-8-sig")
    return [Path(line.strip()).stem for line in text.splitlines() if line.strip()]


def mask_iou(a, b):
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 0.0


class SegDataset(torch.utils.data.Dataset):
    def __init__(self, split, train=False):
        self.coco = COCO(str(ROOT / "dataset" / "annotations.json"))
        by_stem = {Path(i["file_name"]).stem: i["id"] for i in self.coco.dataset["images"]}
        self.items = []
        for stem in split_stems(split):
            if stem in by_stem and self.coco.getAnnIds(imgIds=by_stem[stem]):
                self.items.append((stem, by_stem[stem]))
        print(f"{split}: {len(self.items)} labelled images")
        self.train = train

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        stem, image_id = self.items[i]
        img = Image.open(ROOT / "dataset" / "images" / f"{stem}.jpg").convert("RGB")
        anns = self.coco.loadAnns(self.coco.getAnnIds(imgIds=image_id))
        masks = np.stack([self.coco.annToMask(a) for a in anns]).astype(np.uint8)
        boxes = []
        for m in masks:
            ys, xs = np.where(m)
            boxes.append([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1])
        image = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
        masks = torch.from_numpy(masks)
        boxes = torch.tensor(boxes, dtype=torch.float32)
        if self.train and random.random() < 0.5:
            width = image.shape[-1]
            image = image.flip(-1)
            masks = masks.flip(-1)
            boxes = torch.stack([width - boxes[:, 2], boxes[:, 1], width - boxes[:, 0], boxes[:, 3]], dim=1)
        target = {"boxes": boxes, "labels": torch.ones(len(boxes), dtype=torch.int64), "masks": masks}
        return image, target


@torch.no_grad()
def predict_top(model, image, device):
    out = model([image.to(device)])[0]
    if len(out["scores"]) == 0:
        return None, 0.0
    k = int(out["scores"].argmax())
    return (out["masks"][k, 0] > 0.5).cpu().numpy(), float(out["scores"][k])
