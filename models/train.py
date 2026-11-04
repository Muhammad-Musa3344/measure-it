import numpy as np
import torch
from torch.utils.data import DataLoader

from models.common import CFG, ROOT, SegDataset, build_model, get_device, mask_iou, predict_top, seed_all


def collate(batch):
    return tuple(zip(*batch))


def main():
    seed_all(CFG["seed"])
    device = get_device()
    print("device:", device)
    train_set, val_set = SegDataset("train", train=True), SegDataset("val")
    gen = torch.Generator().manual_seed(CFG["seed"])
    loader = DataLoader(train_set, batch_size=CFG["batch_size"], shuffle=True, collate_fn=collate, generator=gen)
    model = build_model().to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=CFG["lr"], momentum=0.9, weight_decay=5e-4)
    best = -1.0
    log = open(ROOT / "models" / "train_log.txt", "w")
    for epoch in range(1, CFG["epochs"] + 1):
        model.train()
        total = 0.0
        for images, targets in loader:
            images = [i.to(device) for i in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss = sum(model(images, targets).values())
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
        model.eval()
        ious = []
        for image, target in val_set:
            mask, _ = predict_top(model, image, device)
            gt = target["masks"].any(0).numpy().astype(bool)
            ious.append(0.0 if mask is None else mask_iou(mask, gt))
        val_iou = float(np.mean(ious))
        line = f"epoch {epoch} train_loss {total / len(loader):.4f} val_iou {val_iou:.4f}"
        print(line)
        log.write(line + "\n")
        log.flush()
        if val_iou > best:
            best = val_iou
            torch.save(model.state_dict(), ROOT / CFG["weights"])
    log.write(f"best val_iou {best:.4f} (checkpoint chosen on val only)\n")
    log.close()


if __name__ == "__main__":
    main()
