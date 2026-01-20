# -*- coding: utf-8 -*-
"""
Created on Tue Jan 20 15:08:47 2026

@author: micka
"""

import sys
from pathlib import Path

# Add repo root to PYTHONPATH (fix Spyder / Windows)
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from chicken_dataset import ChickenDataset, collate_fn
from blazeface import BlazeFace, jaccard  # jaccard() is IoU for [ymin,xmin,ymax,xmax]

# -----------------------
# Config
# -----------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = REPO_ROOT / "dataset"

BACK_MODEL = False   # False => 128x128 (front) | True => 256x256 (back)
BATCH_SIZE = 16
EPOCHS = 25
LR = 1e-3
POS_IOU_THRESH = 0.35         # with 1 object/image, keep it modest
NEG_TO_POS_RATIO = 3          # hard-negative mining ratio
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

WEIGHTS_PATH = REPO_ROOT / ("blazefaceback.pth" if BACK_MODEL else "blazeface.pth")
ANCHORS_PATH = REPO_ROOT / ("anchorsback.npy" if BACK_MODEL else "anchors.npy")

OUT_DIR = REPO_ROOT / "runs"
OUT_DIR.mkdir(exist_ok=True, parents=True)
BEST_PATH = OUT_DIR / ("best_back.pth" if BACK_MODEL else "best_front.pth")

# -----------------------
# Helpers
# -----------------------
def anchors_to_yminxminymaxxmax(anchors_xywh: torch.Tensor) -> torch.Tensor:
    # anchors: (A,4) [x_center, y_center, w, h] normalized
    x = anchors_xywh[:, 0]
    y = anchors_xywh[:, 1]
    w = anchors_xywh[:, 2]
    h = anchors_xywh[:, 3]
    ymin = y - h / 2.0
    xmin = x - w / 2.0
    ymax = y + h / 2.0
    xmax = x + w / 2.0
    return torch.stack([ymin, xmin, ymax, xmax], dim=1)

def xyxy_abs_to_yminxminymaxxmax_norm(boxes_xyxy_abs: torch.Tensor, img_w: int, img_h: int) -> torch.Tensor:
    # boxes: (N,4) [x1,y1,x2,y2] absolute pixels
    x1, y1, x2, y2 = boxes_xyxy_abs[:, 0], boxes_xyxy_abs[:, 1], boxes_xyxy_abs[:, 2], boxes_xyxy_abs[:, 3]
    ymin = y1 / img_h
    xmin = x1 / img_w
    ymax = y2 / img_h
    xmax = x2 / img_w
    return torch.stack([ymin, xmin, ymax, xmax], dim=1).clamp(0, 1)

def encode_bbox_to_raw_offsets(gt_yminxminymaxxmax: torch.Tensor, anchors_xywh: torch.Tensor,
                              x_scale: float, y_scale: float, w_scale: float, h_scale: float) -> torch.Tensor:
    # gt box => center/size
    ymin, xmin, ymax, xmax = gt_yminxminymaxxmax.unbind(dim=-1)
    gt_w = (xmax - xmin).clamp(min=1e-6)
    gt_h = (ymax - ymin).clamp(min=1e-6)
    gt_xc = (xmin + xmax) / 2.0
    gt_yc = (ymin + ymax) / 2.0

    axc, ayc, aw, ah = anchors_xywh.unbind(dim=-1)

    # Inverse of BlazeFace decode:
    # x_center = raw0/x_scale * aw + axc  => raw0 = (x_center-axc)/aw * x_scale
    # w       = raw2/w_scale * aw         => raw2 = (w/aw) * w_scale
    raw0 = (gt_xc - axc) / aw * x_scale
    raw1 = (gt_yc - ayc) / ah * y_scale
    raw2 = gt_w / aw * w_scale
    raw3 = gt_h / ah * h_scale

    return torch.stack([raw0, raw1, raw2, raw3], dim=-1)

@torch.no_grad()
def match_anchors_single_box(gt_box_norm_yminxminymaxxmax: torch.Tensor,
                             anchors_xywh: torch.Tensor,
                             anchors_yminxminymaxxmax: torch.Tensor,
                             pos_iou_thresh: float) -> torch.Tensor:
    # gt: (4,), anchors: (A,4)
    ious = jaccard(gt_box_norm_yminxminymaxxmax.unsqueeze(0), anchors_yminxminymaxxmax).squeeze(0)  # (A,)
    best_idx = torch.argmax(ious).item()

    pos = ious >= pos_iou_thresh
    pos[best_idx] = True  # always keep best
    return pos, ious

def hard_negative_mining(loss_neg: torch.Tensor, pos_mask: torch.Tensor, neg_to_pos_ratio: int) -> torch.Tensor:
    # loss_neg: (A,) BCE loss per anchor
    # pos_mask: (A,)
    num_pos = int(pos_mask.sum().item())
    if num_pos == 0:
        # fallback: keep small number of negatives
        k = min(50, loss_neg.numel())
    else:
        k = min(loss_neg.numel(), neg_to_pos_ratio * num_pos)

    # exclude positives
    loss_neg = loss_neg.clone()
    loss_neg[pos_mask] = -1.0
    _, idx = torch.topk(loss_neg, k=k, largest=True)
    neg_mask = torch.zeros_like(pos_mask)
    neg_mask[idx] = True
    return neg_mask

# -----------------------
# Train
# -----------------------
def main():
    # dataset: if BACK_MODEL => you MUST have 256x256 images, otherwise 128x128
    img_size = (256, 256) if BACK_MODEL else (128, 128)

    train_ds = ChickenDataset(root=str(DATASET_ROOT), split="train", img_size=img_size)
    val_ds   = ChickenDataset(root=str(DATASET_ROOT), split="val",   img_size=img_size)

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    model = BlazeFace(back_model=BACK_MODEL).to(DEVICE)
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
    model.load_anchors(str(ANCHORS_PATH))
    model.train()

    # Freeze backbone at start (recommended with small dataset)
    for name, p in model.named_parameters():
        p.requires_grad = ("classifier" in name) or ("regressor" in name)

    optim = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=LR)

    anchors_xywh = model.anchors.detach()  # (896,4) normalized
    anchors_yxyx = anchors_to_yminxminymaxxmax(anchors_xywh)

    best_val = 1e18

    for epoch in range(1, EPOCHS + 1):
        # ---- train ----
        model.train()
        total = 0.0
        n_batches = 0

        for imgs, boxes_list, _, _ in train_dl:
            imgs = imgs.to(DEVICE)

            # forward => [raw_boxes (B,896,16), raw_scores (B,896,1)]
            raw_boxes, raw_scores = model(imgs)
            raw_scores = raw_scores.squeeze(-1)  # (B,896)

            B, A = raw_scores.shape
            img_h, img_w = imgs.shape[-2], imgs.shape[-1]

            loss_cls_sum = 0.0
            loss_box_sum = 0.0

            for b in range(B):
                boxes_abs = boxes_list[b].to(DEVICE)  # (N,4) xyxy abs
                if boxes_abs.numel() == 0:
                    # No object: train all anchors as negatives
                    target_scores = torch.zeros((A,), device=DEVICE)
                    loss_per_anchor = F.binary_cross_entropy_with_logits(raw_scores[b], target_scores, reduction="none")
                    loss_cls_sum = loss_cls_sum + loss_per_anchor.mean()
                    continue

                # You said "poule seul" => 1 box, but we keep it generic
                # Take the largest box if multiple
                areas = (boxes_abs[:, 2] - boxes_abs[:, 0]).clamp(min=0) * (boxes_abs[:, 3] - boxes_abs[:, 1]).clamp(min=0)
                gt_idx = torch.argmax(areas).item()
                gt_norm_yxyx = xyxy_abs_to_yminxminymaxxmax_norm(boxes_abs[gt_idx:gt_idx+1], img_w, img_h).squeeze(0)

                pos_mask, _ = match_anchors_single_box(gt_norm_yxyx, anchors_xywh, anchors_yxyx, POS_IOU_THRESH)

                target_scores = torch.zeros((A,), device=DEVICE)
                target_scores[pos_mask] = 1.0

                # classification loss (per-anchor)
                loss_per_anchor = F.binary_cross_entropy_with_logits(raw_scores[b], target_scores, reduction="none")

                # hard negative mining
                neg_mask = hard_negative_mining(loss_per_anchor.detach(), pos_mask, NEG_TO_POS_RATIO)

                cls_mask = pos_mask | neg_mask
                loss_cls = loss_per_anchor[cls_mask].mean()

                # bbox regression loss only on positives, coords 0..3
                if pos_mask.any():
                    # encode GT to raw offsets for positive anchors
                    gt_for_pos = gt_norm_yxyx.unsqueeze(0).repeat(int(pos_mask.sum()), 1)
                    anc_for_pos = anchors_xywh[pos_mask]
                    tgt_offsets = encode_bbox_to_raw_offsets(
                        gt_for_pos, anc_for_pos,
                        model.x_scale, model.y_scale, model.w_scale, model.h_scale
                    )  # (P,4)

                    pred_offsets = raw_boxes[b, pos_mask, 0:4]  # (P,4)
                    loss_box = F.smooth_l1_loss(pred_offsets, tgt_offsets, reduction="mean")
                else:
                    loss_box = torch.tensor(0.0, device=DEVICE)

                loss_cls_sum = loss_cls_sum + loss_cls
                loss_box_sum = loss_box_sum + loss_box

            loss = (loss_cls_sum / B) + (loss_box_sum / B)

            optim.zero_grad()
            loss.backward()
            optim.step()

            total += loss.item()
            n_batches += 1

        train_loss = total / max(1, n_batches)

        # ---- val ----
        model.eval()
        with torch.no_grad():
            total = 0.0
            n_batches = 0
            for imgs, boxes_list, _, _ in val_dl:
                imgs = imgs.to(DEVICE)
                raw_boxes, raw_scores = model(imgs)
                raw_scores = raw_scores.squeeze(-1)

                B, A = raw_scores.shape
                img_h, img_w = imgs.shape[-2], imgs.shape[-1]

                loss_cls_sum = 0.0
                loss_box_sum = 0.0

                for b in range(B):
                    boxes_abs = boxes_list[b].to(DEVICE)
                    if boxes_abs.numel() == 0:
                        target_scores = torch.zeros((A,), device=DEVICE)
                        loss_per_anchor = F.binary_cross_entropy_with_logits(raw_scores[b], target_scores, reduction="none")
                        loss_cls_sum += loss_per_anchor.mean()
                        continue

                    areas = (boxes_abs[:, 2] - boxes_abs[:, 0]).clamp(min=0) * (boxes_abs[:, 3] - boxes_abs[:, 1]).clamp(min=0)
                    gt_idx = torch.argmax(areas).item()
                    gt_norm_yxyx = xyxy_abs_to_yminxminymaxxmax_norm(boxes_abs[gt_idx:gt_idx+1], img_w, img_h).squeeze(0)

                    pos_mask, _ = match_anchors_single_box(gt_norm_yxyx, anchors_xywh, anchors_yxyx, POS_IOU_THRESH)
                    target_scores = torch.zeros((A,), device=DEVICE)
                    target_scores[pos_mask] = 1.0

                    loss_per_anchor = F.binary_cross_entropy_with_logits(raw_scores[b], target_scores, reduction="none")
                    neg_mask = hard_negative_mining(loss_per_anchor, pos_mask, NEG_TO_POS_RATIO)
                    cls_mask = pos_mask | neg_mask
                    loss_cls = loss_per_anchor[cls_mask].mean()

                    if pos_mask.any():
                        gt_for_pos = gt_norm_yxyx.unsqueeze(0).repeat(int(pos_mask.sum()), 1)
                        anc_for_pos = anchors_xywh[pos_mask]
                        tgt_offsets = encode_bbox_to_raw_offsets(
                            gt_for_pos, anc_for_pos,
                            model.x_scale, model.y_scale, model.w_scale, model.h_scale
                        )
                        pred_offsets = raw_boxes[b, pos_mask, 0:4]
                        loss_box = F.smooth_l1_loss(pred_offsets, tgt_offsets, reduction="mean")
                    else:
                        loss_box = torch.tensor(0.0, device=DEVICE)

                    loss_cls_sum += loss_cls
                    loss_box_sum += loss_box

                loss = (loss_cls_sum / B) + (loss_box_sum / B)
                total += loss.item()
                n_batches += 1

            val_loss = total / max(1, n_batches)

        print(f"[Epoch {epoch:02d}] train={train_loss:.4f} | val={val_loss:.4f}")

        # Save best
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), BEST_PATH)
            print(f"  -> saved best to {BEST_PATH}")

    print("Done. Best val:", best_val, "| saved:", BEST_PATH)

if __name__ == "__main__":
    main()
