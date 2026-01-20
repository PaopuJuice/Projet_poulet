# -*- coding: utf-8 -*-
"""
Created on Tue Jan 20 09:16:19 2026

@author: micka
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2
import numpy as np
import torch

from blazeface import BlazeFace

BACK_MODEL = False  # same as training
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

WEIGHTS = REPO_ROOT / "runs" / ("best_back.pth" if BACK_MODEL else "best_front.pth")
ANCHORS = REPO_ROOT / ("anchorsback.npy" if BACK_MODEL else "anchors.npy")

# Pick a test image
IMG_PATH = REPO_ROOT / "dataset" / "images" / "test"  # folder
# change this line to a specific file if you want:
# IMG_PATH = REPO_ROOT / "dataset" / "images" / "test" / "xxx.jpg"

CONF_THRESH = 0.5

def nms_yxyx(boxes, scores, iou_threshold=0.3):
    """
    boxes: (N,4) in [ymin,xmin,ymax,xmax] normalized
    scores: (N,)
    returns indices kept
    """
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.long, device=boxes.device)

    y1, x1, y2, x2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)

    order = scores.argsort(descending=True)
    keep = []

    while order.numel() > 0:
        i = order[0].item()
        keep.append(i)
        if order.numel() == 1:
            break

        rest = order[1:]
        yy1 = torch.maximum(y1[i], y1[rest])
        xx1 = torch.maximum(x1[i], x1[rest])
        yy2 = torch.minimum(y2[i], y2[rest])
        xx2 = torch.minimum(x2[i], x2[rest])

        inter = (xx2 - xx1).clamp(min=0) * (yy2 - yy1).clamp(min=0)
        iou = inter / (areas[i] + areas[rest] - inter + 1e-6)

        order = rest[iou <= iou_threshold]

    return torch.tensor(keep, dtype=torch.long, device=boxes.device)

def main():
    model = BlazeFace(back_model=BACK_MODEL).to(DEVICE)
    model.load_state_dict(torch.load(WEIGHTS, map_location=DEVICE))
    model.load_anchors(str(ANCHORS))
    model.eval()

    if IMG_PATH.is_dir():
        imgs = sorted([p for p in IMG_PATH.iterdir() if p.suffix.lower() in [".jpg", ".png", ".jpeg"]])
        if not imgs:
            raise RuntimeError("No images found in test folder")
        img_path = imgs[0]
    else:
        img_path = IMG_PATH

    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        raise RuntimeError(f"Cannot read {img_path}")

    size = 256 if BACK_MODEL else 128
    img_in = cv2.resize(img_bgr, (size, size))
    img_rgb = cv2.cvtColor(img_in, cv2.COLOR_BGR2RGB)
    x = torch.from_numpy(img_rgb).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    x = x.to(DEVICE)

    with torch.no_grad():
        raw_boxes, raw_scores = model(x)  # (1,A,16), (1,A,1)
        scores = torch.sigmoid(raw_scores.squeeze(0).squeeze(-1))  # (A,)

        # Decode boxes (ymin,xmin,ymax,xmax) normalized
        boxes = model._decode_boxes(raw_boxes.squeeze(0), model.anchors)[:, :4]  # (A,4)

        # 1) pre-filter by confidence
        keep = scores >= CONF_THRESH
        boxes = boxes[keep]
        scores = scores[keep]

        # 2) top-k before NMS
        TOP_K = 200
        if scores.numel() > TOP_K:
            scores, idx = torch.topk(scores, k=TOP_K)
            boxes = boxes[idx]

        # 3) NMS
        keep_idx = nms_yxyx(boxes, scores, iou_threshold=0.3)
        boxes = boxes[keep_idx]
        scores = scores[keep_idx]

        # 4) keep only best N
        MAX_DET = 5
        if scores.numel() > MAX_DET:
            scores, idx = torch.topk(scores, k=MAX_DET)
            boxes = boxes[idx]

    # Draw on resized image (outside no_grad is fine)
    out = img_in.copy()
    h, w = out.shape[:2]

    for b, s in zip(boxes.cpu().numpy(), scores.cpu().numpy()):
        ymin, xmin, ymax, xmax = b
        x1, y1 = int(xmin * w), int(ymin * h)
        x2, y2 = int(xmax * w), int(ymax * h)
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(out, f"{float(s):.2f}", (x1, max(0, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("chicken detections", out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
