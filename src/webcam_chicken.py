# -*- coding: utf-8 -*-
"""
Created on Tue Jan 20 16:42:25 2026

@author: micka
"""

import sys
import time
from pathlib import Path

from box_fusion import weighted_box_fusion


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2
import torch
from blazeface import BlazeFace

# -----------------------
# Config
# -----------------------
BACK_MODEL = False                    # same as training
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

WEIGHTS = REPO_ROOT / "runs" / ("best_back.pth" if BACK_MODEL else "best_front.pth")
ANCHORS = REPO_ROOT / ("anchorsback.npy" if BACK_MODEL else "anchors.npy")

CONF_THRESH = 0.55                    # start around 0.5-0.6
NMS_IOU = 0.3
TOP_K = 200
MAX_DET = 3

# Temporal filtering (robust "door logic")
REQUIRED_SECONDS = 5.0                # must be detected for 5s
MISS_GRACE_SECONDS = 1.0              # allow brief dropouts


def nms_yxyx(boxes, scores, iou_threshold=0.3):
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


def postprocess(model, raw_boxes, raw_scores):
    """
    raw_boxes: (1, A, 16)
    raw_scores: (1, A, 1)
    returns boxes (N,4) [ymin,xmin,ymax,xmax] normalized + scores (N,)
    """
    scores = torch.sigmoid(raw_scores.squeeze(0).squeeze(-1))  # (A,)
    boxes = model._decode_boxes(raw_boxes.squeeze(0), model.anchors)[:, :4]  # (A,4)

    keep = scores >= CONF_THRESH
    boxes = boxes[keep]
    scores = scores[keep]

    if scores.numel() == 0:
        return boxes, scores

    if scores.numel() > TOP_K:
        scores, idx = torch.topk(scores, k=TOP_K)
        boxes = boxes[idx]
    boxes_np, scores_np = weighted_box_fusion(boxes.cpu().numpy(),scores.cpu().numpy(),iou_thr=0.35)

    boxes = torch.from_numpy(boxes_np).to(boxes.device)
    scores = torch.from_numpy(scores_np).to(scores.device)
    
    MAX_DET=1


    if scores.numel() > MAX_DET:
        scores, idx = torch.topk(scores, k=MAX_DET)
        boxes = boxes[idx]

    return boxes, scores


def main():
    # Load model
    model = BlazeFace(back_model=BACK_MODEL).to(DEVICE)
    model.load_state_dict(torch.load(WEIGHTS, map_location=DEVICE))
    model.load_anchors(str(ANCHORS))
    model.eval()

    size = 256 if BACK_MODEL else 128

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # CAP_DSHOW helps on Windows sometimes
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam (index 0). Try index 1 or check camera permissions.")

    detected_since = None
    last_seen = None
    door_open = False

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            # Resize for inference
            frame_in = cv2.resize(frame, (size, size))
            rgb = cv2.cvtColor(frame_in, cv2.COLOR_BGR2RGB)
            x = torch.from_numpy(rgb).permute(2, 0, 1).float().unsqueeze(0) / 255.0
            x = x.to(DEVICE)

            with torch.no_grad():
                raw_boxes, raw_scores = model(x)
                boxes, scores = postprocess(model, raw_boxes, raw_scores)

            now = time.time()
            has_chicken = scores.numel() > 0

            # Temporal filter
            if has_chicken:
                if detected_since is None:
                    detected_since = now
                last_seen = now
            else:
                # if we lost detection briefly, keep timer for a short grace period
                if last_seen is not None and (now - last_seen) > MISS_GRACE_SECONDS:
                    detected_since = None
                    last_seen = None

            # Decide door state
            if detected_since is not None and (now - detected_since) >= REQUIRED_SECONDS:
                door_open = True
            else:
                door_open = False

            # Draw
            out = frame_in.copy()
            h, w = out.shape[:2]

            for b, s in zip(boxes.cpu().numpy(), scores.cpu().numpy()):
                ymin, xmin, ymax, xmax = b
                x1, y1 = int(xmin * w), int(ymin * h)
                x2, y2 = int(xmax * w), int(ymax * h)
                cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(out, f"{float(s):.2f}", (x1, max(0, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            status = "OPEN" if door_open else "CLOSED"
            cv2.putText(out, f"DOOR: {status}", (8, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0) if door_open else (0, 0, 255), 2)

            if detected_since is not None:
                cv2.putText(out, f"stable: {now - detected_since:.1f}s / {REQUIRED_SECONDS:.1f}s",
                            (8, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            cv2.imshow("Chicken webcam", out)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
