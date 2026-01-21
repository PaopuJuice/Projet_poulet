# -*- coding: utf-8 -*-
"""
Created on Wed Jan 21 16:22:16 2026

@author: micka
"""

import numpy as np

def iou_yxyx(a, b):
    ay1, ax1, ay2, ax2 = a
    by1, bx1, by2, bx2 = b

    iy1 = max(ay1, by1)
    ix1 = max(ax1, bx1)
    iy2 = min(ay2, by2)
    ix2 = min(ax2, bx2)

    ih = max(0.0, iy2 - iy1)
    iw = max(0.0, ix2 - ix1)
    inter = ih * iw

    area_a = max(0.0, ay2 - ay1) * max(0.0, ax2 - ax1)
    area_b = max(0.0, by2 - by1) * max(0.0, bx2 - bx1)
    union = area_a + area_b - inter + 1e-9

    return inter / union


def weighted_box_fusion(boxes, scores, iou_thr=0.35):
    """
    boxes: (N,4) ymin,xmin,ymax,xmax normalized
    scores: (N,)
    """
    if len(boxes) == 0:
        return boxes, scores

    boxes = np.asarray(boxes)
    scores = np.asarray(scores)

    order = scores.argsort()[::-1]
    boxes = boxes[order]
    scores = scores[order]

    clusters = []

    for i in range(len(boxes)):
        found = False
        for c in clusters:
            if iou_yxyx(boxes[i], boxes[c[0]]) >= iou_thr:
                c.append(i)
                found = True
                break
        if not found:
            clusters.append([i])

    fused_boxes = []
    fused_scores = []

    for c in clusters:
        b = boxes[c]
        s = scores[c]
        w = s / (s.sum() + 1e-9)
        fused_boxes.append((b * w[:, None]).sum(axis=0))
        fused_scores.append(s.max())

    return np.array(fused_boxes), np.array(fused_scores)
