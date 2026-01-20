# -*- coding: utf-8 -*-
"""
Created on Tue Jan 20 11:50:00 2026

@author: micka
"""

# src/chicken_dataset.py
import os
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def _list_images(folder: Path) -> List[Path]:
    files = []
    for ext in IMG_EXTS:
        files.extend(folder.glob(f"*{ext}"))
    return sorted(files)


def _read_yolo_txt(txt_path: Path) -> np.ndarray:
    """
    YOLO label format per line:
      cls xc yc w h
    with values normalized to [0,1].

    Returns: (N, 5) float32 array. If file missing/empty -> (0,5)
    """
    if not txt_path.exists():
        return np.zeros((0, 5), dtype=np.float32)

    lines = [ln.strip() for ln in txt_path.read_text(encoding="utf-8", errors="ignore").splitlines()]
    lines = [ln for ln in lines if ln]

    if not lines:
        return np.zeros((0, 5), dtype=np.float32)

    rows = []
    for ln in lines:
        parts = ln.split()
        if len(parts) != 5:
            # skip malformed line
            continue
        rows.append([float(p) for p in parts])
    if not rows:
        return np.zeros((0, 5), dtype=np.float32)

    return np.asarray(rows, dtype=np.float32)


def _yolo_to_xyxy_abs(labels: np.ndarray, img_w: int, img_h: int) -> np.ndarray:
    """
    labels: (N,5) [cls, xc, yc, w, h] normalized
    returns: (N,5) [cls, x1, y1, x2, y2] absolute pixels (float32)
    """
    if labels.shape[0] == 0:
        return np.zeros((0, 5), dtype=np.float32)

    cls = labels[:, 0:1]
    xc = labels[:, 1] * img_w
    yc = labels[:, 2] * img_h
    bw = labels[:, 3] * img_w
    bh = labels[:, 4] * img_h

    x1 = xc - bw / 2.0
    y1 = yc - bh / 2.0
    x2 = xc + bw / 2.0
    y2 = yc + bh / 2.0

    # clip to image bounds
    x1 = np.clip(x1, 0, img_w - 1)
    y1 = np.clip(y1, 0, img_h - 1)
    x2 = np.clip(x2, 0, img_w - 1)
    y2 = np.clip(y2, 0, img_h - 1)

    out = np.concatenate([cls, x1[:, None], y1[:, None], x2[:, None], y2[:, None]], axis=1).astype(np.float32)
    return out


class ChickenDataset(Dataset):
    """
    Loads images + YOLO labels.
    Returns:
      image: torch.float32 (3,H,W) in [0,1]
      boxes: torch.float32 (N,4) absolute XYXY in pixels
      labels: torch.int64 (N,)  (class index, here should be 0)
      meta: dict (paths, original size)
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        img_size: Optional[Tuple[int, int]] = None,  # (W,H) if you want resize here
    ):
        self.root = Path(root)
        self.split = split

        self.img_dir = self.root / "images" / split
        self.lbl_dir = self.root / "labels" / split

        if not self.img_dir.exists():
            raise FileNotFoundError(f"Image folder not found: {self.img_dir}")
        if not self.lbl_dir.exists():
            raise FileNotFoundError(f"Label folder not found: {self.lbl_dir}")

        self.images = _list_images(self.img_dir)
        if len(self.images) == 0:
            raise RuntimeError(f"No images found in: {self.img_dir}")

        self.img_size = img_size  # optional resize

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        img_path = self.images[idx]
        txt_path = self.lbl_dir / (img_path.stem + ".txt")

        # Read image (BGR)
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            raise RuntimeError(f"Failed to read image: {img_path}")

        img_h, img_w = img_bgr.shape[:2]

        # Read labels
        yolo = _read_yolo_txt(txt_path)  # (N,5)
        xyxy = _yolo_to_xyxy_abs(yolo, img_w, img_h)  # (N,5)

        # Keep only class 0 (safety)
        if xyxy.shape[0] > 0:
            keep = (xyxy[:, 0] == 0)
            xyxy = xyxy[keep]

        boxes = xyxy[:, 1:5] if xyxy.shape[0] > 0 else np.zeros((0, 4), dtype=np.float32)
        labels = np.zeros((boxes.shape[0],), dtype=np.int64)  # single class => 0

        # Optional resize (keep it simple for now: resize image and scale boxes)
        if self.img_size is not None:
            new_w, new_h = self.img_size
            sx = new_w / img_w
            sy = new_h / img_h
            img_bgr = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            if boxes.shape[0] > 0:
                boxes[:, [0, 2]] *= sx
                boxes[:, [1, 3]] *= sy
            img_w, img_h = new_w, new_h

        # Convert to RGB + tensor CHW float
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0

        boxes_t = torch.from_numpy(boxes).float()
        labels_t = torch.from_numpy(labels).long()

        meta = {
            "img_path": str(img_path),
            "txt_path": str(txt_path),
            "orig_size": (img_h, img_w),
        }

        return img, boxes_t, labels_t, meta


def collate_fn(batch):
    """
    For detection tasks, keep variable number of boxes per image.
    """
    imgs, boxes, labels, metas = zip(*batch)
    imgs = torch.stack(imgs, dim=0)
    return imgs, list(boxes), list(labels), list(metas)
