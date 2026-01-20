# -*- coding: utf-8 -*-
"""
Created on Tue Jan 20 13:41:58 2026

@author: micka
"""

from pathlib import Path
from chicken_dataset import ChickenDataset, collate_fn
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]   # .../BlazeFace-PyTorch
DATASET_ROOT = REPO_ROOT / "dataset"

ds = ChickenDataset(root=str(DATASET_ROOT), split="train", img_size=None)
print("len:", len(ds))

img, boxes, labels, meta = ds[0]
print(img.shape, boxes.shape, labels.shape)
print(meta)

dl = DataLoader(ds, batch_size=4, shuffle=True, collate_fn=collate_fn)
imgs, boxes_list, labels_list, metas = next(iter(dl))
print("batch imgs:", imgs.shape)
print("first sample boxes:", boxes_list[0].shape)
