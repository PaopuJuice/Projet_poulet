# -*- coding: utf-8 -*-
"""
Created on Wed Jan 21 15:47:33 2026

@author: micka
"""

import torch
from pathlib import Path
from blazeface import BlazeFace

# Mets ici ton meilleur poids (fine-tuné)
WEIGHTS = Path(__file__).resolve().parents[1] / "runs" / "best_front.pth"  # adapte si besoin

# front model = back_model False (128x128)
model = BlazeFace(back_model=False)
state = torch.load(WEIGHTS, map_location="cpu")
model.load_state_dict(state, strict=True)

total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

print("Weights:", WEIGHTS)
print("Total params:", total)
print("Trainable params:", trainable)

# Optionnel: détail par couche
for name, p in model.named_parameters():
    print(f"{name:40s} {tuple(p.shape)!s:20s} -> {p.numel()}")
