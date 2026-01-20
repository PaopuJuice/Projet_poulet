# -*- coding: utf-8 -*-
"""
Created on Tue Jan 20 11:21:50 2026

@author: micka
"""

import os
import cv2
import random

BASE = "dataset"
SPLIT = "train"  # change en val / test pour vérifier

img_dir = os.path.join(BASE, "images", SPLIT)
lbl_dir = os.path.join(BASE, "labels", SPLIT)

images = [f for f in os.listdir(img_dir) if f.endswith((".jpg", ".png"))]
random.shuffle(images)

for img_name in images[:10]:  # on affiche 10 images au hasard
    img_path = os.path.join(img_dir, img_name)
    lbl_path = os.path.join(lbl_dir, img_name.replace(".jpg", ".txt").replace(".png", ".txt"))

    img = cv2.imread(img_path)
    h, w = img.shape[:2]

    if os.path.exists(lbl_path):
        with open(lbl_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    print("❌ format invalide :", lbl_path)
                    continue

                cls, xc, yc, bw, bh = map(float, parts)
                x1 = int((xc - bw / 2) * w)
                y1 = int((yc - bh / 2) * h)
                x2 = int((xc + bw / 2) * w)
                y2 = int((yc + bh / 2) * h)

                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow("check", img)
    if cv2.waitKey(0) == 27:
        break

cv2.destroyAllWindows()
