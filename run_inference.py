# -*- coding: utf-8 -*-
"""
Created on Mon Jan 19 14:12:31 2026

@author: micka
"""

import cv2
import torch
from blazeface import BlazeFace

# ----------------------------
# Setup
# ----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

model = BlazeFace(back_model=True).to(device)
model.load_weights("blazefaceback.pth")
model.load_anchors("anchors.npy")   # OBLIGATOIRE
model.eval()

print("Model loaded")

# ----------------------------
# Load + preprocess image
# ----------------------------
img = cv2.imread("1face.png")
if img is None:
    raise RuntimeError("Could not read test.jpg")

img_resized = cv2.resize(img, (256, 256))

# ----------------------------
# Inference (ALL postprocess inside)
# ----------------------------
detections = model.predict_on_image(img_resized)

print("Detections shape:", detections.shape)

# ----------------------------
# Draw detections
# ----------------------------
h, w = img_resized.shape[:2]

for det in detections:
    ymin, xmin, ymax, xmax = det[:4]
    score = det[16]

    # coords are normalized [0,1]
    x1 = int(xmin * w)
    y1 = int(ymin * h)
    x2 = int(xmax * w)
    y2 = int(ymax * h)

    cv2.rectangle(img_resized, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(
        img_resized,
        f"{score:.2f}",
        (x1, max(0, y1 - 5)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        1,
    )

# ----------------------------
# Show
# ----------------------------
cv2.imshow("BlazeFace result", img_resized)
cv2.waitKey(0)
cv2.destroyAllWindows()
