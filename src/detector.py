# -*- coding: utf-8 -*-
"""
Created on Mon Jan 19 14:37:18 2026

@author: micka
"""
import sys
import os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)

import cv2
import torch
from blazeface import BlazeFace

class BlazeFaceDetector:
    def __init__(self, back_model: bool, score_thresh: float):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.back_model = back_model
        self.score_thresh = float(score_thresh)
        self.input_size = 256 if back_model else 128

        self.model = BlazeFace(back_model=back_model).to(self.device)

        if back_model:
            self.model.load_weights("../blazefaceback.pth")
            self.model.load_anchors("../anchorsback.npy")
        else:
            self.model.load_weights("../blazeface.pth")
            self.model.load_anchors("../anchors.npy")

        self.model.eval()

    def predict(self, frame_bgr):
        # BlazeFace.py prend np.ndarray (H,W,3) et fait tout le reste. 
        inp = cv2.resize(frame_bgr, (self.input_size, self.input_size))
        dets = self.model.predict_on_image(inp)

        # Filtre score
        if dets.shape[0] == 0:
            return inp, dets

        keep = dets[:, 16] >= self.score_thresh
        return inp, dets[keep]
