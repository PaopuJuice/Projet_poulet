# -*- coding: utf-8 -*-
"""
Created on Mon Jan 19 14:22:35 2026

@author: micka
"""

import cv2
import torch
from blazeface import BlazeFace

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Back model: expects 256x256
    model = BlazeFace(back_model=True).to(device)
    model.load_weights("blazefaceback.pth")
    model.load_anchors("anchors.npy")
    model.eval()

    cap = cv2.VideoCapture(0)  # 0 = webcam par défaut
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam (VideoCapture(0) failed).")

    # Optionnel: réduire la résolution capture pour CPU
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("Press 'q' to quit.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # Resize for BlazeFace
        inp = cv2.resize(frame, (256, 256))

        # Inference (postprocess inside)
        detections = model.predict_on_image(inp)

        # Draw on the 256x256 frame (simple)
        for det in detections:
            ymin, xmin, ymax, xmax = det[:4]
            score = float(det[16])

            if score < 0.75:
                continue

            x1 = int(xmin * 256)
            y1 = int(ymin * 256)
            x2 = int(xmax * 256)
            y2 = int(ymax * 256)

            cv2.rectangle(inp, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                inp,
                f"{score:.2f}",
                (x1, max(0, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )

        cv2.imshow("BlazeFace webcam (256x256)", inp)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
