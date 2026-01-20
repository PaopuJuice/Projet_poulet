# -*- coding: utf-8 -*-
"""
Created on Mon Jan 19 14:39:15 2026

@author: micka
"""

import cv2
from detector import BlazeFaceDetector
import config

def main():
    det = BlazeFaceDetector(
        back_model=config.BACK_MODEL,
        score_thresh=config.SCORE_THRESH,
    )

    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open webcam index={config.CAMERA_INDEX}")

    print("Press 'q' to quit.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # Prediction on resized input
        inp, dets = det.predict(frame)

        # Remap boxes from inp -> original frame
        H, W = frame.shape[:2]
        h_inp, w_inp = inp.shape[:2]

        for d in dets:
            ymin, xmin, ymax, xmax = d[:4]
            score = float(d[16])

            # coords normalisées 0..1 (voir README) 
            x1_inp = int(xmin * w_inp)
            y1_inp = int(ymin * h_inp)
            x2_inp = int(xmax * w_inp)
            y2_inp = int(ymax * h_inp)

            # scale to original frame
            sx = W / w_inp
            sy = H / h_inp
            x1 = int(x1_inp * sx)
            y1 = int(y1_inp * sy)
            x2 = int(x2_inp * sx)
            y2 = int(y2_inp * sy)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{score:.2f}", (x1, max(0, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Optionnel: resize affichage
        if config.DISPLAY_WIDTH is not None:
            scale = config.DISPLAY_WIDTH / W
            frame_disp = cv2.resize(frame, (config.DISPLAY_WIDTH, int(H * scale)))
        else:
            frame_disp = frame

        cv2.imshow("Webcam BlazeFace", frame_disp)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
