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
    model.load_weights("model_poule_final.pth")
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
        ret, frame = cap.read()
        if not ret: break

        # 1. PRE-PROCESSING (Taille et Couleur)
        # On redimensionne d'abord
        inp = cv2.resize(frame, (256, 256))
        # PUIS on convertit en RGB pour le modèle
        inp_rgb = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB)

        # 2. DETECTION
        # On envoie l'image RGB au modèle
        detections = model.predict_on_image(inp_rgb)

        # 3. DESSIN DES RESULTATS
        # On dessine sur 'inp' (qui est l'image BGR pour OpenCV)
        if len(detections) > 0:
            for i in range(len(detections)):
                # On récupère le score (souvent à l'index 16)
                score = detections[i, 16]

                # --- COMPARAISON SCORE_THRESHOLD ---
                # On ne dessine que si le modèle est sûr à plus de 50%
                if score > 0.7:
                    # Récupération des coordonnées normalisées (0.0 à 1.0)
                    ymin, xmin, ymax, xmax = detections[i, 0:4]

                    # --- SCALING (Conversion en pixels) ---
                    # On multiplie par 256 pour remettre à l'échelle de l'image
                    left = int(xmin * 256)
                    top = int(ymin * 256)
                    right = int(xmax * 256)
                    bottom = int(ymax * 256)

                    # Dessin du rectangle et du texte
                    cv2.rectangle(inp, (left, top), (right, bottom), (0, 255, 0), 2)
                    cv2.putText(inp, f"Poule: {score:.2f}", (left, top - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Affichage
        cv2.imshow("BlazeFace Poule Detection", inp)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
