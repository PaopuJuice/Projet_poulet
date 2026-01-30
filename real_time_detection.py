# real_time_detection.py
import cv2
import torch
from torchvision import transforms
from PIL import Image
import numpy as np

# 1. Définir l'architecture du modèle
class BlazeBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv_dw = torch.nn.Conv2d(in_channels, in_channels, kernel_size=5, stride=stride, padding=2, groups=in_channels)
        self.conv_pw = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.activation = torch.nn.ReLU()

    def forward(self, x):
        x = self.conv_dw(x)
        x = self.conv_pw(x)
        x = self.pool(x)
        return self.activation(x)

class PouleDetector(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 24, kernel_size=5, stride=2, padding=2)
        self.block1 = BlazeBlock(24, 24)
        self.block2 = BlazeBlock(24, 48)
        self.block3 = BlazeBlock(48, 48)
        self.block4 = BlazeBlock(48, 96)
        self.block5 = BlazeBlock(96, 96)
        self.global_pool = torch.nn.AdaptiveAvgPool2d(1)
        self.fc = torch.nn.Linear(96, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return self.sigmoid(x)

def main():
    # 2. Charger le modèle entraîné
    device = torch.device("cpu")
    model = PouleDetector().to(device)
    model.load_state_dict(torch.load('poule_detector_best.pth'))
    model.eval()

    # 3. Définir les transformations
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 4. Initialiser la webcam
    cap = cv2.VideoCapture(0)

    # Vérifier que la webcam est ouverte
    if not cap.isOpened():
        print("Erreur: Impossible d'ouvrir la webcam")
        return

    # Créer une seule fenêtre
    cv2.namedWindow('Détection de poules', cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Erreur: Impossible de lire le flux vidéo")
            break

        try:
            # Convertir l'image OpenCV en PIL Image
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img)

            # Appliquer les transformations
            img_tensor = transform(img_pil).unsqueeze(0).to(device)

            # Faire la prédiction
            with torch.no_grad():
                output = model(img_tensor)
                pred = output.item()
                confidence = pred if pred > 0.5 else 1 - pred

            # Afficher le résultat
            h, w, _ = frame.shape

            status = "POULE DETECTEE" if pred > 0.4 else "AUCUNE POULE"
            color = (0, 200, 0) if pred > 0.4 else (0, 0, 200)

            # Overlay translucide
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, h), color, -1)
            alpha = 0.15 if pred > 0.4 else 0.08
            frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

            # Texte
            cv2.putText(
                frame,
                f"{status}",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.4,
                color,
                3,
            )

            cv2.putText(
                frame,
                f"Confidence: {pred*100:.1f} %",
                (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                2,
            )


            # Afficher dans la même fenêtre
            cv2.imshow('Détection de poules', frame)

        except Exception as e:
            print(f"Erreur lors du traitement: {e}")
            break

        # Quitter si 'q' est pressé
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Libérer les ressources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

