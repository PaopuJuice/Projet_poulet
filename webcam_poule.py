import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torchvision import transforms
import os
import pandas as pd
from torchvision.io import decode_image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import cv2

# --- 1. CLASSES BlazeBlock ET BlazeFacePoule ---
class BlazeBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1):
        super(BlazeBlock, self).__init__()
        self.stride = stride
        self.channel_pad = out_channels - in_channels

        # TSize du padding pour garder la même taille spatiale (si stride=1)
        # Pour kernel=5, padding=2. Pour kernel=3, padding=1.
        padding = (kernel_size - 1) // 2

        # 1. Depthwise Convolution (Traite l'espace)
        # groups=in_channels est l'astuce qui rend la convolution "Depthwise"
        self.conv_dw = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, 
                                 stride=stride, padding=padding, groups=in_channels, bias=False)
        self.bn_dw = nn.BatchNorm2d(in_channels)

        # 2. Pointwise Convolution (Mélange les canaux)
        self.conv_pw = nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                                 stride=1, padding=0, bias=False)
        self.bn_pw = nn.BatchNorm2d(out_channels)

        # 3. Gestion de la connexion résiduelle (Shortcut)
        # Si on réduit la taille (stride=2), on doit aussi réduire l'entrée pour l'additionner
        self.pool = nn.MaxPool2d(kernel_size=stride, stride=stride) if stride > 1 else nn.Identity()

    def forward(self, x):
        h = self.conv_dw(x)
        h = self.bn_dw(h)
        h = F.relu(h)
        h = self.conv_pw(h)
        h = self.bn_pw(h)
        
        x_sc = self.pool(x)
        
        # On applique le padding seulement si on augmente le nombre de canaux
        if self.channel_pad > 0:
            # (padding_gauche, droite, haut, bas, canaux_devant, canaux_derriere)
            x_sc = F.pad(x_sc, (0, 0, 0, 0, 0, self.channel_pad))
        elif self.channel_pad < 0:
            # Sécurité : si on voulait réduire les canaux, on tronque (pas idéal mais évite le crash)
            x_sc = x_sc[:, :h.shape[1], :, :]
            
        return F.relu(h + x_sc)


class BlazeFacePoule(nn.Module):
    def __init__(self, num_classes=3):
        super(BlazeFacePoule, self).__init__()
        
        # Conv1 Initiale : 128x128x3 -> 64x64x24
        self.conv1 = nn.Conv2d(3, 24, kernel_size=5, stride=2, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(24)
        
        # --- BACKBONE ---
        self.blocks = nn.Sequential(
            # Stage 1 : reste à 24 canaux (taille 64x64)
            BlazeBlock(24, 24),
            BlazeBlock(24, 24),
            
            # Stage 2 : passe à 48 canaux (taille 64x64 -> 32x32)
            BlazeBlock(24, 48, stride=2), 
            BlazeBlock(48, 48),
            BlazeBlock(48, 48),
            
            # Stage 3 : passe à 96 canaux (taille 32x32 -> 16x16)
            BlazeBlock(48, 96, stride=2),
            BlazeBlock(96, 96),
            BlazeBlock(96, 96),
            
            # Optionnel : un dernier bloc pour plus de profondeur (taille 16x16 -> 8x8)
            BlazeBlock(96, 96, stride=2),
            BlazeBlock(96, 96),
        )
        
        self.final_conv = nn.Conv2d(96, 96, kernel_size=1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1) 
        
        self.fc_class = nn.Linear(96, num_classes)
        self.fc_box = nn.Linear(96, 4) 

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.blocks(x)
        x = self.final_conv(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        
        class_score = self.fc_class(x)
        box_coords = torch.sigmoid(self.fc_box(x))
        return class_score, box_coords

# --- 2. CONFIGURATION ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
labels_map = {0: "Poule", 1: "Predateur", 2: "Rien"}

# Charger le modèle
model = BlazeFacePoule(num_classes=3).to(device)
model.load_state_dict(torch.load("blazeface_poule.pth", map_location=device))
model.eval()

# Transformation identique à l'entraînement
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# --- 3. Lancement de la Webcam ---
cap = cv2.VideoCapture(0) # 0 est l'index de la webcam par défaut

print("Appuyez sur 'q' pour quitter.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Préparation de l'image pour le modèle
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_tensor = transform(img_rgb).unsqueeze(0).to(device) # Ajoute la dimension "Batch"

    # Inférence (Prédiction)
    with torch.no_grad():
        pred_class, pred_box = model(input_tensor)
        
        # Récupérer la classe la plus probable
        probs = F.softmax(pred_class, dim=1)
        conf, class_idx = torch.max(probs, dim=1)
        class_idx = class_idx.item()
        conf = conf.item()

    # Si le modèle est assez sûr de lui et que ce n'est pas "Rien"
    if class_idx != 2 and conf > 0.6:
        # Coordonnées prédites (normalisées 0-1)
        box = pred_box[0].cpu().numpy()
        h_img, w_img, _ = frame.shape
        
        # Conversion YOLO [x_c, y_c, w, h] -> Pixels [x1, y1, x2, y2]
        x_c, y_c, w, h = box[0] * w_img, box[1] * h_img, box[2] * w_img, box[3] * h_img
        x1 = int(x_c - w/2)
        y1 = int(y_c - h/2)
        x2 = int(x_c + w/2)
        y2 = int(y_c + h/2)

        # Dessiner le rectangle et le texte
        color = (0, 255, 0) if class_idx == 0 else (0, 0, 255) # Vert pour Poule, Rouge pour Predateur
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label_text = f"{labels_map[class_idx]} ({conf:.2f})"
        cv2.putText(frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Affichage
    cv2.imshow("Detection Poule en Direct", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
