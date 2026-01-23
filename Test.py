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


#images
TRAIN_CHICK_PATH = "images/images/train"
TEST_CHICK_PATH = "images/images/test"
VAL_CHICK_PATH = "images/images/val"
#labels
TRAIN_LAB_PATH = "labels/labels/train"
TEST_LAB_PATH = "labels/labels/test"
VAL_LAB_PATH = "labels/labels/val"


class PouleDataset(Dataset):
    def __init__(self, img_dir, lab_dir, transform=None):
        self.img_dir = img_dir
        self.lab_dir = lab_dir
        self.transform = transform
        
        # On liste toutes les images présentes dans le dossier
        # (Seulement les fichiers .jpg ou .png)
        self.img_names = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        # 1. Chemin de l'image
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_dir, img_name)
        
        # 2. Chargement et normalisation de l'image
        # On passe en float et on divise par 255 pour avoir des valeurs entre 0 et 1
        image = decode_image(img_path).float() / 255.0
        
        # 3. Valeurs par défaut si le fichier label est vide ou absent
        label = 2  # Classe "Rien" par défaut
        box = torch.zeros(4) # [0, 0, 0, 0]

        # 4. Lecture du fichier d'annotation .txt
        # On cherche le fichier qui porte le même nom que l'image
        label_name = os.path.splitext(img_name)[0] + ".txt"
        label_path = os.path.join(self.lab_dir, label_name)
        
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                lines = f.readlines()
                if len(lines) > 0:
                    # On prend la première ligne (première poule détectée)
                    parts = lines[0].split()
                    if len(parts) >= 5:
                        label = int(parts[0]) # La classe (0, 1 ou 2)
                        box = torch.tensor([float(x) for x in parts[1:5]]) # [x, y, w, h]

        # 5. Application des transformations (Redimensionnement en 128x128)
        if self.transform:
            image = self.transform(image)

        # On retourne l'image et un tuple contenant la classe et la boîte
        return image, (label, box)
        
        
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
        
device = "cuda" if torch.cuda.is_available() else "cpu"
#Parametres :
input_size = 16384  # 128x128 pixels
num_classes = 3 # Poule ou pas poule ou prédateur
learning_rate = 0.001
batch_size = 64
num_epochs = 20  # Reduced for demonstration purposes

labels_map = {
    0: "Poule",
    1: "Prédateur",
    2: "Rien",
}

# 1. Définir les transformations (ex: forcer la taille à 128x128)
transform = transforms.Compose([
    transforms.Resize((128, 128)),
])

# 2. Créer les instances de Dataset
train_data = PouleDataset(TRAIN_CHICK_PATH, TRAIN_LAB_PATH, transform=transform)
val_data   = PouleDataset(VAL_CHICK_PATH, VAL_LAB_PATH, transform=transform)
test_data  = PouleDataset(TEST_CHICK_PATH, TEST_LAB_PATH, transform=transform)

# 3. Créer les DataLoaders (C'est eux que tu utiliseras dans ta boucle d'entraînement)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_data, batch_size=batch_size, shuffle=False)
test_loader  = DataLoader(test_data, batch_size=batch_size, shuffle=False)

################# Entrainement #####################
# 1. Initialisation du modèle
model = BlazeFacePoule(num_classes=3).to(device)

# 2. Optimiseur (Adam est souvent plus efficace pour débuter que SGD)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 3. Fonctions de perte (Loss functions)
criterion_class = nn.CrossEntropyLoss() # Pour les labels (0, 1, 2)
criterion_box = nn.MSELoss()            # Pour les coordonnées [x, y, w, h]


print(f"Début de l'entraînement sur {device}...")

for epoch in range(num_epochs):
    model.train() # Mode entraînement
    running_loss = 0.0
    
    for batch_idx, (images, targets) in enumerate(train_loader):
        # Transférer les données sur GPU/CPU
        images = images.to(device)
        
        # Séparer les cibles (targets)
        # On suppose ici que ton Dataset renvoie (classe, [x, y, w, h])
        true_labels = targets[0].to(device) # Classes (N,)
        true_boxes = targets[1].to(device)  # Coordonnées (N, 4)

        # --- 1. Forward pass ---
        pred_class, pred_box = model(images)

        # --- 2. Calcul de la perte ---
        loss_cls = criterion_class(pred_class, true_labels)
        loss_box = criterion_box(pred_box, true_boxes)
        
        # On pondère la perte de la boîte (souvent x2 ou x5 pour équilibrer)
        total_loss = loss_cls + (2.0 * loss_box)

        # --- 3. Backward pass (L'apprentissage) ---
        optimizer.zero_grad()  # Reset les gradients
        total_loss.backward()  # Calcule l'erreur pour chaque neurone
        optimizer.step()       # Ajuste les neurones

        running_loss += total_loss.item()

    # --- Validation (Optionnel mais recommandé) ---
    model.eval() # Mode évaluation
    val_loss = 0.0
    with torch.no_grad(): # Pas besoin de calculer les gradients ici
        for images, targets in val_loader:
            images = images.to(device)
            p_cls, p_box = model(images)
            loss = criterion_class(p_cls, targets[0].to(device)) + \
                   criterion_box(p_box, targets[1].to(device))
            val_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {running_loss/len(train_loader):.4f} - Val Loss: {val_loss/len(val_loader):.4f}")

torch.save(model.state_dict(), "blazeface_poule.pth")

print("Entraînement terminé !")












