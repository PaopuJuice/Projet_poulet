import torch
import torch.optim as optim
import cv2
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from blazeface import BlazeFace

# --- 1. Gestion du Dataset ---
class PouleDataset(Dataset):
    def __init__(self, img_dir, label_dir, transform=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        # On liste les images (on suppose que le nom est identique au .txt)
        self.img_names = [f for f in os.listdir(img_dir) if f.endswith('.jpg')]

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        # Charger l'image
        img_path = os.path.join(self.img_dir, self.img_names[idx])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (256, 256)) # Taille pour BlazeFace back_model
        
        # Normalisation comme dans BlazeFace (_preprocess)
        image = (image.astype(np.float32) / 127.5) - 1.0
        image = torch.from_numpy(image).permute(2, 0, 1) # HWC vers CHW

        # Charger le label correspondant
        label_path = os.path.join(self.label_dir, self.img_names[idx].replace('.jpg', '.txt'))
        
        # Lecture simplifiée du premier objet trouvé dans le .txt (Format YOLO: class x y w h)
        targets = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    data = list(map(float, line.split()))
                    targets.append(torch.tensor(data[1:])) # On garde x, y, w, h
        
        # Si vide, on met des zéros (attention, BlazeFace nécessite une gestion complexe des targets)
        if len(targets) == 0:
            targets = [torch.zeros(4)]
            
        return image, targets[0] # Retourne l'image et la première boîte

# --- 2. Configuration et Chemins ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Utilisation du périphérique : {device}")

img_path = "images/images/train"
label_path = "labels/labels/train"

dataset = PouleDataset(img_path, label_path)
train_loader = DataLoader(dataset, batch_size=8, shuffle=True)

# --- 3. Initialisation Modèle ---
model = BlazeFace(back_model=True).to(device)
model.train() 

optimizer = optim.Adam(model.parameters(), lr=1e-4)

# --- 4. Boucle d'entraînement ---
for epoch in range(10):
    total_loss = 0
    for images, targets in train_loader:
        images = images.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        
        # Passage dans le modèle
        # BlazeFace renvoie [regresseur, classifieur]
        predictions = model(images) 
        
        # Note : La fonction compute_blazeface_loss n'existe pas nativement.
        # Pour l'exemple, on simule une MSE simple sur la régression
        # Dans un vrai cas, il faut implémenter une Loss SSD (Multibox Loss)
        loss = torch.nn.functional.mse_loss(predictions[0][:, 0, :4], targets)
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
    print(f"Epoch {epoch} - Loss: {total_loss/len(train_loader):.4f}")
    
    if epoch % 10 == 0:
        torch.save(model.state_dict(), f"test_poule_epoch_{epoch}.pth")

torch.save(model.state_dict(), "model_poule_final.pth")
