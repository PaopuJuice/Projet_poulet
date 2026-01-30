# train.py
import torch
import math
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os

# 1. Définir l'architecture du modèle
class BlazeBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv_dw = nn.Conv2d(in_channels, in_channels, kernel_size=5, stride=stride, padding=2, groups=in_channels)
        self.conv_pw = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.conv_dw(x)
        x = self.conv_pw(x)
        x = self.pool(x)
        return self.activation(x)

class PouleDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 24, kernel_size=5, stride=2, padding=2)
        self.block1 = BlazeBlock(24, 24)
        self.block2 = BlazeBlock(24, 48)
        self.block3 = BlazeBlock(48, 48)
        self.block4 = BlazeBlock(48, 96)
        self.block5 = BlazeBlock(96, 96)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(96, 1)
        self.sigmoid = nn.Sigmoid()

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

# 2. Préparer les données
def prepare_data():
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Charger les datasets depuis votre structure de dossiers
    train_dataset = datasets.ImageFolder(root='./data/train', transform=transform)
    val_dataset = datasets.ImageFolder(root='./data/val', transform=transform)

    return train_dataset, val_dataset

# 3. Fonction d'entraînement
def train_model():
    # Préparer les données
    train_dataset, val_dataset = prepare_data()

    # Créer les DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Initialiser le modèle
    device = torch.device("cpu")
    model = PouleDetector().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Entraînement
    num_epochs = 40
    train_losses = []
    val_losses = []
    best_val=math.inf
    best_epoch = 0
    for epoch in range(num_epochs):
        # Entraînement
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.float().to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels.unsqueeze(1))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.float().to(device)
                outputs = model(images)
                loss = criterion(outputs, labels.unsqueeze(1))
                val_loss += loss.item()
        val_loss = val_loss / len(val_loader)
        val_losses.append(val_loss)
        if val_loss < best_val :
            best_val = val_loss
            best_epoch = epoch + 1
            torch.save(model.state_dict(), 'poule_detector_best.pth')
        else :
            torch.save(model.state_dict(), 'poule_detector_last.pth')

        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f} ===> best_val : {best_val:.4f} ; Best epoch : {best_epoch:.1f}')

    

    # Sauvegarder les courbes
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.savefig('loss_curve.png')
    plt.close()

if __name__ == "__main__":
    train_model()

