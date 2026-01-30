# evaluate.py
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

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

def print_dataset_statistics(root="data"):
    splits = ["train", "val", "test"]

    # ---- comptage global
    split_counts = {}
    class_counts = {}
    total_images = 0

    for split in splits:
        split_path = os.path.join(root, split)
        split_total = 0
        class_counts[split] = {}

        for cls in os.listdir(split_path):
            cls_path = os.path.join(split_path, cls)
            if not os.path.isdir(cls_path):
                continue

            n = len([
                f for f in os.listdir(cls_path)
                if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))
            ])
            class_counts[split][cls] = n
            split_total += n

        split_counts[split] = split_total
        total_images += split_total

    # ---- affichage global
    print("\n📊 DATASET SPLIT DISTRIBUTION\n")
    print("GLOBAL:")
    for split in splits:
        pct = 100 * split_counts[split] / total_images
        print(f"  {split:<5}: {split_counts[split]:4d} images ({pct:5.1f} %)")
    print(f"  TOTAL : {total_images:4d} images\n")

    # ---- détail par split
    print("DETAIL PAR SPLIT:\n")
    for split in splits:
        print(f"{split.upper()}:")
        split_total = split_counts[split]
        for cls, n in class_counts[split].items():
            pct = 100 * n / split_total if split_total > 0 else 0
            print(f"  {cls:<12}: {n:4d} ({pct:5.1f} %)")
        print()
        
        
def evaluate_model():
    print_dataset_statistics("data")
    # 2. Charger le modèle entraîné
    device = torch.device("cpu")
    model = PouleDetector().to(device)
    model.load_state_dict(torch.load('poule_detector.pth'))
    model.eval()

    # 3. Préparer les données de test
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_dataset = datasets.ImageFolder(root='./data/test', transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # 4. Évaluer le modèle
    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = (outputs > 0.5).float()
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(outputs.view(-1).cpu().numpy())

    # 5. Calculer les métriques
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)

    print('Résultats de l\'évaluation:')
    print(f'Précision: {precision:.4f}')
    print(f'Rappel: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')
    print('Matrice de confusion:')
    print(cm)

    # 6. Visualiser la matrice de confusion
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Non-poule', 'Poule'],
                yticklabels=['Non-poule', 'Poule'])
    plt.title('Matrice de confusion')
    plt.ylabel('Vraies classes')
    plt.xlabel('Prédictions')
    plt.savefig('confusion_matrix.png')
    plt.close()

    # 7. Visualiser les probabilités (corrigé)
    plt.figure(figsize=(10, 6))
    plt.hist(all_probs, bins=20, color=['skyblue'], edgecolor='black')  # Modification ici
    plt.title('Distribution des probabilités de prédiction')
    plt.xlabel('Probabilité')
    plt.ylabel('Fréquence')
    plt.savefig('prediction_probabilities.png')
    plt.close()

    # 8. Sauvegarder les résultats
    with open('evaluation_results.txt', 'w') as f:
        f.write(f'Précision: {precision:.4f}\n')
        f.write(f'Rappel: {recall:.4f}\n')
        f.write(f'F1 Score: {f1:.4f}\n')
        f.write(f'Matrice de confusion:\n{cm}')

if __name__ == "__main__":
    evaluate_model()

