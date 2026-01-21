"""
Script d'entraînement complet pour BlazeFace - Détection de poules
Comprend : data augmentation, génération d'anchors, validation, metrics
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
from sklearn.cluster import KMeans
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt

from blazeface import BlazeFace


class PouleDataset(Dataset):
    """Dataset pour les images de poules avec augmentation"""
    
    def __init__(self, images_dir, labels_dir, img_size=256, augment=True):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.img_size = img_size
        self.augment = augment
        
        # Liste tous les fichiers images
        self.image_files = sorted(list(self.images_dir.glob("*.jpg")))
        print(f"Found {len(self.image_files)} images in {images_dir}")
        
        # Augmentation robuste
        if augment:
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=20, p=0.3),
                A.GaussianBlur(blur_limit=(3, 5), p=0.2),
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
                A.RandomScale(scale_limit=0.1, p=0.3),
                A.Affine(rotate=(-10, 10), translate_percent=0.1, scale=(0.9, 1.1), p=0.3),
            ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
        else:
            self.transform = None
    
    def __len__(self):
        return len(self.image_files)
    
    def yolo_to_corners(self, x_center, y_center, width, height):
        """Convertit format YOLO vers [ymin, xmin, ymax, xmax]"""
        xmin = x_center - width / 2
        xmax = x_center + width / 2
        ymin = y_center - height / 2
        ymax = y_center + height / 2
        return ymin, xmin, ymax, xmax
    
    def __getitem__(self, idx):
        # Charger l'image
        img_path = self.image_files[idx]
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Charger les labels YOLO
        label_path = self.labels_dir / (img_path.stem + ".txt")
        bboxes_yolo = []
        class_labels = []
        
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls, x, y, w, h = map(float, parts[:5])
                        bboxes_yolo.append([x, y, w, h])
                        class_labels.append(int(cls))
        
        # Appliquer augmentation
        if self.augment and len(bboxes_yolo) > 0:
            transformed = self.transform(
                image=img,
                bboxes=bboxes_yolo,
                class_labels=class_labels
            )
            img = transformed['image']
            bboxes_yolo = transformed['bboxes']
            class_labels = transformed['class_labels']
        
        # Resize
        img = cv2.resize(img, (self.img_size, self.img_size))
        
        # Normaliser [-1, 1]
        img = torch.from_numpy(img).permute(2, 0, 1).float()
        img = img / 127.5 - 1.0
        
        # Convertir boxes en format corners
        boxes = []
        for bbox, cls in zip(bboxes_yolo, class_labels):
            x, y, w, h = bbox
            ymin, xmin, ymax, xmax = self.yolo_to_corners(x, y, w, h)
            boxes.append([ymin, xmin, ymax, xmax, 1.0])  # classe=1 pour poule
        
        if len(boxes) == 0:
            boxes = torch.zeros((0, 5))
        else:
            boxes = torch.tensor(boxes, dtype=torch.float32)
        
        return img, boxes


def generate_anchors_from_dataset(dataset, num_anchors=896, img_size=256):
    """
    Génère des anchors optimisés pour le dataset en utilisant K-means
    sur les dimensions des bounding boxes
    """
    print("Generating custom anchors from dataset...")
    
    all_boxes = []
    for idx in tqdm(range(len(dataset))):
        _, boxes = dataset[idx]
        for box in boxes:
            ymin, xmin, ymax, xmax = box[:4]
            w = xmax - xmin
            h = ymax - ymin
            all_boxes.append([w, h])
    
    all_boxes = np.array(all_boxes)
    print(f"Collected {len(all_boxes)} boxes from dataset")
    
    if len(all_boxes) == 0:
        print("WARNING: No boxes found, using default anchors")
        return None
    
    # Statistiques des boxes
    print(f"Box width  - mean: {all_boxes[:, 0].mean():.4f}, std: {all_boxes[:, 0].std():.4f}")
    print(f"Box height - mean: {all_boxes[:, 1].mean():.4f}, std: {all_boxes[:, 1].std():.4f}")
    
    # K-means pour trouver les tailles d'anchors optimales
    num_anchor_shapes = 6  # Nombre de formes d'anchors différentes
    kmeans = KMeans(n_clusters=num_anchor_shapes, random_state=42, n_init=10)
    kmeans.fit(all_boxes)
    
    anchor_shapes = kmeans.cluster_centers_
    print(f"Anchor shapes found: {anchor_shapes}")
    
    # Créer la grille d'anchors pour BlazeFace
    # BlazeFace utilise 2 feature maps: 16x16 et 8x8
    anchors = []
    
    # Feature map 16x16 (512 anchors - 2 par position)
    stride_16 = 1.0 / 16
    for i in range(16):
        for j in range(16):
            x_center = (j + 0.5) * stride_16
            y_center = (i + 0.5) * stride_16
            
            # 2 anchors par position (petites formes)
            for shape_idx in [0, 1]:
                w, h = anchor_shapes[shape_idx]
                anchors.append([x_center, y_center, w, h])
    
    # Feature map 8x8 (384 anchors - 6 par position)
    stride_8 = 1.0 / 8
    for i in range(8):
        for j in range(8):
            x_center = (j + 0.5) * stride_8
            y_center = (i + 0.5) * stride_8
            
            # 6 anchors par position (toutes les formes)
            for shape_idx in range(num_anchor_shapes):
                w, h = anchor_shapes[shape_idx]
                anchors.append([x_center, y_center, w, h])
    
    anchors = np.array(anchors, dtype=np.float32)
    print(f"Generated {len(anchors)} anchors")
    
    return anchors


class BlazeFaceLoss(nn.Module):
    """Loss function complète pour BlazeFace avec Focal Loss"""
    
    def __init__(self, num_anchors=896, alpha=0.25, gamma=2.0, box_weight=2.0):
        super().__init__()
        self.num_anchors = num_anchors
        self.alpha = alpha
        self.gamma = gamma
        self.box_weight = box_weight
        
    def focal_loss(self, pred, target):
        """Focal Loss pour gérer le déséquilibre de classes"""
        pred_sigmoid = torch.sigmoid(pred)
        pt = pred_sigmoid * target + (1 - pred_sigmoid) * (1 - target)
        focal_weight = (self.alpha * target + (1 - self.alpha) * (1 - target)) * torch.pow(1 - pt, self.gamma)
        bce_loss = nn.functional.binary_cross_entropy_with_logits(pred, target, reduction='none')
        return (focal_weight * bce_loss).mean()
    
    def smooth_l1_loss(self, pred, target, beta=1.0):
        """Smooth L1 Loss pour la régression"""
        diff = torch.abs(pred - target)
        loss = torch.where(diff < beta, 0.5 * diff ** 2 / beta, diff - 0.5 * beta)
        return loss.mean()
    
    def compute_iou_batch(self, box1, box2):
        """Calcule l'IoU entre deux ensembles de boxes de manière vectorisée"""
        # box1: (N, 4), box2: (M, 4) -> (N, M)
        box1 = box1.unsqueeze(1)  # (N, 1, 4)
        box2 = box2.unsqueeze(0)  # (1, M, 4)
        
        # Intersection
        y1 = torch.max(box1[..., 0], box2[..., 0])
        x1 = torch.max(box1[..., 1], box2[..., 1])
        y2 = torch.min(box1[..., 2], box2[..., 2])
        x2 = torch.min(box1[..., 3], box2[..., 3])
        
        inter_area = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
        
        # Union
        box1_area = (box1[..., 2] - box1[..., 0]) * (box1[..., 3] - box1[..., 1])
        box2_area = (box2[..., 2] - box2[..., 0]) * (box2[..., 3] - box2[..., 1])
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / (union_area + 1e-6)
    
    def encode_boxes(self, boxes, anchors, x_scale=256.0, y_scale=256.0, w_scale=256.0, h_scale=256.0):
        """
        Encode les boxes ground truth en format BlazeFace
        boxes: (N, 4) [ymin, xmin, ymax, xmax]
        anchors: (N, 4) [x_center, y_center, w, h]
        Returns: (N, 16) encodées
        """
        # Convertir boxes en center format
        box_y = (boxes[:, 0] + boxes[:, 2]) / 2
        box_x = (boxes[:, 1] + boxes[:, 3]) / 2
        box_h = boxes[:, 2] - boxes[:, 0]
        box_w = boxes[:, 3] - boxes[:, 1]
        
        # Encoder par rapport aux anchors
        encoded = torch.zeros((boxes.shape[0], 16), device=boxes.device)
        
        # Center offsets
        encoded[:, 0] = (box_x - anchors[:, 0]) / anchors[:, 2] * x_scale
        encoded[:, 1] = (box_y - anchors[:, 1]) / anchors[:, 3] * y_scale
        
        # Size
        encoded[:, 2] = box_w / anchors[:, 2] * w_scale
        encoded[:, 3] = box_h / anchors[:, 3] * h_scale
        
        # Keypoints (pas utilisés pour poules, mettre à 0)
        encoded[:, 4:16] = 0
        
        return encoded
    
    def match_anchors(self, anchors, target_boxes, iou_threshold=0.5):
        """
        Associe chaque anchor à une target box basé sur l'IoU
        Returns: 
            - pos_mask: (num_anchors,) bool - True si positif
            - neg_mask: (num_anchors,) bool - True si négatif
            - matched_gt_boxes: (num_anchors, 4) - box GT associée
            - matched_idx: (num_anchors,) - index de la GT box
        """
        num_anchors = anchors.shape[0]
        num_targets = target_boxes.shape[0]
        
        # Convertir anchors en format corners
        anchor_boxes = torch.zeros((num_anchors, 4), device=anchors.device)
        anchor_boxes[:, 0] = anchors[:, 1] - anchors[:, 3] / 2  # ymin
        anchor_boxes[:, 1] = anchors[:, 0] - anchors[:, 2] / 2  # xmin
        anchor_boxes[:, 2] = anchors[:, 1] + anchors[:, 3] / 2  # ymax
        anchor_boxes[:, 3] = anchors[:, 0] + anchors[:, 2] / 2  # xmax
        
        if num_targets == 0:
            # Pas de targets, tous négatifs
            pos_mask = torch.zeros(num_anchors, dtype=torch.bool, device=anchors.device)
            neg_mask = torch.ones(num_anchors, dtype=torch.bool, device=anchors.device)
            matched_gt_boxes = torch.zeros((num_anchors, 4), device=anchors.device)
            matched_idx = torch.zeros(num_anchors, dtype=torch.long, device=anchors.device)
            return pos_mask, neg_mask, matched_gt_boxes, matched_idx
        
        # Calculer IoU pour toutes les paires (vectorisé)
        ious = self.compute_iou_batch(anchor_boxes, target_boxes[:, :4])  # (num_anchors, num_targets)
        
        # Pour chaque anchor, trouver la meilleure target
        best_iou, best_target_idx = ious.max(dim=1)
        
        # Positifs: IoU > iou_threshold
        pos_mask = best_iou > iou_threshold
        
        # Négatifs: IoU < 0.3
        neg_mask = best_iou < 0.3
        
        # Boxes GT matchées
        matched_gt_boxes = target_boxes[best_target_idx, :4]
        matched_idx = best_target_idx
        
        return pos_mask, neg_mask, matched_gt_boxes, matched_idx
    
    def forward(self, predictions, targets, anchors):
        """
        predictions: liste [reg_output, cls_output]
            - reg_output: (batch, 896, 16)
            - cls_output: (batch, 896, 1)
        targets: liste de tensors de boxes par image (liste de longueur batch)
        anchors: (896, 4)
        """
        reg_pred, cls_pred = predictions
        batch_size = reg_pred.shape[0]
        
        total_cls_loss = torch.tensor(0.0, device=reg_pred.device)
        total_reg_loss = torch.tensor(0.0, device=reg_pred.device)
        num_pos = 0
        
        for b in range(batch_size):
            # Match anchors avec targets
            pos_mask, neg_mask, matched_gt_boxes, matched_idx = self.match_anchors(
                anchors, targets[b], iou_threshold=0.5
            )
            
            num_pos += pos_mask.sum().item()
            
            # === Classification Loss ===
            cls_target = torch.zeros_like(cls_pred[b])
            cls_target[pos_mask] = 1.0
            
            # Utiliser seulement pos + neg pour la loss (ignorer les ambigus)
            cls_mask = pos_mask | neg_mask
            
            if cls_mask.sum() > 0:
                cls_loss = self.focal_loss(
                    cls_pred[b][cls_mask],
                    cls_target[cls_mask]
                )
                total_cls_loss += cls_loss
            
            # === Regression Loss (seulement sur les positives) ===
            if pos_mask.sum() > 0:
                # Encoder les GT boxes
                matched_anchors = anchors[pos_mask]
                matched_boxes = matched_gt_boxes[pos_mask]
                
                encoded_targets = self.encode_boxes(
                    matched_boxes,
                    matched_anchors
                )
                
                # Loss seulement sur les 4 premières coordonnées (bbox)
                reg_loss = self.smooth_l1_loss(
                    reg_pred[b][pos_mask, :4],
                    encoded_targets[:, :4]
                )
                total_reg_loss += reg_loss
        
        # Moyenne sur le batch
        total_cls_loss /= batch_size
        total_reg_loss /= batch_size
        
        # Pondération
        total_loss = total_cls_loss + self.box_weight * total_reg_loss
        
        return {
            'total': total_loss,
            'cls': total_cls_loss,
            'reg': total_reg_loss,
            'num_pos': num_pos / batch_size
        }


def compute_map(model, dataloader, device, iou_threshold=0.5):
    """Calcule le mAP sur un dataset"""
    model.eval()
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc="Computing mAP"):
            images = torch.stack(images).to(device)
            
            # Prédictions
            detections = model.predict_on_batch(images)
            
            for det, target in zip(detections, targets):
                all_predictions.append(det.cpu())
                all_targets.append(target.cpu())
    
    # Calculer mAP
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    
    for pred, target in zip(all_predictions, all_targets):
        num_pred = pred.shape[0]
        num_target = target.shape[0]
        
        if num_pred == 0 and num_target == 0:
            continue
        elif num_pred == 0:
            false_negatives += num_target
            continue
        elif num_target == 0:
            false_positives += num_pred
            continue
        
        # Calculer IoU entre prédictions et targets
        pred_boxes = pred[:, :4]
        target_boxes = target[:, :4]
        
        matched_targets = set()
        
        for i in range(num_pred):
            best_iou = 0
            best_idx = -1
            
            for j in range(num_target):
                if j in matched_targets:
                    continue
                
                # Calculer IoU
                iou = compute_single_iou(pred_boxes[i], target_boxes[j])
                
                if iou > best_iou:
                    best_iou = iou
                    best_idx = j
            
            if best_iou >= iou_threshold:
                true_positives += 1
                matched_targets.add(best_idx)
            else:
                false_positives += 1
        
        false_negatives += (num_target - len(matched_targets))
    
    # Calculer precision, recall, F1
    precision = true_positives / (true_positives + false_positives + 1e-6)
    recall = true_positives / (true_positives + false_negatives + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': true_positives,
        'fp': false_positives,
        'fn': false_negatives
    }


def compute_single_iou(box1, box2):
    """Calcule IoU entre deux boxes [ymin, xmin, ymax, xmax]"""
    y1 = max(box1[0].item(), box2[0].item())
    x1 = max(box1[1].item(), box2[1].item())
    y2 = min(box1[2].item(), box2[2].item())
    x2 = min(box1[3].item(), box2[3].item())
    
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / (union_area + 1e-6)


def train_epoch(model, dataloader, optimizer, criterion, anchors, device, epoch):
    model.train()
    
    # === CES LIGNES DOIVENT ÊTRE PRÉSENTES ===
    total_loss = 0.0
    total_cls_loss = 0.0
    total_reg_loss = 0.0
    total_num_pos = 0.0
    # =========================================
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch_idx, (images, targets) in enumerate(pbar):
        # Correction précédente pour le tuple d'images
        images = torch.stack(images).to(device)
        targets = [t.to(device) for t in targets]
        
        optimizer.zero_grad()
        
        # Forward
        predictions = model(images)
        
        # Loss
        loss_dict = criterion(predictions, targets, anchors)
        loss = loss_dict['total']
        
        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()
        
        # === STATS (Utilisation de float() pour la robustesse) ===
        total_loss += float(loss)
        total_cls_loss += float(loss_dict['cls'])
        total_reg_loss += float(loss_dict['reg'])
        total_num_pos += float(loss_dict['num_pos'])
        
        # Mise à jour de la barre de progression
        pbar.set_postfix({
            'loss': f"{float(loss):.4f}",
            'cls': f"{float(loss_dict['cls']):.4f}",
            'reg': f"{float(loss_dict['reg']):.4f}",
            'pos': f"{float(loss_dict['num_pos']):.1f}"
        })
    
    return {
        'loss': total_loss / len(dataloader),
        'cls_loss': total_cls_loss / len(dataloader),
        'reg_loss': total_reg_loss / len(dataloader),
        'avg_pos': total_num_pos / len(dataloader)
    }

def validate_epoch(model, dataloader, criterion, anchors, device):
    model.eval()
    
    total_loss = 0
    total_cls_loss = 0
    total_reg_loss = 0
    
    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc="Validation"):
            images = torch.stack(images).to(device)
            targets = [t.to(device) for t in targets]
            
            # Forward
            predictions = model(images)
            
            # Loss
            loss_dict = criterion(predictions, targets, anchors)
            
            total_loss += loss_dict['total'].item()
            total_cls_loss += loss_dict['cls'].item()
            total_reg_loss += loss_dict['reg'].item()
    
    return {
        'loss': total_loss / len(dataloader),
        'cls_loss': total_cls_loss / len(dataloader),
        'reg_loss': total_reg_loss / len(dataloader)
    }


def main():
    # Configuration
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # ==================== HYPERPARAMÈTRES ====================
    BATCH_SIZE = 8
    EPOCHS = 20
    LEARNING_RATE = 1e-3
    IMG_SIZE = 256
    WEIGHT_DECAY = 1e-4
    
    # Paths
    TRAIN_IMAGES = "images/images/train"
    TRAIN_LABELS = "labels/labels/train"
    VAL_IMAGES = "images/images/val"  # À créer si nécessaire
    VAL_LABELS = "labels/labels/val"
    
    OUTPUT_DIR = Path("outputs")
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # ==================== DATASETS ====================
    print("\n=== Loading Datasets ===")
    train_dataset = PouleDataset(
        images_dir=TRAIN_IMAGES,
        labels_dir=TRAIN_LABELS,
        img_size=IMG_SIZE,
        augment=True
    )
    
    # Validation dataset (pas d'augmentation)
    val_dataset = None
    if Path(VAL_IMAGES).exists():
        val_dataset = PouleDataset(
            images_dir=VAL_IMAGES,
            labels_dir=VAL_LABELS,
            img_size=IMG_SIZE,
            augment=False
        )
        print(f"Validation set: {len(val_dataset)} images")
    else:
        print("No validation set found, using 20% of training data")
        # Split train/val
        train_size = int(0.8 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            train_dataset, [train_size, val_size]
        )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        collate_fn=lambda x: tuple(zip(*x)),
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        collate_fn=lambda x: tuple(zip(*x)),
        pin_memory=True
    )
    
    # ==================== GENERATE ANCHORS ====================
    print("\n=== Generating Custom Anchors ===")
    # Utiliser le train dataset original (pas le split)
    if isinstance(train_dataset, torch.utils.data.Subset):
        anchor_dataset = train_dataset.dataset
    else:
        anchor_dataset = train_dataset
    
    custom_anchors = generate_anchors_from_dataset(anchor_dataset, num_anchors=896, img_size=IMG_SIZE)
    
    if custom_anchors is not None:
        np.save(OUTPUT_DIR / "anchors.npy", custom_anchors)
        print(f"Saved custom anchors to {OUTPUT_DIR / 'anchors.npy'}")
        anchors_path = OUTPUT_DIR / "anchors.npy"
    else:
        print("Using default anchors")
        anchors_path = "anchors.npy"
    
    # ==================== MODEL ====================
    print("\n=== Initializing Model ===")
    model = BlazeFace(back_model=True).to(device)
    
    # Optionnel: charger poids pré-entraînés pour le backbone
    # model.load_weights("blazeface_back.pth")
    # print("Loaded pretrained weights")
    
    # Charger les anchors
    model.load_anchors(str(anchors_path))
    
    # Ne pas freeze le backbone pour un entraînement complet
    # Si vous voulez commencer avec backbone freezé:
    # for param in model.backbone.parameters():
    #     param.requires_grad = False
    
    # ==================== LOSS & OPTIMIZER ====================
    criterion = BlazeFaceLoss(alpha=0.25, gamma=2.0, box_weight=2.0)
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-6
    )
    
    # ==================== TRAINING LOOP ====================
    print("\n=== Starting Training ===")
    
    best_val_loss = float('inf')
    best_f1 = 0.0
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_f1': []
    }
    
    for epoch in range(1, EPOCHS + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{EPOCHS}")
        print(f"Learning rate: {optimizer.param_groups[0]['lr']:.6f}")
        print(f"{'='*60}")
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, criterion, model.anchors, device, epoch
        )
        
        print(f"\nTraining - Loss: {train_metrics['loss']:.4f}, "
              f"Cls: {train_metrics['cls_loss']:.4f}, "
              f"Reg: {train_metrics['reg_loss']:.4f}, "
              f"Avg Pos: {train_metrics['avg_pos']:.1f}")
        
        # Validate
        val_metrics = validate_epoch(model, val_loader, criterion, model.anchors, device)
        
        print(f"Validation - Loss: {val_metrics['loss']:.4f}, "
              f"Cls: {val_metrics['cls_loss']:.4f}, "
              f"Reg: {val_metrics['reg_loss']:.4f}")
        
        # Compute mAP every 5 epochs
        if epoch % 5 == 0:
            print("\nComputing mAP...")
            map_metrics = compute_map(model, val_loader, device, iou_threshold=0.5)
            print(f"mAP Metrics - Precision: {map_metrics['precision']:.4f}, "
                  f"Recall: {map_metrics['recall']:.4f}, "
                  f"F1: {map_metrics['f1']:.4f}")
            history['val_f1'].append(map_metrics['f1'])
            
            # Save best model based on F1
            if map_metrics['f1'] > best_f1:
                best_f1 = map_metrics['f1']
                torch.save(model.state_dict(), OUTPUT_DIR / "best_model_f1.pth")
                print(f"✓ Saved best F1 model (F1={best_f1:.4f})")
        
        # Save best model based on validation loss
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            torch.save(model.state_dict(), OUTPUT_DIR / "best_model_loss.pth")
            print(f"✓ Saved best loss model (loss={best_val_loss:.4f})")
        
        # History
        history['train_loss'].append(train_metrics['loss'])
        history['val_loss'].append(val_metrics['loss'])
        
        # Step scheduler
        scheduler.step()
        
        # Save checkpoint every 10 epochs
        if epoch % 10 == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_metrics['loss'],
                'val_loss': val_metrics['loss'],
            }
            torch.save(checkpoint, OUTPUT_DIR / f"checkpoint_epoch{epoch}.pth")
            print(f"✓ Saved checkpoint at epoch {epoch}")
    
    # ==================== SAVE FINAL MODEL ====================
    print("\n=== Training Completed ===")
    torch.save(model.state_dict(), OUTPUT_DIR / "model_poule_final.pth")
    print(f"Final model saved to {OUTPUT_DIR / 'model_poule_final.pth'}")
    
    # Save history
    with open(OUTPUT_DIR / "training_history.json", 'w') as f:
        json.dump(history, f, indent=2)
    
    # Plot training curves
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.grid(True)
    
    if len(history['val_f1']) > 0:
        plt.subplot(1, 2, 2)
        epochs_f1 = list(range(5, len(history['val_f1']) * 5 + 1, 5))
        plt.plot(epochs_f1, history['val_f1'], marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('F1 Score')
        plt.title('Validation F1 Score')
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "training_curves.png", dpi=150)
    print(f"Training curves saved to {OUTPUT_DIR / 'training_curves.png'}")
    
    print("\n" + "="*60)
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best F1 score: {best_f1:.4f}")
    print("="*60)


if __name__ == "__main__":
    main()
