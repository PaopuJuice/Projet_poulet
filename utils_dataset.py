"""
Utilitaires pour préparer et analyser le dataset
"""

import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import defaultdict
import shutil


def analyze_dataset(images_dir, labels_dir):
    """
    Analyse complète du dataset : stats, distribution, visualisations
    """
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    
    print("="*60)
    print("DATASET ANALYSIS")
    print("="*60)
    
    image_files = sorted(list(images_dir.glob("*.jpg")))
    print(f"\nTotal images: {len(image_files)}")
    
    # Stats sur les boxes
    all_widths = []
    all_heights = []
    all_aspect_ratios = []
    images_with_boxes = 0
    total_boxes = 0
    boxes_per_image = []
    
    for img_path in image_files:
        label_path = labels_dir / (img_path.stem + ".txt")
        
        if not label_path.exists():
            boxes_per_image.append(0)
            continue
        
        with open(label_path, 'r') as f:
            lines = f.readlines()
            num_boxes = len(lines)
            
            if num_boxes > 0:
                images_with_boxes += 1
                total_boxes += num_boxes
                boxes_per_image.append(num_boxes)
                
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        _, x, y, w, h = map(float, parts[:5])
                        all_widths.append(w)
                        all_heights.append(h)
                        all_aspect_ratios.append(w / h if h > 0 else 1.0)
    
    print(f"Images with boxes: {images_with_boxes}")
    print(f"Total boxes: {total_boxes}")
    print(f"Average boxes per image: {np.mean(boxes_per_image):.2f}")
    print(f"Max boxes per image: {max(boxes_per_image)}")
    
    if len(all_widths) > 0:
        print(f"\nBox width statistics (normalized):")
        print(f"  Mean: {np.mean(all_widths):.4f}")
        print(f"  Std: {np.std(all_widths):.4f}")
        print(f"  Min: {np.min(all_widths):.4f}")
        print(f"  Max: {np.max(all_widths):.4f}")
        
        print(f"\nBox height statistics (normalized):")
        print(f"  Mean: {np.mean(all_heights):.4f}")
        print(f"  Std: {np.std(all_heights):.4f}")
        print(f"  Min: {np.min(all_heights):.4f}")
        print(f"  Max: {np.max(all_heights):.4f}")
        
        print(f"\nAspect ratio (w/h) statistics:")
        print(f"  Mean: {np.mean(all_aspect_ratios):.4f}")
        print(f"  Std: {np.std(all_aspect_ratios):.4f}")
        
        # Visualisations
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Distribution des largeurs
        axes[0, 0].hist(all_widths, bins=30, edgecolor='black')
        axes[0, 0].set_xlabel('Width (normalized)')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].set_title('Distribution of Box Widths')
        axes[0, 0].axvline(np.mean(all_widths), color='r', linestyle='--', label=f'Mean: {np.mean(all_widths):.3f}')
        axes[0, 0].legend()
        
        # Distribution des hauteurs
        axes[0, 1].hist(all_heights, bins=30, edgecolor='black')
        axes[0, 1].set_xlabel('Height (normalized)')
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].set_title('Distribution of Box Heights')
        axes[0, 1].axvline(np.mean(all_heights), color='r', linestyle='--', label=f'Mean: {np.mean(all_heights):.3f}')
        axes[0, 1].legend()
        
        # Distribution aspect ratios
        axes[1, 0].hist(all_aspect_ratios, bins=30, edgecolor='black')
        axes[1, 0].set_xlabel('Aspect Ratio (w/h)')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].set_title('Distribution of Aspect Ratios')
        axes[1, 0].axvline(np.mean(all_aspect_ratios), color='r', linestyle='--', label=f'Mean: {np.mean(all_aspect_ratios):.3f}')
        axes[1, 0].legend()
        
        # Scatter width vs height
        axes[1, 1].scatter(all_widths, all_heights, alpha=0.5)
        axes[1, 1].set_xlabel('Width (normalized)')
        axes[1, 1].set_ylabel('Height (normalized)')
        axes[1, 1].set_title('Width vs Height')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('dataset_analysis.png', dpi=150)
        print(f"\nSaved analysis plot to dataset_analysis.png")
        plt.close()
    
    return {
        'total_images': len(image_files),
        'images_with_boxes': images_with_boxes,
        'total_boxes': total_boxes,
        'avg_boxes_per_image': np.mean(boxes_per_image),
        'widths': all_widths,
        'heights': all_heights,
        'aspect_ratios': all_aspect_ratios
    }


def visualize_samples(images_dir, labels_dir, num_samples=9, save_path='dataset_samples.png'):
    """
    Visualise un échantillon d'images avec leurs bounding boxes
    """
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    
    image_files = sorted(list(images_dir.glob("*.jpg")))
    
    # Prendre des samples aléatoires
    np.random.seed(42)
    sample_indices = np.random.choice(len(image_files), min(num_samples, len(image_files)), replace=False)
    
    rows = int(np.sqrt(num_samples))
    cols = int(np.ceil(num_samples / rows))
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 15))
    axes = axes.flatten() if num_samples > 1 else [axes]
    
    for idx, ax in enumerate(axes):
        if idx >= len(sample_indices):
            ax.axis('off')
            continue
        
        img_path = image_files[sample_indices[idx]]
        label_path = labels_dir / (img_path.stem + ".txt")
        
        # Charger image
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(img_path.name, fontsize=8)
        
        # Dessiner les boxes
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        _, x_center, y_center, width, height = map(float, parts[:5])
                        
                        # Convertir YOLO vers pixels
                        x_center *= w
                        y_center *= h
                        width *= w
                        height *= h
                        
                        x1 = x_center - width / 2
                        y1 = y_center - height / 2
                        
                        rect = patches.Rectangle(
                            (x1, y1), width, height,
                            linewidth=2, edgecolor='lime', facecolor='none'
                        )
                        ax.add_patch(rect)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved sample visualizations to {save_path}")
    plt.close()


def split_train_val(images_dir, labels_dir, val_ratio=0.2, seed=42):
    """
    Split le dataset en train/val en créant une copie des fichiers
    """
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    
    # Créer les dossiers val
    val_images_dir = images_dir.parent / "val"
    val_labels_dir = labels_dir.parent / "val"
    
    val_images_dir.mkdir(exist_ok=True)
    val_labels_dir.mkdir(exist_ok=True)
    
    # Liste des images
    image_files = sorted(list(images_dir.glob("*.jpg")))
    
    # Split aléatoire
    np.random.seed(seed)
    indices = np.random.permutation(len(image_files))
    val_size = int(len(image_files) * val_ratio)
    val_indices = indices[:val_size]
    
    print(f"Splitting dataset: {len(image_files)} images")
    print(f"Train: {len(image_files) - val_size} images")
    print(f"Val: {val_size} images")
    
    # Copier les fichiers de validation
    for idx in val_indices:
        img_path = image_files[idx]
        label_path = labels_dir / (img_path.stem + ".txt")
        
        # Copier image
        shutil.copy(img_path, val_images_dir / img_path.name)
        
        # Copier label si existe
        if label_path.exists():
            shutil.copy(label_path, val_labels_dir / label_path.name)
    
    print(f"\nValidation set created in:")
    print(f"  Images: {val_images_dir}")
    print(f"  Labels: {val_labels_dir}")


def check_dataset_integrity(images_dir, labels_dir):
    """
    Vérifie l'intégrité du dataset
    """
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    
    print("="*60)
    print("DATASET INTEGRITY CHECK")
    print("="*60)
    
    image_files = sorted(list(images_dir.glob("*.jpg")))
    
    issues = []
    
    for img_path in image_files:
        label_path = labels_dir / (img_path.stem + ".txt")
        
        # Vérifier que l'image peut être lue
        img = cv2.imread(str(img_path))
        if img is None:
            issues.append(f"Cannot read image: {img_path.name}")
            continue
        
        # Vérifier les labels
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    parts = line.strip().split()
                    if len(parts) < 5:
                        issues.append(f"{img_path.name} - Line {line_num}: Invalid format (expected 5 values)")
                        continue
                    
                    try:
                        cls, x, y, w, h = map(float, parts[:5])
                        
                        # Vérifier les valeurs
                        if not (0 <= x <= 1 and 0 <= y <= 1):
                            issues.append(f"{img_path.name} - Line {line_num}: Center out of bounds")
                        
                        if not (0 < w <= 1 and 0 < h <= 1):
                            issues.append(f"{img_path.name} - Line {line_num}: Size out of bounds")
                        
                        if x - w/2 < 0 or x + w/2 > 1 or y - h/2 < 0 or y + h/2 > 1:
                            issues.append(f"{img_path.name} - Line {line_num}: Box extends outside image")
                    
                    except ValueError:
                        issues.append(f"{img_path.name} - Line {line_num}: Invalid number format")
    
    if len(issues) == 0:
        print("✓ No issues found!")
    else:
        print(f"✗ Found {len(issues)} issues:")
        for issue in issues[:20]:  # Show first 20
            print(f"  - {issue}")
        if len(issues) > 20:
            print(f"  ... and {len(issues) - 20} more")
    
    return issues


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Dataset utilities')
    parser.add_argument('--images', type=str, required=True, help='Path to images directory')
    parser.add_argument('--labels', type=str, required=True, help='Path to labels directory')
    parser.add_argument('--action', type=str, required=True, 
                       choices=['analyze', 'visualize', 'split', 'check'],
                       help='Action to perform')
    parser.add_argument('--val-ratio', type=float, default=0.2, help='Validation ratio for split')
    
    args = parser.parse_args()
    
    if args.action == 'analyze':
        analyze_dataset(args.images, args.labels)
    elif args.action == 'visualize':
        visualize_samples(args.images, args.labels)
    #elif args.action == 'split':
        #split_train_val(args.images, args.labels, val_ratio=args.val_ratio)
    elif args.action == 'check':
        check_dataset_integrity(args.images, args.labels)
