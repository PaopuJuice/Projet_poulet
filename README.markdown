# 🐔 Chicken Detection with BlazeFace (Projet IESE5)

## 1. Présentation du projet

Ce projet vise à développer un **système de détection automatique de poules** destiné à piloter l’ouverture d’un poulailler intelligent.

Le système repose sur un **réseau de neurones convolutifs léger**, capable de fonctionner en **temps réel**, avec pour objectif final un **déploiement embarqué sur une puce NPU (Colibri – ASYGN)**.

Le développement est réalisé en priorité sur PC afin de valider :
- la faisabilité algorithmique,
- la robustesse du pipeline,
- et la logique système avant portage embarqué.

---

## 2. Choix du modèle : BlazeFace

Nous utilisons **BlazeFace**, un modèle initialement conçu pour la détection de visages sur mobile.

**Référence scientifique :**  
> *BlazeFace: Sub-millisecond Neural Face Detection on Mobile GPUs*  
> https://arxiv.org/abs/1907.05047

### Pourquoi BlazeFace ?
- Architecture **SSD-like** (Single Shot Detector)
- Réseau **très léger** (depthwise convolutions)
- Faible latence
- Compatible avec les opérateurs supportés par la puce Colibri :
  - Convolution
  - Depthwise Convolution
  - MaxPooling
  - Global Average Pooling
  - Dense

Nous utilisons l’implémentation PyTorch open-source :
- **BlazeFace-PyTorch** (repo original : hollance)

---

## 3. Environnement de développement

- **OS** : Windows
- **Python** : 3.10
- **Gestion d’environnement** : Anaconda
- **IDE** : Spyder
- **Framework DL** : PyTorch (CPU)
- **Vision** : OpenCV

> Le développement est volontairement effectué sur PC avant portage sur Raspberry Pi puis NPU.

---

## 4. Dataset

### 4.1 Données
- Images de **poules seules**
- Variabilité :
  - points de vue (profil, face, plongée),
  - tailles,
  - arrière-plans,
  - conditions lumineuses.

### 4.2 Répartition
| Split | Nombre d’images |
|-----|-----------------|
| Train | 509 |
| Validation | 142 |
| Test | 59 |

---

## 5. Annotation

BlazeFace étant un **modèle de détection**, les images ont été **annotées manuellement** avec des **bounding boxes**.

- Outil : `labelImg`
- Format : YOLO (`.txt`)
- Classe unique : `poule`

Chaque image possède un fichier d’annotation décrivant la position de la poule dans l’image.

---

## 6. Structure du projet

```text
BlazeFace-PyTorch/
│
├── blazeface.py                # Modèle BlazeFace
├── blazeface.pth               # Poids pré-entraînés (front)
├── blazefaceback.pth           # Poids pré-entraînés (back)
├── anchors.npy                 # Anchors front
├── anchorsback.npy             # Anchors back
│
├── dataset/
│   ├── images/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── labels/
│       ├── train/
│       ├── val/
│       └── test/
│
├── src/
│   ├── chicken_dataset.py      # Dataset PyTorch personnalisé
│   ├── test_dataset.py         # Vérification dataset
│   ├── train_blazeface_chicken.py  # Entraînement / fine-tuning
│   ├── infer_chicken.py        # Inférence sur image
│   └── webcam_chicken.py       # Inférence temps réel webcam
│
└── README.md
