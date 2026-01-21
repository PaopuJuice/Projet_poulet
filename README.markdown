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
├── runs/
|   ├── best_front.pth      # Poids fine-tuné (avec front)
|
└── README.md
```

## 7. Entraînement (fine-tuning)

Le modèle BlazeFace est **ré-entraîné (fine-tuning)** sur une seule classe : `poule`, à partir des poids pré-entraînés fournis avec le modèle original.

L’objectif n’est pas de réentraîner un détecteur from scratch, mais de **réutiliser les capacités de localisation déjà apprises** par BlazeFace et de les adapter au nouveau domaine visuel.

### Principe
- Chargement des poids pré-entraînés BlazeFace
- Conservation de l’architecture (anchors, backbone, tête SSD)
- Apprentissage sur le dataset annoté de poules

### Paramètres principaux
- **Entrée réseau** : images redimensionnées en `128 × 128`
- **Sorties** :
  - score de présence de l’objet
  - coordonnées de la bounding box
- **Fonction de perte** :
  - perte de localisation (régression des bounding boxes)
  - perte de classification (présence / absence)

### Script utilisé
- `train_blazeface_chicken.py`

L’entraînement est réalisé sur PC (CPU), avec pour objectif principal la **validation fonctionnelle** du pipeline avant optimisation et portage embarqué.

### 7.1 Modèle BlazeFace utilisé

L’entraînement repose sur la variante **BlazeFace Front**, sélectionnée explicitement lors de l’initialisation du modèle :

- Paramètre : `back_model = False`
- Architecture : BlazeFace Front
- Résolution d’entrée : `128 × 128`

Ce choix est cohérent avec le contexte du projet :
- caméra fixe,
- champ de vision limité,
- poules occupant une portion significative de l’image.

La variante *Back* de BlazeFace (prévue pour des visages éloignés et des résolutions élevées) n’a pas été utilisée.

---

### 7.2 Poids pré-entraînés

Le fine-tuning est effectué à partir des **poids pré-entraînés du modèle BlazeFace Front** :

- Fichier : `blazeface.pth`
- Entraînement initial : détection de visages humains

Ces poids permettent d’initialiser :
- le backbone convolutionnel,
- la tête de détection SSD-like,
- avec des représentations visuelles génériques (bords, textures, formes).

Le modèle n’est donc **pas entraîné from scratch**, mais adapté par transfert de connaissances au domaine des poules.

---

### 7.3 Anchors utilisées

BlazeFace repose sur un mécanisme d’**anchors fixes**, prédéfinies pour différentes tailles et positions d’objets.

Dans ce projet :
- seules les **anchors du modèle Front** sont utilisées,
- fichier : `anchors.npy`,
- les anchors sont **figées** pendant l’entraînement (non ré-apprises).

Ces anchors sont adaptées à la détection d’objets compacts en temps réel.

Les anchors associées au modèle Back (`anchorsback.npy`) ne sont pas utilisées.

---

### 7.4 Modèle final entraîné

Le modèle final issu de l’entraînement est sauvegardé sous la forme :

- `best_front.pth`

Ce fichier correspond à :
- un modèle **BlazeFace Front**,
- initialisé à partir des poids pré-entraînés `blazeface.pth`,
- utilisant les **anchors Front (`anchors.npy`)**,
- fine-tuné sur le dataset annoté de poules.

Schéma récapitulatif :

```text
blazeface.pth
   ↓  (initialisation)
BlazeFace FRONT + anchors.npy
   ↓  (fine-tuning sur poules)
best_front.pth
```

---

## 8. Inférence

### 8.1 Inférence sur image

Le script `infer_chicken.py` permet de tester le modèle entraîné sur des images statiques.

Pipeline d’inférence :
1. chargement du modèle et des poids entraînés,
2. pré-traitement de l’image (resize, normalisation),
3. passage dans le réseau (forward pass),
4. décodage des bounding boxes à partir des anchors,
5. filtrage par seuil de confiance,
6. suppression des doublons via **Non-Maximum Suppression (NMS)**,
7. affichage des résultats (bounding box + score).

Ce script permet de vérifier visuellement la qualité de la détection.

---

### 8.2 Inférence temps réel (webcam)

Le script `webcam_chicken.py` implémente une détection **temps réel** à partir d’une webcam via OpenCV.

Pour chaque frame :
- capture de l’image,
- inférence BlazeFace,
- décodage et filtrage des détections,
- affichage des bounding boxes et scores.

---

### 8.3 Logique temporelle (robustesse système)

#### Problème observé
Le modèle peut produire des **faux positifs** (ex. visages), phénomène courant lors de l’adaptation d’un détecteur CNN à un nouveau domaine visuel.

#### Solution mise en place
Plutôt que de prendre une décision sur une seule frame, une **logique temporelle** est appliquée :

- une poule doit être détectée **pendant un temps minimal continu**,
- de courtes pertes de détection sont tolérées.

Règle de décision :
```text
Si poule détectée ≥ 2 secondes → OUVERTURE
Sinon → FERMÉ

