# 🔬 XAI Medical Imaging: Vision Transformers vs CNN

## IA Explicable Centrée sur les Radiologues pour l'Assistance Diagnostique

Ce projet implémente une comparaison complète entre les architectures CNN et Vision Transformer (ViT) pour la classification d'images médicales, avec un focus particulier sur l'explicabilité (XAI).

---

## 📁 Structure du Projet

```
xai_medical_imaging/
├── configs/
│   └── config.yaml          # Configuration principale
├── data/
│   └── isic_dataset.py      # Chargement du dataset ISIC
├── models/
│   └── architectures.py     # CNN, ViT, Hybride
├── xai_methods/
│   └── explainers.py        # Méthodes XAI
├── notebooks/
│   └── demo_xai_medical.ipynb  # Notebook de démonstration
├── results/                  # Résultats sauvegardés
├── train.py                  # Script d'entraînement
├── evaluate_xai.py           # Évaluation XAI
├── requirements.txt          # Dépendances
└── README.md
```

---

## 🚀 Installation

### 1. Créer l'environnement

```bash
# Créer un environnement virtuel
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou: venv\Scripts\activate  # Windows

# Installer les dépendances
pip install -r requirements.txt
```

### 2. Télécharger le Dataset ISIC

```bash
# Option 1: Téléchargement manuel
# Visitez: https://challenge.isic-archive.com/data/
# Téléchargez ISIC 2019 Training Input + Ground Truth

# Option 2: Kaggle
kaggle competitions download -c siim-isic-melanoma-classification
```

Structure attendue:
```
data/ISIC2019/
├── ISIC_2019_Training_Input/
│   ├── ISIC_0000000.jpg
│   └── ...
└── ISIC_2019_Training_GroundTruth.csv
```

---

## 🎯 Utilisation

### Entraînement

```bash
# Entraîner ResNet-50 (CNN baseline)
python train.py --model resnet50 --epochs 50 --batch_size 32

# Entraîner ViT-Base
python train.py --model vit_base --epochs 50 --batch_size 32

# Entraîner Swin Transformer
python train.py --model swin_base --epochs 50 --batch_size 32

# Options disponibles
python train.py --help
```

### Évaluation XAI

```bash
# Évaluer les méthodes XAI sur un modèle entraîné
python evaluate_xai.py \
    --model resnet50 \
    --checkpoint results/resnet50_xxx/best_model.pth \
    --num_samples 100
```

### Notebook Interactif

```bash
jupyter notebook notebooks/demo_xai_medical.ipynb
```

---

## 🧠 Modèles Disponibles

| Modèle | Type | Paramètres | Pré-entraînement |
|--------|------|------------|------------------|
| `resnet50` | CNN | ~25M | ImageNet-1K |
| `efficientnet_b4` | CNN | ~19M | ImageNet-1K |
| `vit_base` | ViT | ~86M | ImageNet-21K |
| `deit_base` | ViT | ~86M | ImageNet-1K (distillé) |
| `swin_base` | ViT | ~88M | ImageNet-22K |
| `convnext_base` | Hybride | ~89M | ImageNet-22K |

---

## 🔍 Méthodes XAI Implémentées

### Pour tous les modèles
- **Grad-CAM**: Gradient-weighted Class Activation Mapping
- **Integrated Gradients**: Attribution basée sur l'intégrale des gradients
- **LIME**: Local Interpretable Model-agnostic Explanations

### Spécifiques aux ViT
- **Attention Rollout**: Agrégation des poids d'attention
- **Generic Attention** (Chefer et al.): Combinaison attention + gradients

---

## 📊 Métriques d'Évaluation

### Performance du Modèle
- Accuracy
- AUC-ROC (multiclass)
- F1-Score
- Matrice de Confusion

### Qualité XAI (Fidélité)
- **Insertion AUC**: Ajouter les pixels importants → confiance augmente
- **Deletion AUC**: Supprimer les pixels importants → confiance diminue
- **Pointing Game**: Le pixel le plus saillant est-il dans la lésion?
- **Sanity Checks**: Différence entre modèle entraîné vs aléatoire

---

## 📈 Exemple de Résultats

```
Métriques de Fidélité (Test sur ISIC 2019):
--------------------------------------------------
Méthode                  Insertion AUC    Deletion AUC
--------------------------------------------------
Grad-CAM (ResNet)        0.7234±0.05      0.1523±0.03
Grad-CAM (ViT)           0.7512±0.04      0.1398±0.02
Attention Rollout        0.6891±0.06      0.1687±0.04
Generic Attention        0.7823±0.03      0.1245±0.02
Integrated Gradients     0.7456±0.04      0.1412±0.03
```

---

## 📚 Références

### Architectures
1. [ViT: An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929)
2. [Swin Transformer](https://arxiv.org/abs/2103.14030)
3. [DeiT: Training Data-Efficient Image Transformers](https://arxiv.org/abs/2012.12877)

### XAI
4. [Grad-CAM](https://arxiv.org/abs/1610.02391)
5. [Transformer Interpretability Beyond Attention Visualization](https://arxiv.org/abs/2012.09838)
6. [SHAP](https://arxiv.org/abs/1705.07874)
7. [LIME](https://arxiv.org/abs/1602.04938)
8. [Integrated Gradients](https://arxiv.org/abs/1703.01365)

### Dataset
9. [ISIC Challenge](https://challenge.isic-archive.com/)

---

## 🤝 Contribution

Ce projet a été développé dans le cadre d'une thèse sur l'IA explicable en imagerie médicale.

## 📄 Licence

MIT License

---

## ⚠️ Avertissement

Ce projet est destiné à des fins de recherche uniquement. Les modèles ne sont pas validés cliniquement et ne doivent pas être utilisés pour des décisions médicales réelles.
