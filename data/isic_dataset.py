"""
ISIC Dataset Loader
===================
Module pour charger et prétraiter le dataset ISIC 2019 pour la classification
des lésions cutanées.

Auteur: [youssef nouiouar]
Date: 2025
"""

import os
import pandas as pd
import numpy as np
from PIL import Image
from typing import Tuple, Optional, Dict, List
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split


class ISICDataset(Dataset):
    """
    Dataset personnalisé pour ISIC 2019.
    
    Structure attendue du dataset:
    data/ISIC2019/
    ├── ISIC_2019_Training_Input/
    │   ├── ISIC_0000000.jpg
    │   ├── ISIC_0000001.jpg
    │   └── ...
    ├── ISIC_2019_Training_GroundTruth.csv
    └── ISIC_2019_Training_Metadata.csv (optionnel)
    """
    
    # Mapping des classes ISIC 2019 (UNK excluded — 0 samples in dataset)
    CLASS_NAMES = ['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC']
    CLASS_FULL_NAMES = {
        'MEL': 'Melanoma',
        'NV': 'Melanocytic nevus',
        'BCC': 'Basal cell carcinoma',
        'AK': 'Actinic keratosis',
        'BKL': 'Benign keratosis',
        'DF': 'Dermatofibroma',
        'VASC': 'Vascular lesion',
        'SCC': 'Squamous cell carcinoma',
    }
    
    def __init__(
        self,
        root_dir: str,
        split: str = 'train',
        transform: Optional[transforms.Compose] = None,
        image_size: int = 224,
        use_albumentations: bool = True
    ):
        """
        Args:
            root_dir: Chemin vers le dossier ISIC2019
            split: 'train', 'val', ou 'test'
            transform: Transformations torchvision (si use_albumentations=False)
            image_size: Taille des images en sortie
            use_albumentations: Utiliser albumentations pour l'augmentation
        """
        self.root_dir = root_dir
        self.split = split
        self.image_size = image_size
        self.use_albumentations = use_albumentations
        
        # Charger les labels
        self.df = self._load_data()
        
        # Configurer les transformations
        if use_albumentations:
            self.transform = self._get_albumentations_transform()
        else:
            self.transform = transform or self._get_default_transform()
            
    def _load_data(self) -> pd.DataFrame:
        """Charger et préparer les données."""
        # Chemin vers le fichier CSV des labels
        csv_path = os.path.join(self.root_dir, 'ISIC_2019_Training_GroundTruth.csv')
        
        if not os.path.exists(csv_path):
            raise FileNotFoundError(
                f"Fichier CSV non trouvé: {csv_path}\n"
                "Téléchargez le dataset depuis: https://challenge.isic-archive.com/data/"
            )
        
        df = pd.read_csv(csv_path)

        # Drop UNK column if present (0 samples in ISIC 2019)
        if 'UNK' in df.columns:
            df = df.drop(columns=['UNK'])

        # Convertir one-hot encoding en labels
        label_columns = [col for col in df.columns if col != 'image']
        df['label'] = df[label_columns].values.argmax(axis=1)
        df['class_name'] = df['label'].map(lambda x: self.CLASS_NAMES[x])
        
        # Split des données
        train_df, temp_df = train_test_split(
            df, test_size=0.2, stratify=df['label'], random_state=42
        )
        val_df, test_df = train_test_split(
            temp_df, test_size=0.5, stratify=temp_df['label'], random_state=42
        )
        
        if self.split == 'train':
            return train_df.reset_index(drop=True)
        elif self.split == 'val':
            return val_df.reset_index(drop=True)
        else:  # test
            return test_df.reset_index(drop=True)
    
    def _get_albumentations_transform(self) -> A.Compose:
        """Transformations avec Albumentations."""
        if self.split == 'train':
            return A.Compose([
                A.RandomResizedCrop(
                  size=(self.image_size, self.image_size),
                  scale=(0.8, 1.0),
                  ratio=(0.75, 1.33)
                ),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Rotate(limit=20, p=0.5),
                A.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.1,
                    p=0.5
                ),
                A.GaussNoise(std_range=(0.02, 0.1), p=0.2),
                A.GaussianBlur(blur_limit=(3, 7), p=0.2),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                ToTensorV2()
            ])
        else:
            return A.Compose([
                A.Resize(self.image_size, self.image_size),
                A.CenterCrop(self.image_size, self.image_size),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                ToTensorV2()
            ])
    
    def _get_default_transform(self) -> transforms.Compose:
        """Transformations par défaut avec torchvision."""
        if self.split == 'train':
            return transforms.Compose([
                transforms.RandomResizedCrop(self.image_size),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(20),
                transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            return transforms.Compose([
                transforms.Resize(self.image_size + 32),
                transforms.CenterCrop(self.image_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        """
        Returns:
            image: Tensor de l'image transformée
            label: Label de classe (int)
            image_id: ID de l'image pour le tracking
        """
        row = self.df.iloc[idx]
        image_id = row['image']
        label = row['label']
        
        # Charger l'image
        image_path = os.path.join(
            self.root_dir,
            'ISIC_2019_Training_Input',
            f'{image_id}.jpg'
        )
        
        image = Image.open(image_path).convert('RGB')
        
        # Appliquer les transformations
        if self.use_albumentations:
            image = np.array(image)
            transformed = self.transform(image=image)
            image = transformed['image']
        else:
            image = self.transform(image)
        
        return image, label, image_id
    
    def get_class_weights(self) -> torch.Tensor:
        """Calculer les poids des classes pour gérer le déséquilibre."""
        num_classes = len(self.CLASS_NAMES)
        class_counts = np.zeros(num_classes)
        for label, count in self.df['label'].value_counts().items():
            class_counts[label] = count
        class_counts = np.maximum(class_counts, 1)  # avoid division by zero
        total = len(self.df)
        weights = total / (num_classes * class_counts)
        return torch.FloatTensor(weights)
    
    def get_sample_weights(self) -> np.ndarray:
        """Poids par échantillon pour WeightedRandomSampler."""
        class_weights = self.get_class_weights().numpy()
        sample_weights = np.array([class_weights[label] for label in self.df['label']])
        return sample_weights

################################# #####################################

def get_data_loaders(
    root_dir: str,
    batch_size: int = 32,
    image_size: int = 224,
    num_workers: int = 4,
    use_weighted_sampling: bool = True
) -> Dict[str, DataLoader]:
    """
    Créer les DataLoaders pour train, val et test.
    
    Args:
        root_dir: Chemin vers le dataset ISIC
        batch_size: Taille des batches
        image_size: Taille des images
        num_workers: Nombre de workers pour le chargement
        use_weighted_sampling: Utiliser le sampling pondéré pour train
        
    Returns:
        Dict avec les DataLoaders 'train', 'val', 'test'
    """
    loaders = {}
    pin = torch.cuda.is_available()

    for split in ['train', 'val', 'test']:
        dataset = ISICDataset(
            root_dir=root_dir,
            split=split,
            image_size=image_size
        )

        if split == 'train' and use_weighted_sampling:
            # Weighted sampling pour gérer le déséquilibre des classes
            sample_weights = dataset.get_sample_weights()
            sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(sample_weights),
                replacement=True
            )
            loaders[split] = DataLoader(
                dataset,
                batch_size=batch_size,
                sampler=sampler,
                num_workers=num_workers,
                pin_memory=pin
            )
        else:
            loaders[split] = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=(split == 'train'),
                num_workers=num_workers,
                pin_memory=pin
            )
    
    return loaders


def download_isic_instructions():
    """Afficher les instructions pour télécharger le dataset ISIC."""
    instructions = """
    ╔══════════════════════════════════════════════════════════════════╗
    ║              INSTRUCTIONS DE TÉLÉCHARGEMENT ISIC 2019            ║
    ╠══════════════════════════════════════════════════════════════════╣
    ║                                                                  ║
    ║  1. Visitez: https://challenge.isic-archive.com/data/            ║
    ║                                                                  ║
    ║  2. Téléchargez les fichiers suivants pour ISIC 2019:            ║
    ║     - ISIC_2019_Training_Input.zip (~9 GB)                       ║
    ║     - ISIC_2019_Training_GroundTruth.csv                         ║
    ║     - ISIC_2019_Training_Metadata.csv (optionnel)                ║
    ║                                                                  ║
    ║  3. Extrayez les fichiers dans la structure suivante:            ║
    ║                                                                  ║
    ║     data/ISIC2019/                                               ║
    ║     ├── ISIC_2019_Training_Input/                                ║
    ║     │   ├── ISIC_0000000.jpg                                     ║
    ║     │   ├── ISIC_0000001.jpg                                     ║
    ║     │   └── ...                                                  ║
    ║     ├── ISIC_2019_Training_GroundTruth.csv                       ║
    ║     └── ISIC_2019_Training_Metadata.csv                          ║
    ║                                                                  ║
    ║  Alternative: Utilisez l'API ISIC ou Kaggle                      ║
    ║  - Kaggle: kaggle competitions download -c siim-isic-melanoma    ║
    ║                                                                  ║
    ╚══════════════════════════════════════════════════════════════════╝
    """
    print(instructions)


# Test du module
if __name__ == "__main__":
    download_isic_instructions()
    
    # Test avec données fictives (si disponibles)
    print("\n[TEST] Vérification de la structure du module...")
    print("✓ ISICDataset classe définie")
    print("✓ get_data_loaders fonction définie")
    print("✓ Classes ISIC:", ISICDataset.CLASS_NAMES)
