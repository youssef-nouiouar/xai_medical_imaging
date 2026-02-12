"""
ISIC Dataset Loader
===================
Module pour charger et prétraiter le dataset ISIC 2019 pour la classification
des lésions cutanées.

Supporte deux modes:
1. Split automatique du Training set (par défaut si test officiel non disponible)
2. Utilisation du Test set officiel ISIC 2019 (recommandé pour publications)

Auteur: Youssef Nouiouar
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
    │   └── ...
    ├── ISIC_2019_Training_GroundTruth.csv
    ├── ISIC_2019_Test_Input/          (optionnel - pour test officiel)
    │   ├── ISIC_0000000.jpg
    │   └── ...
    └── ISIC_2019_Test_GroundTruth.csv (optionnel - pour test officiel)
    """
    
    # Mapping des classes ISIC 2019 (UNK excluded - 0 samples in dataset)
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
        use_albumentations: bool = True,
        use_official_test: bool = True,
        val_ratio: float = 0.1,
        random_state: int = 42
    ):
        """
        Args:
            root_dir: Chemin vers le dossier ISIC2019
            split: 'train', 'val', ou 'test'
            transform: Transformations torchvision (si use_albumentations=False)
            image_size: Taille des images en sortie
            use_albumentations: Utiliser albumentations pour l'augmentation
            use_official_test: Si True, utilise ISIC_2019_Test_Input pour le test
            val_ratio: Ratio du training set pour la validation
            random_state: Seed pour reproductibilité
        """
        self.root_dir = root_dir
        self.split = split
        self.image_size = image_size
        self.use_albumentations = use_albumentations
        self.use_official_test = use_official_test
        self.val_ratio = val_ratio
        self.random_state = random_state
        
        # Charger les labels
        self.df, self.image_dir = self._load_data()
        
        # Configurer les transformations
        if use_albumentations:
            self.transform = self._get_albumentations_transform()
        else:
            self.transform = transform or self._get_default_transform()
    
    def _check_official_test_available(self) -> bool:
        """Vérifier si le test set officiel est disponible."""
        test_csv = os.path.join(self.root_dir, 'ISIC_2019_Test_GroundTruth.csv')
        test_dir = os.path.join(self.root_dir, 'ISIC_2019_Test_Input')
        return os.path.exists(test_csv) and os.path.exists(test_dir)
    
    def _load_data(self) -> Tuple[pd.DataFrame, str]:
        """Charger et préparer les données."""
        
        # === CAS 1: Utiliser le test set officiel ===
        if self.use_official_test and self.split == 'test':
            if self._check_official_test_available():
                return self._load_official_test()
            else:
                print("Warning: Test set officiel non trouvé. Utilisation du split automatique.")
                self.use_official_test = False
        
        # === CAS 2: Split automatique du Training set ===
        return self._load_training_split()
    
    def _load_official_test(self) -> Tuple[pd.DataFrame, str]:
        """Charger le test set officiel ISIC 2019."""
        csv_path = os.path.join(self.root_dir, 'ISIC_2019_Test_GroundTruth.csv')
        image_dir = os.path.join(self.root_dir, 'ISIC_2019_Test_Input')
        
        print(f"Loading official Test set: {csv_path}")
        
        df = pd.read_csv(csv_path)
        
        # Drop UNK column if present
        if 'UNK' in df.columns:
            df = df.drop(columns=['UNK'])
        
        # Convertir one-hot encoding en labels
        label_columns = [col for col in df.columns if col != 'image']
        df['label'] = df[label_columns].values.argmax(axis=1)
        df['class_name'] = df['label'].map(lambda x: self.CLASS_NAMES[x])
        
        print(f"   {len(df)} test images loaded")
        
        return df.reset_index(drop=True), image_dir
    
    def _load_training_split(self) -> Tuple[pd.DataFrame, str]:
        """Charger et splitter le Training set."""
        csv_path = os.path.join(self.root_dir, 'ISIC_2019_Training_GroundTruth.csv')
        image_dir = os.path.join(self.root_dir, 'ISIC_2019_Training_Input')
        
        if not os.path.exists(csv_path):
            raise FileNotFoundError(
                f"CSV file not found: {csv_path}\n"
                "Download the dataset from: https://challenge.isic-archive.com/data/"
            )
        
        df = pd.read_csv(csv_path)
        
        # Drop UNK column if present (0 samples in ISIC 2019)
        if 'UNK' in df.columns:
            df = df.drop(columns=['UNK'])
        
        # Convertir one-hot encoding en labels
        label_columns = [col for col in df.columns if col != 'image']
        df['label'] = df[label_columns].values.argmax(axis=1)
        df['class_name'] = df['label'].map(lambda x: self.CLASS_NAMES[x])
        
        # === Split des données ===
        if self.use_official_test:
            # Mode: Train/Val split seulement (test = officiel)
            train_df, val_df = train_test_split(
                df, 
                test_size=self.val_ratio, 
                stratify=df['label'], 
                random_state=self.random_state
            )
            
            if self.split == 'train':
                result_df = train_df
            elif self.split == 'val':
                result_df = val_df
            else:
                # Ne devrait pas arriver si use_official_test=True
                result_df = val_df
        else:
            # Mode: Train/Val/Test split du training set
            train_df, temp_df = train_test_split(
                df, 
                test_size=0.2, 
                stratify=df['label'], 
                random_state=self.random_state
            )
            val_df, test_df = train_test_split(
                temp_df, 
                test_size=0.5, 
                stratify=temp_df['label'], 
                random_state=self.random_state
            )
            
            if self.split == 'train':
                result_df = train_df
            elif self.split == 'val':
                result_df = val_df
            else:
                result_df = test_df
        
        return result_df.reset_index(drop=True), image_dir
    
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
        image_path = os.path.join(self.image_dir, f'{image_id}.jpg')
        
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
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
    
    def get_class_distribution(self) -> pd.DataFrame:
        """Obtenir la distribution des classes."""
        dist = self.df['label'].value_counts().sort_index()
        dist_df = pd.DataFrame({
            'Class': [self.CLASS_NAMES[i] for i in dist.index],
            'Full Name': [self.CLASS_FULL_NAMES[self.CLASS_NAMES[i]] for i in dist.index],
            'Count': dist.values,
            'Percentage': (dist.values / len(self.df) * 100).round(2)
        })
        return dist_df


###############################################################################
# DATA LOADERS
###############################################################################

def get_data_loaders(
    root_dir: str,
    batch_size: int = 32,
    image_size: int = 224,
    num_workers: int = 4,
    use_weighted_sampling: bool = True,
    use_official_test: bool = True,
    val_ratio: float = 0.1
) -> Dict[str, DataLoader]:
    """
    Créer les DataLoaders pour train, val et test.
    
    Args:
        root_dir: Chemin vers le dataset ISIC
        batch_size: Taille des batches
        image_size: Taille des images
        num_workers: Nombre de workers pour le chargement
        use_weighted_sampling: Utiliser le sampling pondéré pour train
        use_official_test: Utiliser le test set officiel ISIC 2019
        val_ratio: Ratio du training set pour validation (si use_official_test=True)
        
    Returns:
        Dict avec les DataLoaders 'train', 'val', 'test'
    """
    loaders = {}
    pin = torch.cuda.is_available()
    
    print("\n" + "="*60)
    print("LOADING ISIC 2019 DATASET")
    print("="*60)
    
    # Check if official test is available
    test_csv = os.path.join(root_dir, 'ISIC_2019_Test_GroundTruth.csv')
    test_dir = os.path.join(root_dir, 'ISIC_2019_Test_Input')
    official_test_available = os.path.exists(test_csv) and os.path.exists(test_dir)
    
    if use_official_test and not official_test_available:
        print("Warning: Official test set not found. Using automatic split.")
        use_official_test = False
    
    if use_official_test:
        print(f"Mode: Train/Val split + Official Test")
        print(f"   - Train: {100-val_ratio*100:.0f}% of Training set")
        print(f"   - Val: {val_ratio*100:.0f}% of Training set")
        print(f"   - Test: ISIC_2019_Test_Input (official)")
    else:
        print(f"Mode: Automatic split 80/10/10")
        print(f"   - Train: 80% of Training set")
        print(f"   - Val: 10% of Training set")
        print(f"   - Test: 10% of Training set")
    print("-"*60)

    for split in ['train', 'val', 'test']:
        dataset = ISICDataset(
            root_dir=root_dir,
            split=split,
            image_size=image_size,
            use_official_test=use_official_test,
            val_ratio=val_ratio
        )
        
        print(f"\n{split.upper():6s}: {len(dataset):,} images")
        
        # Afficher la distribution pour le premier split
        if split == 'train':
            dist = dataset.get_class_distribution()
            print("\n   Class distribution:")
            for _, row in dist.iterrows():
                print(f"   - {row['Class']:5s}: {row['Count']:6,} ({row['Percentage']:5.1f}%)")

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
    
    print("\n" + "="*60)
    print("DataLoaders created successfully!")
    print("="*60 + "\n")
    
    return loaders


def download_isic_instructions():
    """Afficher les instructions pour télécharger le dataset ISIC."""
    instructions = """
    ╔══════════════════════════════════════════════════════════════════╗
    ║              INSTRUCTIONS DE TELECHARGEMENT ISIC 2019            ║
    ╠══════════════════════════════════════════════════════════════════╣
    ║                                                                  ║
    ║  1. Visitez: https://challenge.isic-archive.com/data/            ║
    ║                                                                  ║
    ║  2. Telechargez les fichiers suivants pour ISIC 2019:            ║
    ║                                                                  ║
    ║     OBLIGATOIRES:                                                ║
    ║     - ISIC_2019_Training_Input.zip (~9 GB, 25,331 images)        ║
    ║     - ISIC_2019_Training_GroundTruth.csv                         ║
    ║                                                                  ║
    ║     RECOMMANDES (pour test officiel):                            ║
    ║     - ISIC_2019_Test_Input.zip (~2 GB, 8,238 images)             ║
    ║     - ISIC_2019_Test_GroundTruth.csv                             ║
    ║                                                                  ║
    ║     OPTIONNEL:                                                   ║
    ║     - ISIC_2019_Training_Metadata.csv                            ║
    ║                                                                  ║
    ║  3. Extrayez les fichiers dans la structure suivante:            ║
    ║                                                                  ║
    ║     data/ISIC2019/                                               ║
    ║     ├── ISIC_2019_Training_Input/                                ║
    ║     │   ├── ISIC_0024306.jpg                                     ║
    ║     │   └── ... (25,331 images)                                  ║
    ║     ├── ISIC_2019_Training_GroundTruth.csv                       ║
    ║     ├── ISIC_2019_Test_Input/           <- RECOMMANDE            ║
    ║     │   ├── ISIC_0024307.jpg                                     ║
    ║     │   └── ... (8,238 images)                                   ║
    ║     └── ISIC_2019_Test_GroundTruth.csv  <- RECOMMANDE            ║
    ║                                                                  ║
    ╠══════════════════════════════════════════════════════════════════╣
    ║  UTILISATION:                                                    ║
    ║                                                                  ║
    ║  # Avec test set officiel (recommande pour publications)         ║
    ║  loaders = get_data_loaders(                                     ║
    ║      root_dir='data/ISIC2019',                                   ║
    ║      use_official_test=True   # <- utilise Test_Input            ║
    ║  )                                                               ║
    ║                                                                  ║
    ║  # Sans test set officiel (split automatique)                    ║
    ║  loaders = get_data_loaders(                                     ║
    ║      root_dir='data/ISIC2019',                                   ║
    ║      use_official_test=False  # <- split du Training             ║
    ║  )                                                               ║
    ║                                                                  ║
    ╚══════════════════════════════════════════════════════════════════╝
    """
    print(instructions)


###############################################################################
# TEST DU MODULE
###############################################################################

if __name__ == "__main__":
    download_isic_instructions()
    
    print("\n[TEST] Module structure verification...")
    print("- ISICDataset class defined")
    print("- get_data_loaders function defined")
    print(f"- ISIC Classes: {ISICDataset.CLASS_NAMES}")
    print(f"- Number of classes: {len(ISICDataset.CLASS_NAMES)}")
    
    # Test si le dataset existe
    test_path = "data/ISIC2019"
    if os.path.exists(test_path):
        print(f"\n[TEST] Dataset found in {test_path}")
        try:
            loaders = get_data_loaders(
                root_dir=test_path,
                batch_size=4,
                use_official_test=True
            )
            
            # Test d'un batch
            images, labels, ids = next(iter(loaders['train']))
            print(f"\n[TEST] Batch loaded successfully:")
            print(f"   - Images shape: {images.shape}")
            print(f"   - Labels: {labels.tolist()}")
            print(f"   - IDs: {ids[:2]}...")
            
        except Exception as e:
            print(f"\n[ERROR] {e}")
    else:
        print(f"\n[INFO] Dataset not found in {test_path}")
        print("       Follow the instructions above to download the dataset.")
