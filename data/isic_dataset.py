"""
ISIC 2019 Dataset with Anti-Overfitting Augmentation Support
=============================================================
Supports multiple augmentation strengths based on Gayatri & Aarthy (2023):
"Reduction of overfitting on the highly imbalanced ISIC-2019 skin dataset"
"""

import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Normalisation ImageNet
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Class names for ISIC 2019
CLASS_NAMES = ['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC']


class ISICDataset(Dataset):
    """
    ISIC 2019 Dataset with configurable augmentation strength.
    
    Args:
        root_dir: Path to ISIC2019 data directory
        split: 'train', 'val', or 'test'
        transform: Custom transform (overrides augmentation_strength)
        image_size: Target image size (default: 224)
        use_albumentations: Use albumentations library (default: True)
        use_official_test: Use official test set (default: False)
        val_ratio: Validation split ratio (default: 0.1)
        random_state: Random seed (default: 42)
        augmentation_strength: 'light', 'medium', or 'strong' (default: 'medium')
                              'strong' is recommended for anti-overfitting
    """
    
    CLASS_NAMES = CLASS_NAMES
    
    def __init__(self, root_dir, split='train', transform=None, image_size=224,
                 use_albumentations=True, use_official_test=False, val_ratio=0.1,
                 test_ratio=0.1, random_state=42, augmentation_strength='medium'):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.image_size = image_size
        self.use_albumentations = use_albumentations
        self.use_official_test = use_official_test
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.random_state = random_state
        self.augmentation_strength = augmentation_strength

        if self.use_official_test:
            self.image_dir_train = os.path.join(root_dir, 'ISIC_2019_Training_Input')
            self.labels_path_train = os.path.join(root_dir, 'ISIC_2019_Training_GroundTruth.csv')
            self.image_dir_test = os.path.join(root_dir, 'ISIC_2019_Test_Input')
            self.labels_path_test = os.path.join(root_dir, 'ISIC_2019_Test_GroundTruth.csv')
        else:
            self.image_dir = os.path.join(root_dir, 'ISIC_2019_Training_Input')
            self.labels_path = os.path.join(root_dir, 'ISIC_2019_Training_GroundTruth.csv')

        self._load_data()

        # Configurer les transformations
        if use_albumentations:
            self.transform = self._get_albumentations_transform()
        else:
            self.transform = transform or self._get_default_transform()

    def _load_data(self):
        if self.use_official_test:
            train_df = pd.read_csv(self.labels_path_train)
            test_df_official = pd.read_csv(self.labels_path_test)

            # Convert labels to numerical format if not already
            self.class_names = [col for col in train_df.columns if col not in ['image', 'UNK']]

            if self.split == 'test':
                self.data = test_df_official
                self.image_folder = self.image_dir_test
                # Remove UNK column if it exists in test_df_official
                if 'UNK' in self.data.columns:
                    self.data = self.data.drop(columns=['UNK'])
            else: # 'train' or 'val' split from the training data
                # Drop UNK column from training data
                if 'UNK' in train_df.columns:
                    train_df = train_df.drop(columns=['UNK'])

                train_indices, val_indices = train_test_split(
                    train_df.index, test_size=self.val_ratio, random_state=self.random_state,
                    stratify=train_df[self.class_names].values.argmax(axis=1)
                )

                if self.split == 'train':
                    self.data = train_df.loc[train_indices].reset_index(drop=True)
                elif self.split == 'val':
                    self.data = train_df.loc[val_indices].reset_index(drop=True)
                self.image_folder = self.image_dir_train

        else: # Standard train/val/test split from training data only
            df_full = pd.read_csv(self.labels_path)
            self.class_names = [col for col in df_full.columns if col not in ['image', 'UNK']]
            if 'UNK' in df_full.columns:
                df_full = df_full.drop(columns=['UNK'])

            train_val_df, test_df = train_test_split(
                df_full, test_size=self.test_ratio, random_state=self.random_state,
                stratify=df_full[self.class_names].values.argmax(axis=1)
            )
            val_size = self.val_ratio / (1 - self.test_ratio)
            train_df, val_df = train_test_split(
                train_val_df, test_size=val_size, random_state=self.random_state,
                stratify=train_val_df[self.class_names].values.argmax(axis=1)
            )

            if self.split == 'train':
                self.data = train_df.reset_index(drop=True)
            elif self.split == 'val':
                self.data = val_df.reset_index(drop=True)
            elif self.split == 'test':
                self.data = test_df.reset_index(drop=True)
            self.image_folder = self.image_dir

        # Convert one-hot encoded labels to single label index
        self.labels = self.data[self.class_names].values.argmax(axis=1)

    def get_class_weights(self):
        class_counts = Counter(self.labels)
        total_samples = len(self.labels)
        num_classes = len(self.class_names)
        # Calculate inverse frequency weights
        weights = torch.tensor([total_samples / (num_classes * class_counts[i]) for i in range(num_classes)], dtype=torch.float)
        return weights

    def _get_default_transform(self):
        return transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
        ])

    def _get_albumentations_transform(self):
        """
        Get Albumentations transforms based on split and augmentation_strength.
        
        Augmentation strengths (for training):
        - 'light': Basic augmentations (flips, small rotation)
        - 'medium': Moderate augmentations (default)
        - 'strong': Aggressive augmentations for anti-overfitting (Gayatri & Aarthy 2023)
        """
        if self.split != 'train':
            # Validation and Test: no augmentation, just resize and normalize
            return A.Compose([
                A.Resize(self.image_size, self.image_size),
                A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                ToTensorV2()
            ])
        
        # Training transforms based on strength
        if self.augmentation_strength == 'light':
            return A.Compose([
                A.RandomResizedCrop(
                    size=(self.image_size, self.image_size),
                    scale=(0.85, 1.0), ratio=(0.9, 1.1), p=0.5
                ),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Rotate(limit=15, p=0.5),
                A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                ToTensorV2()
            ])
        
        elif self.augmentation_strength == 'medium':
            return A.Compose([
                A.RandomResizedCrop(
                    size=(self.image_size, self.image_size),
                    scale=(0.8, 1.0), ratio=(0.75, 1.333), p=0.5
                ),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.2),
                A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.2),
                A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                ToTensorV2()
            ])
        
        else:  # 'strong' - recommended for anti-overfitting (Gayatri & Aarthy 2023)
            return A.Compose([
                # Geometric transforms (crucial for dermoscopy)
                A.RandomResizedCrop(
                    size=(self.image_size, self.image_size),
                    scale=(0.6, 1.0), ratio=(0.75, 1.33), p=1.0
                ),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.15, scale_limit=0.2, rotate_limit=45,
                    border_mode=0, p=0.7
                ),
                
                # Color augmentations (skin lesion variability)
                A.OneOf([
                    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3),
                    A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20),
                    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                ], p=0.7),
                
                # Dermoscopy-specific: simulate occlusions (hair, artifacts)
                A.OneOf([
                    A.CoarseDropout(
                        num_holes_range=(4, 12),
                        hole_height_range=(8, 16), hole_width_range=(8, 16),
                        fill=0, p=1.0
                    ),
                    A.GridDropout(ratio=0.2, p=1.0),
                ], p=0.3),
                
                # Noise and blur (camera variability)
                A.OneOf([
                    A.GaussianBlur(blur_limit=(3, 7)),
                    A.MotionBlur(blur_limit=5),
                    A.MedianBlur(blur_limit=5),
                ], p=0.2),
                A.GaussNoise(var_limit=(10, 50), p=0.3),
                
                # Elastic transforms
                A.OneOf([
                    A.ElasticTransform(alpha=120, sigma=6.0, p=1.0),
                    A.GridDistortion(p=1.0),
                    A.OpticalDistortion(distort_limit=0.5, shift_limit=0.5, p=1.0),
                ], p=0.2),
                
                A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                ToTensorV2()
            ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data.iloc[idx]['image'] + '.jpg'
        img_path = os.path.join(self.image_folder, img_name)
        image = Image.open(img_path).convert('RGB')

        # Explicitly resize image to IMAGE_SIZE before Albumentations (robustness against inconsistent input sizes)
        image = image.resize((self.image_size, self.image_size), Image.LANCZOS)

        # Apply Albumentations transform if configured
        if self.use_albumentations:
            image = np.array(image) # Albumentations works with NumPy arrays
            image = self.transform(image=image)['image']
        else:
            image = self.transform(image)

        label = self.labels[idx]
        return image, label, img_name

def get_data_loaders(root_dir, batch_size=32, image_size=224, num_workers=4,
                     use_weighted_sampling=False, use_official_test=False, val_ratio=0.1):
    """
    Creates and returns data loaders for ISIC 2019 dataset.

    Args:
        root_dir (str): Root directory containing the ISIC2019 dataset.
        batch_size (int): Batch size for data loaders.
        image_size (int): Desired image size (height and width).
        num_workers (int): Number of worker processes for data loading.
        use_weighted_sampling (bool): Whether to use weighted sampling for imbalanced classes.
        use_official_test (bool): If True, uses the official ISIC 2019 test set.
                                  Otherwise, splits the training set into train/val/test.
        val_ratio (float): Proportion of the dataset to use for validation.

    Returns:
        dict: A dictionary containing 'train', 'val', and 'test' DataLoader objects.
    """
    print("="*60)
    print("LOADING ISIC 2019 DATASET")
    print("="*60)

    loaders = {}

    if use_official_test:
        print("Mode: Train/Val split + Official Test")
        print(f"   - Train: {int((1-val_ratio)*100)}% of Training set")
        print(f"   - Val: {int(val_ratio*100)}% of Training set")
        print("   - Test: ISIC_2019_Test_Input (official)")
    else:
        print("Mode: Train/Val/Test split from Training set")
        print(f"   - Train: {int((1-0.2)*(1-val_ratio)*100)}% of Training set")
        print(f"   - Val: {int((1-0.2)*val_ratio*100)}% of Training set")
        print(f"   - Test: {int(0.2*100)}% of Training set")

    print("-"*60)

    for split in ['train', 'val', 'test']:
        dataset = ISICDataset(
            root_dir=root_dir,
            split=split,
            image_size=image_size,
            use_albumentations=True,
            use_official_test=use_official_test,
            val_ratio=val_ratio
        )

        if split == 'train' and use_weighted_sampling:
            class_weights = dataset.get_class_weights()
            sampler = torch.utils.data.WeightedRandomSampler(
                weights=class_weights[dataset.labels],
                num_samples=len(dataset.labels),
                replacement=True
            )
            loaders[split] = torch.utils.data.DataLoader(
                dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers
            )
        else:
            loaders[split] = torch.utils.data.DataLoader(
                dataset, batch_size=batch_size, shuffle=(split == 'train'),
                num_workers=num_workers, drop_last=(split == 'train')
            )
        print(f"   - {split.capitalize()} loader: {len(dataset)} images in {len(loaders[split])} batches")

    print("="*60)
    return loaders