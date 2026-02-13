# ============================================
# ANTI-OVERFITTING TRAINING FOR ISIC-2019
# ============================================
# Based on:
# 1. Gayatri & Aarthy (2023) - "Reduction of overfitting on ISIC-2019"
# 2. State-of-the-art regularization techniques
#
# Copy and paste this code into your Colab notebook
# ============================================

import os
import sys
import numpy as np
import random
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm.notebook import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# ============================================
# 1. ENHANCED DATA AUGMENTATION
# ============================================
# Strong augmentation as recommended by Gayatri & Aarthy

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def get_strong_train_transform(image_size=224):
    """
    Strong dermoscopy-specific augmentation to reduce overfitting.
    Implements techniques from Gayatri & Aarthy (2023) paper.
    """
    return A.Compose([
        # Geometric transforms (crucial for dermoscopy)
        A.RandomResizedCrop(height=image_size, width=image_size, 
                           scale=(0.6, 1.0), ratio=(0.75, 1.33), p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.15, scale_limit=0.2, rotate_limit=45, 
                          border_mode=0, p=0.7),
        
        # Color augmentations (important for skin lesion variability)
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        ], p=0.7),
        
        # Dermoscopy-specific: simulate occlusions (hair, artifacts)
        A.OneOf([
            A.CoarseDropout(max_holes=12, max_height=16, max_width=16, 
                           min_holes=4, fill_value=0, p=1.0),
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
            A.ElasticTransform(alpha=120, sigma=120 * 0.05, p=1.0),
            A.GridDistortion(p=1.0),
        ], p=0.2),
        
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2()
    ])

def get_val_transform(image_size=224):
    """Standard validation transform."""
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2()
    ])

# ============================================
# 2. MIXUP AND CUTMIX REGULARIZATION
# ============================================

class MixupCutmix:
    """Mixup and CutMix regularization for preventing overfitting."""
    
    def __init__(self, mixup_alpha=0.4, cutmix_alpha=1.0, prob=0.5, switch_prob=0.5):
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.prob = prob
        self.switch_prob = switch_prob
    
    def __call__(self, images, labels):
        if random.random() > self.prob:
            return images, labels, labels, 1.0
        
        batch_size = images.size(0)
        indices = torch.randperm(batch_size, device=images.device)
        
        if random.random() > self.switch_prob:
            # Mixup
            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
            mixed_images = lam * images + (1 - lam) * images[indices]
        else:
            # CutMix
            lam = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
            H, W = images.size(2), images.size(3)
            cut_rat = np.sqrt(1.0 - lam)
            cut_h, cut_w = int(H * cut_rat), int(W * cut_rat)
            
            cy, cx = random.randint(0, H), random.randint(0, W)
            y1, y2 = np.clip(cy - cut_h // 2, 0, H), np.clip(cy + cut_h // 2, 0, H)
            x1, x2 = np.clip(cx - cut_w // 2, 0, W), np.clip(cx + cut_w // 2, 0, W)
            
            mixed_images = images.clone()
            mixed_images[:, :, y1:y2, x1:x2] = images[indices, :, y1:y2, x1:x2]
            lam = 1 - ((y2 - y1) * (x2 - x1)) / (H * W)
        
        return mixed_images, labels, labels[indices], lam
    
    @staticmethod
    def mixup_criterion(criterion, pred, labels_a, labels_b, lam):
        return lam * criterion(pred, labels_a) + (1 - lam) * criterion(pred, labels_b)

# ============================================
# 3. FOCAL LOSS FOR CLASS IMBALANCE
# ============================================

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance in ISIC-2019."""
    
    def __init__(self, alpha=None, gamma=2.0, label_smoothing=0.1):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_weight = (1 - pt) ** self.gamma
        loss = focal_weight * ce_loss
        
        if self.alpha is not None:
            alpha_t = self.alpha.to(targets.device)[targets]
            loss = alpha_t * loss
        
        return loss.mean()

# ============================================
# 4. LEARNING RATE SCHEDULER WITH WARMUP
# ============================================

class LinearWarmupCosineAnnealingLR:
    """LR scheduler with warmup followed by cosine annealing."""
    
    def __init__(self, optimizer, warmup_epochs, max_epochs, warmup_start_lr=1e-6, eta_min=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.warmup_start_lr = warmup_start_lr
        self.eta_min = eta_min
        self.base_lrs = [pg['lr'] for pg in optimizer.param_groups]
        self.current_epoch = 0
    
    def step(self):
        self.current_epoch += 1
        
        if self.current_epoch <= self.warmup_epochs:
            alpha = self.current_epoch / self.warmup_epochs
            for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
                pg['lr'] = self.warmup_start_lr + alpha * (base_lr - self.warmup_start_lr)
        else:
            progress = (self.current_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)
            for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
                pg['lr'] = self.eta_min + 0.5 * (base_lr - self.eta_min) * (1 + np.cos(np.pi * progress))
    
    def get_lr(self):
        return [pg['lr'] for pg in self.optimizer.param_groups]

# ============================================
# 5. EXPONENTIAL MOVING AVERAGE
# ============================================

class EMA:
    """Exponential Moving Average of model weights."""
    
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name].mul_(self.decay).add_(param.data, alpha=1-self.decay)
    
    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])
    
    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.backup[name])
        self.backup = {}

# ============================================
# 6. MODEL-SPECIFIC CONFIGURATIONS
# ============================================

def get_anti_overfit_config(model_name):
    """Get anti-overfitting configuration for each model."""
    configs = {
        'resnet50': {
            'lr': 3e-4,
            'backbone_lr': 3e-5,
            'batch_size': 32,
            'dropout_rate': 0.4,
            'weight_decay': 0.05,
            'warmup_epochs': 5,
            'mixup_alpha': 0.4,
            'cutmix_alpha': 1.0,
            'mixup_prob': 0.5,
            'focal_gamma': 2.0,
            'label_smoothing': 0.1,
            'ema_decay': 0.999,
            'gradient_clip': 1.0,
        },
        'efficientnet_b4': {
            'lr': 2e-4,
            'backbone_lr': 2e-5,
            'batch_size': 24,
            'dropout_rate': 0.4,
            'weight_decay': 0.02,
            'warmup_epochs': 5,
            'mixup_alpha': 0.4,
            'cutmix_alpha': 1.0,
            'mixup_prob': 0.5,
            'focal_gamma': 2.0,
            'label_smoothing': 0.1,
            'ema_decay': 0.999,
            'gradient_clip': 1.0,
        },
        'convnext_base': {
            'lr': 1e-4,
            'backbone_lr': 1e-5,
            'batch_size': 16,
            'dropout_rate': 0.3,
            'weight_decay': 0.05,
            'warmup_epochs': 5,
            'mixup_alpha': 0.4,
            'cutmix_alpha': 1.0,
            'mixup_prob': 0.5,
            'focal_gamma': 2.0,
            'label_smoothing': 0.1,
            'ema_decay': 0.999,
            'gradient_clip': 1.0,
        },
        'vit_base': {
            'lr': 1e-5,
            'backbone_lr': 1e-6,
            'batch_size': 16,
            'dropout_rate': 0.1,
            'weight_decay': 0.1,
            'warmup_epochs': 10,
            'mixup_alpha': 0.4,
            'cutmix_alpha': 1.0,
            'mixup_prob': 0.5,
            'focal_gamma': 2.0,
            'label_smoothing': 0.1,
            'ema_decay': 0.999,
            'gradient_clip': 1.0,
        },
        'deit_base': {
            'lr': 1e-5,
            'backbone_lr': 1e-6,
            'batch_size': 16,
            'dropout_rate': 0.0,
            'weight_decay': 0.05,
            'warmup_epochs': 10,
            'mixup_alpha': 0.4,
            'cutmix_alpha': 1.0,
            'mixup_prob': 0.5,
            'focal_gamma': 2.0,
            'label_smoothing': 0.1,
            'ema_decay': 0.999,
            'gradient_clip': 1.0,
        },
        'swin_base': {
            'lr': 2e-5,
            'backbone_lr': 2e-6,
            'batch_size': 16,
            'dropout_rate': 0.2,
            'weight_decay': 0.05,
            'warmup_epochs': 8,
            'mixup_alpha': 0.4,
            'cutmix_alpha': 1.0,
            'mixup_prob': 0.5,
            'focal_gamma': 2.0,
            'label_smoothing': 0.1,
            'ema_decay': 0.999,
            'gradient_clip': 1.0,
        },
    }
    return configs.get(model_name, configs['resnet50'])

# ============================================
# 7. MAIN TRAINING FUNCTION WITH ANTI-OVERFITTING
# ============================================

def train_model_anti_overfit(model_name, model, train_loader, val_loader, 
                             epochs=50, patience=15, device='cuda'):
    """
    Train model with all anti-overfitting techniques.
    
    Techniques applied:
    - Strong data augmentation (dermoscopy-specific)
    - Mixup/CutMix regularization
    - Focal Loss with label smoothing
    - Learning rate warmup + cosine annealing
    - Exponential Moving Average (EMA)
    - Two-stage learning rates (backbone vs head)
    - Gradient clipping
    - Early stopping
    """
    config = get_anti_overfit_config(model_name)
    
    # Get class weights for imbalanced dataset
    class_weights = train_loader.dataset.get_class_weights().to(device)
    print(f"Class weights: {class_weights.cpu().numpy().round(2)}")
    
    # Setup criterion with Focal Loss
    criterion = FocalLoss(
        alpha=class_weights,
        gamma=config['focal_gamma'],
        label_smoothing=config['label_smoothing']
    )
    
    # Setup optimizer with two-stage learning rates
    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if any(x in name.lower() for x in ['head', 'fc', 'classifier']):
            head_params.append(param)
        else:
            backbone_params.append(param)
    
    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': config['backbone_lr']},
        {'params': head_params, 'lr': config['lr']}
    ], weight_decay=config['weight_decay'])
    
    # Setup scheduler with warmup
    scheduler = LinearWarmupCosineAnnealingLR(
        optimizer, 
        warmup_epochs=config['warmup_epochs'], 
        max_epochs=epochs
    )
    
    # Setup Mixup/CutMix
    mixup = MixupCutmix(
        mixup_alpha=config['mixup_alpha'],
        cutmix_alpha=config['cutmix_alpha'],
        prob=config['mixup_prob']
    )
    
    # Setup EMA
    ema = EMA(model, decay=config['ema_decay'])
    
    # Mixed precision
    scaler = GradScaler()
    
    # Tracking
    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': [],
        'val_f1': [], 'val_auc': [], 'lr': []
    }
    best_val_f1 = 0.0
    best_val_acc = 0.0
    patience_counter = 0
    
    print(f"\n{'='*70}")
    print(f"Training {model_name} with Anti-Overfitting Techniques")
    print(f"{'='*70}")
    print(f"Head LR: {config['lr']}, Backbone LR: {config['backbone_lr']}")
    print(f"Warmup: {config['warmup_epochs']} epochs")
    print(f"Mixup/CutMix prob: {config['mixup_prob']}")
    print(f"Focal Loss gamma: {config['focal_gamma']}")
    print(f"Label smoothing: {config['label_smoothing']}")
    print(f"Weight decay: {config['weight_decay']}")
    print(f"EMA decay: {config['ema_decay']}")
    print(f"{'='*70}\n")
    
    for epoch in range(1, epochs + 1):
        # ========== TRAINING ==========
        model.train()
        train_loss = 0.0
        all_preds, all_labels = [], []
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
        for images, labels, _ in pbar:
            images, labels = images.to(device), labels.to(device)
            
            # Apply Mixup/CutMix
            mixed_images, labels_a, labels_b, lam = mixup(images, labels)
            
            optimizer.zero_grad()
            
            with autocast():
                outputs = model(mixed_images)
                if lam < 1.0:
                    loss = MixupCutmix.mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
                else:
                    loss = criterion(outputs, labels_a)
            
            scaler.scale(loss).backward()
            
            # Gradient clipping
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip'])
            
            scaler.step(optimizer)
            scaler.update()
            
            # Update EMA
            ema.update()
            
            train_loss += loss.item()
            all_preds.extend(outputs.argmax(dim=1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        train_loss /= len(train_loader)
        train_acc = accuracy_score(all_labels, all_preds)
        
        # ========== VALIDATION (using EMA weights) ==========
        ema.apply_shadow()
        model.eval()
        
        val_loss = 0.0
        all_preds, all_labels, all_probs = [], [], []
        
        with torch.no_grad():
            for images, labels, _ in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                all_probs.extend(F.softmax(outputs, dim=1).cpu().numpy())
                all_preds.extend(outputs.argmax(dim=1).cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        ema.restore()
        
        val_loss /= len(val_loader)
        val_acc = accuracy_score(all_labels, all_preds)
        val_f1 = f1_score(all_labels, all_preds, average='macro')
        
        try:
            val_auc = roc_auc_score(all_labels, np.array(all_probs), multi_class='ovr', average='macro')
        except:
            val_auc = 0.0
        
        # Update scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Log
        print(f"Epoch {epoch:3d}/{epochs} | "
              f"Train: {train_loss:.4f}/{train_acc:.4f} | "
              f"Val: {val_loss:.4f}/{val_acc:.4f} F1:{val_f1:.4f} AUC:{val_auc:.4f} | "
              f"LR: {current_lr:.2e}")
        
        # History
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        history['val_auc'].append(val_auc)
        history['lr'].append(current_lr)
        
        # Check improvement (use F1 for imbalanced data)
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_val_acc = val_acc
            patience_counter = 0
            
            # Save best model
            ema.apply_shadow()
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_acc': val_acc,
                'val_f1': val_f1,
                'val_auc': val_auc,
                'config': config
            }, f'{SAVE_DIR}/{model_name}_best.pth')
            ema.restore()
            
            print(f"         ★ New best model! F1: {best_val_f1:.4f}")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch}")
            break
    
    return history, best_val_acc, best_val_f1

# ============================================
# 8. CUSTOM DATASET WITH ENHANCED AUGMENTATION
# ============================================

class ISICDatasetEnhanced(Dataset):
    """ISIC Dataset with enhanced augmentation for anti-overfitting."""
    
    def __init__(self, root_dir, split='train', image_size=224, val_ratio=0.1):
        from PIL import Image
        import pandas as pd
        from sklearn.model_selection import train_test_split
        from collections import Counter
        
        self.root_dir = root_dir
        self.split = split
        self.image_size = image_size
        
        # Load data
        train_df = pd.read_csv(os.path.join(root_dir, 'ISIC_2019_Training_GroundTruth.csv'))
        self.class_names = [col for col in train_df.columns if col not in ['image', 'UNK']]
        if 'UNK' in train_df.columns:
            train_df = train_df.drop(columns=['UNK'])
        
        if split == 'test':
            test_df = pd.read_csv(os.path.join(root_dir, 'ISIC_2019_Test_GroundTruth.csv'))
            if 'UNK' in test_df.columns:
                test_df = test_df.drop(columns=['UNK'])
            self.data = test_df
            self.image_folder = os.path.join(root_dir, 'ISIC_2019_Test_Input')
        else:
            train_indices, val_indices = train_test_split(
                train_df.index, test_size=val_ratio, random_state=42,
                stratify=train_df[self.class_names].values.argmax(axis=1)
            )
            if split == 'train':
                self.data = train_df.loc[train_indices].reset_index(drop=True)
            else:
                self.data = train_df.loc[val_indices].reset_index(drop=True)
            self.image_folder = os.path.join(root_dir, 'ISIC_2019_Training_Input')
        
        self.labels = self.data[self.class_names].values.argmax(axis=1)
        
        # Set transforms
        if split == 'train':
            self.transform = get_strong_train_transform(image_size)
        else:
            self.transform = get_val_transform(image_size)
    
    def get_class_weights(self):
        from collections import Counter
        class_counts = Counter(self.labels)
        total = len(self.labels)
        num_classes = len(self.class_names)
        weights = torch.tensor([total / (num_classes * class_counts[i]) for i in range(num_classes)], dtype=torch.float)
        return weights
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        from PIL import Image
        img_name = self.data.iloc[idx]['image'] + '.jpg'
        img_path = os.path.join(self.image_folder, img_name)
        image = Image.open(img_path).convert('RGB')
        image = image.resize((self.image_size, self.image_size), Image.LANCZOS)
        image = np.array(image)
        image = self.transform(image=image)['image']
        return image, self.labels[idx], img_name

print("✓ Anti-overfitting training module loaded!")
print("\nTechniques available:")
print("  • Strong dermoscopy-specific data augmentation")
print("  • Mixup & CutMix regularization")
print("  • Focal Loss with label smoothing")
print("  • Learning rate warmup + cosine annealing")
print("  • Exponential Moving Average (EMA)")
print("  • Two-stage learning rates (backbone vs head)")
print("  • Gradient clipping")
print("  • Early stopping")
