"""
Anti-Overfitting Strategies for ISIC-2019 Skin Lesion Classification
=====================================================================
Combines techniques from:
1. Gayatri & Aarthy (2023) - "Reduction of overfitting on the highly imbalanced ISIC-2019 skin dataset"
2. State-of-the-art regularization methods

This module provides:
- Enhanced data augmentation (dermoscopy-specific)
- Mixup / CutMix regularization
- Focal Loss for class imbalance
- Stochastic Depth for deep networks
- Model wrapper with dropout injection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Tuple, Optional
import random

# ============================================
# 1. ENHANCED DATA AUGMENTATION
# ============================================
# Based on Gayatri & Aarthy paper recommendations
# + dermoscopy-specific augmentations

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_anti_overfit_train_transform(image_size: int = 224, strength: str = 'strong'):
    """
    Enhanced training augmentations to reduce overfitting.
    
    Args:
        image_size: Target image size
        strength: 'light', 'medium', 'strong' augmentation intensity
        
    Based on:
    - Gayatri & Aarthy (2023): Aggressive augmentation for ISIC-2019
    - DermNet augmentation strategies for skin lesion classification
    """
    if strength == 'light':
        return A.Compose([
            A.RandomResizedCrop(height=image_size, width=image_size, scale=(0.85, 1.0)),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=15, p=0.5),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2()
        ])
    
    elif strength == 'medium':
        return A.Compose([
            A.RandomResizedCrop(height=image_size, width=image_size, scale=(0.75, 1.0)),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15, rotate_limit=30, p=0.6),
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
                A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10),
            ], p=0.5),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
            A.GaussNoise(var_limit=(5, 25), p=0.2),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2()
        ])
    
    else:  # 'strong' - recommended by Gayatri & Aarthy for ISIC-2019
        return A.Compose([
            # Geometric transforms (crucial for dermoscopy)
            A.RandomResizedCrop(height=image_size, width=image_size, scale=(0.6, 1.0), ratio=(0.75, 1.33)),
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
            
            # Dermoscopy-specific: hair and artifact simulation
            A.OneOf([
                A.CoarseDropout(max_holes=12, max_height=16, max_width=16, 
                               min_holes=4, fill_value=0, p=1.0),  # Simulate occlusions
                A.GridDropout(ratio=0.2, p=1.0),  # Grid-based dropout
            ], p=0.3),
            
            # Noise and blur (camera variability)
            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 7)),
                A.MotionBlur(blur_limit=5),
                A.MedianBlur(blur_limit=5),
            ], p=0.2),
            A.GaussNoise(var_limit=(10, 50), p=0.3),
            
            # Elastic and morphological transforms
            A.OneOf([
                A.ElasticTransform(alpha=120, sigma=120 * 0.05, p=1.0),
                A.GridDistortion(p=1.0),
                A.OpticalDistortion(distort_limit=0.5, shift_limit=0.5, p=1.0),
            ], p=0.2),
            
            # Normalization
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2()
        ])


def get_val_test_transform(image_size: int = 224):
    """Standard validation/test transform."""
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2()
    ])


# ============================================
# 2. MIXUP AND CUTMIX
# ============================================
# Powerful regularization techniques

class MixupCutmix:
    """
    Mixup (Zhang et al., 2018) and CutMix (Yun et al., 2019) regularization.
    
    These techniques create virtual training examples by mixing inputs and labels,
    which prevents memorization and improves generalization.
    """
    
    def __init__(self, mixup_alpha: float = 0.4, cutmix_alpha: float = 1.0,
                 prob: float = 0.5, switch_prob: float = 0.5, num_classes: int = 8):
        """
        Args:
            mixup_alpha: Beta distribution parameter for Mixup
            cutmix_alpha: Beta distribution parameter for CutMix
            prob: Probability of applying either Mixup or CutMix
            switch_prob: Probability of using CutMix over Mixup
            num_classes: Number of classes for label smoothing
        """
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.prob = prob
        self.switch_prob = switch_prob
        self.num_classes = num_classes
    
    def __call__(self, images: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """
        Apply Mixup or CutMix to batch.
        
        Returns:
            mixed_images: Mixed images
            labels_a: Original labels
            labels_b: Mixed labels
            lam: Mixing coefficient
        """
        if random.random() > self.prob:
            # No mixing, return original
            return images, labels, labels, 1.0
        
        batch_size = images.size(0)
        indices = torch.randperm(batch_size, device=images.device)
        
        if random.random() > self.switch_prob:
            # Apply Mixup
            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
            mixed_images = lam * images + (1 - lam) * images[indices]
        else:
            # Apply CutMix
            lam = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
            
            H, W = images.size(2), images.size(3)
            cut_rat = np.sqrt(1.0 - lam)
            cut_h = int(H * cut_rat)
            cut_w = int(W * cut_rat)
            
            # Uniform random center
            cy = random.randint(0, H)
            cx = random.randint(0, W)
            
            # Bounding box
            y1 = np.clip(cy - cut_h // 2, 0, H)
            y2 = np.clip(cy + cut_h // 2, 0, H)
            x1 = np.clip(cx - cut_w // 2, 0, W)
            x2 = np.clip(cx + cut_w // 2, 0, W)
            
            mixed_images = images.clone()
            mixed_images[:, :, y1:y2, x1:x2] = images[indices, :, y1:y2, x1:x2]
            
            # Adjust lambda to actual box area
            lam = 1 - ((y2 - y1) * (x2 - x1)) / (H * W)
        
        return mixed_images, labels, labels[indices], lam
    
    @staticmethod
    def mixup_criterion(criterion: nn.Module, pred: torch.Tensor, 
                        labels_a: torch.Tensor, labels_b: torch.Tensor, 
                        lam: float) -> torch.Tensor:
        """Mixed loss computation."""
        return lam * criterion(pred, labels_a) + (1 - lam) * criterion(pred, labels_b)


# ============================================
# 3. FOCAL LOSS FOR CLASS IMBALANCE
# ============================================
# From Lin et al., "Focal Loss for Dense Object Detection"
# Crucial for ISIC-2019's severe class imbalance

class FocalLoss(nn.Module):
    """
    Focal Loss - addresses class imbalance by down-weighting easy examples.
    
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    
    Recommended by Gayatri & Aarthy for ISIC-2019.
    """
    
    def __init__(self, alpha: Optional[torch.Tensor] = None, gamma: float = 2.0,
                 reduction: str = 'mean', label_smoothing: float = 0.1):
        """
        Args:
            alpha: Per-class weights (can be class frequencies inverse)
            gamma: Focusing parameter (0 = CE, higher = more focus on hard examples)
            reduction: 'mean', 'sum', or 'none'
            label_smoothing: Label smoothing factor
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Apply label smoothing
        num_classes = inputs.size(1)
        smooth_targets = torch.zeros_like(inputs).scatter_(
            1, targets.unsqueeze(1), 1.0
        )
        smooth_targets = smooth_targets * (1 - self.label_smoothing) + \
                        self.label_smoothing / num_classes
        
        # Compute focal loss
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_weight = (1 - pt) ** self.gamma
        
        loss = focal_weight * ce_loss
        
        # Apply class weights if provided
        if self.alpha is not None:
            if self.alpha.device != targets.device:
                self.alpha = self.alpha.to(targets.device)
            alpha_t = self.alpha[targets]
            loss = alpha_t * loss
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Cross-entropy with label smoothing - prevents overconfident predictions.
    """
    
    def __init__(self, smoothing: float = 0.1, weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.smoothing = smoothing
        self.weight = weight
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(inputs, dim=-1)
        num_classes = inputs.size(-1)
        
        # Smooth labels
        smooth_targets = torch.zeros_like(log_probs).scatter_(
            -1, targets.unsqueeze(-1), 1.0
        )
        smooth_targets = smooth_targets * (1 - self.smoothing) + \
                        self.smoothing / num_classes
        
        # Cross-entropy
        loss = -(smooth_targets * log_probs).sum(dim=-1)
        
        # Apply class weights
        if self.weight is not None:
            if self.weight.device != targets.device:
                self.weight = self.weight.to(targets.device)
            loss = loss * self.weight[targets]
        
        return loss.mean()


# ============================================
# 4. DROPOUT INJECTION FOR MODELS
# ============================================

class ModelWithDropout(nn.Module):
    """
    Wrapper to add dropout to any model's classifier head.
    
    Prevents overfitting by adding dropout before final classification layer.
    """
    
    def __init__(self, base_model: nn.Module, dropout_rate: float = 0.5):
        """
        Args:
            base_model: The base model (e.g., from timm)
            dropout_rate: Dropout probability
        """
        super().__init__()
        self.base_model = base_model
        self.dropout = nn.Dropout(p=dropout_rate)
        
        # Get the classifier
        if hasattr(base_model, 'fc'):
            self.fc = base_model.fc
            base_model.fc = nn.Identity()
        elif hasattr(base_model, 'head'):
            self.fc = base_model.head
            base_model.head = nn.Identity()
        elif hasattr(base_model, 'classifier'):
            self.fc = base_model.classifier
            base_model.classifier = nn.Identity()
        else:
            raise ValueError("Cannot find classifier head in model")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.base_model(x)
        features = self.dropout(features)
        return self.fc(features)


def add_dropout_to_model(model: nn.Module, dropout_rate: float = 0.3) -> nn.Module:
    """
    Recursively add dropout layers after activation functions.
    
    This is a more aggressive approach that adds dropout throughout the network.
    """
    for name, child in model.named_children():
        if isinstance(child, (nn.ReLU, nn.GELU, nn.SiLU)):
            setattr(model, name, nn.Sequential(
                child,
                nn.Dropout(p=dropout_rate)
            ))
        else:
            add_dropout_to_model(child, dropout_rate)
    return model


# ============================================
# 5. LEARNING RATE WARMUP
# ============================================

class LinearWarmupCosineAnnealingLR:
    """
    Learning rate scheduler with linear warmup followed by cosine annealing.
    
    Warmup helps stabilize training in early epochs, preventing overfitting
    to initial gradients.
    """
    
    def __init__(self, optimizer, warmup_epochs: int, max_epochs: int,
                 warmup_start_lr: float = 1e-6, eta_min: float = 1e-6):
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
            # Linear warmup
            alpha = self.current_epoch / self.warmup_epochs
            for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
                pg['lr'] = self.warmup_start_lr + alpha * (base_lr - self.warmup_start_lr)
        else:
            # Cosine annealing
            progress = (self.current_epoch - self.warmup_epochs) / \
                      (self.max_epochs - self.warmup_epochs)
            for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
                pg['lr'] = self.eta_min + 0.5 * (base_lr - self.eta_min) * \
                          (1 + np.cos(np.pi * progress))
    
    def get_lr(self) -> list:
        return [pg['lr'] for pg in self.optimizer.param_groups]


# ============================================
# 6. GRADIENT CLIPPING
# ============================================

def clip_gradients(model: nn.Module, max_norm: float = 1.0) -> float:
    """
    Clip gradients to prevent exploding gradients.
    
    Returns the total gradient norm before clipping.
    """
    return nn.utils.clip_grad_norm_(model.parameters(), max_norm)


# ============================================
# 7. EXPONENTIAL MOVING AVERAGE (EMA)
# ============================================

class EMA:
    """
    Exponential Moving Average of model weights.
    
    Maintains a smoothed version of parameters that often generalizes better.
    """
    
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        """Update shadow weights with current model weights."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name].mul_(self.decay).add_(
                    param.data, alpha=1 - self.decay
                )
    
    def apply_shadow(self):
        """Apply shadow weights to model (for evaluation)."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])
    
    def restore(self):
        """Restore original weights after evaluation."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.backup[name])
        self.backup = {}


# ============================================
# 8. COMPREHENSIVE TRAINING CONFIGURATION
# ============================================

def get_anti_overfit_config(model_name: str) -> dict:
    """
    Get recommended anti-overfitting configuration for each model.
    
    Based on Gayatri & Aarthy (2023) recommendations + additional techniques.
    """
    base_config = {
        'augmentation_strength': 'strong',
        'mixup_alpha': 0.4,
        'cutmix_alpha': 1.0,
        'mixup_prob': 0.5,
        'label_smoothing': 0.1,
        'dropout_rate': 0.3,
        'weight_decay': 0.05,
        'warmup_epochs': 5,
        'use_focal_loss': True,
        'focal_gamma': 2.0,
        'gradient_clip': 1.0,
        'use_ema': True,
        'ema_decay': 0.999,
        'early_stopping_patience': 15,
    }
    
    # Model-specific configs
    model_configs = {
        'resnet50': {
            **base_config,
            'lr': 3e-4,
            'batch_size': 32,
            'dropout_rate': 0.4,
            'weight_decay': 0.05,
        },
        'efficientnet_b4': {
            **base_config,
            'lr': 2e-4,
            'batch_size': 24,
            'dropout_rate': 0.4,  # EfficientNet already has dropout
            'weight_decay': 0.02,
        },
        'convnext_base': {
            **base_config,
            'lr': 1e-4,
            'batch_size': 16,
            'dropout_rate': 0.3,
            'weight_decay': 0.05,
            'augmentation_strength': 'strong',
        },
        'vit_base': {
            **base_config,
            'lr': 1e-5,  # Lower LR for ViT
            'batch_size': 16,
            'dropout_rate': 0.1,
            'weight_decay': 0.1,  # Higher WD for ViT (from DeiT paper)
            'warmup_epochs': 10,  # Longer warmup for ViT
        },
        'deit_base': {
            **base_config,
            'lr': 1e-5,
            'batch_size': 16,
            'dropout_rate': 0.0,  # DeiT already has attention dropout
            'weight_decay': 0.05,
            'warmup_epochs': 10,
        },
        'swin_base': {
            **base_config,
            'lr': 2e-5,
            'batch_size': 16,
            'dropout_rate': 0.2,
            'weight_decay': 0.05,
            'warmup_epochs': 8,
        },
    }
    
    return model_configs.get(model_name, base_config)


# ============================================
# 9. LAYER-WISE LEARNING RATE DECAY (LLRD)
# ============================================

def get_layer_wise_lr_groups(model, base_lr: float, decay_rate: float = 0.9):
    """
    Apply layer-wise learning rate decay.
    
    Earlier layers get smaller learning rates (features already good from pretraining).
    Later layers get higher learning rates (need more adaptation).
    
    This is particularly effective for fine-tuning pretrained models on medical images.
    """
    param_groups = []
    
    # Get all named parameters
    params = list(model.named_parameters())
    num_layers = len(params)
    
    for idx, (name, param) in enumerate(params):
        if not param.requires_grad:
            continue
        
        # Calculate layer-specific LR: earlier layers get smaller LR
        layer_lr = base_lr * (decay_rate ** (num_layers - idx - 1))
        
        param_groups.append({
            'params': [param],
            'lr': layer_lr,
            'name': name
        })
    
    return param_groups


def get_two_stage_lr_groups(model, head_lr: float, backbone_lr: float = None):
    """
    Two-stage learning rates: higher for head, lower for backbone.
    
    Args:
        model: The model
        head_lr: Learning rate for classifier head
        backbone_lr: Learning rate for backbone (default: head_lr / 10)
    """
    if backbone_lr is None:
        backbone_lr = head_lr / 10
    
    backbone_params = []
    head_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        # Common classifier head names
        if any(x in name.lower() for x in ['head', 'fc', 'classifier']):
            head_params.append(param)
        else:
            backbone_params.append(param)
    
    return [
        {'params': backbone_params, 'lr': backbone_lr},
        {'params': head_params, 'lr': head_lr}
    ]


# ============================================
# HELPER FUNCTION TO APPLY ALL TECHNIQUES
# ============================================

def prepare_anti_overfit_training(model, model_name: str, train_loader, device: str = 'cuda'):
    """
    Prepare model and training components with anti-overfitting techniques.
    
    Returns:
        dict with model, optimizer, scheduler, criterion, mixup, ema
    """
    config = get_anti_overfit_config(model_name)
    
    # 1. Get class weights for imbalanced classes
    class_weights = train_loader.dataset.get_class_weights().to(device)
    
    # 2. Choose loss function
    if config['use_focal_loss']:
        criterion = FocalLoss(
            alpha=class_weights,
            gamma=config['focal_gamma'],
            label_smoothing=config['label_smoothing']
        )
    else:
        criterion = LabelSmoothingCrossEntropy(
            smoothing=config['label_smoothing'],
            weight=class_weights
        )
    
    # 3. Setup optimizer with two-stage LR
    optimizer = torch.optim.AdamW(
        get_two_stage_lr_groups(model, head_lr=config['lr']),
        weight_decay=config['weight_decay']
    )
    
    # 4. Setup scheduler with warmup
    scheduler = LinearWarmupCosineAnnealingLR(
        optimizer,
        warmup_epochs=config['warmup_epochs'],
        max_epochs=50,
        eta_min=1e-6
    )
    
    # 5. Setup Mixup/CutMix
    mixup = MixupCutmix(
        mixup_alpha=config['mixup_alpha'],
        cutmix_alpha=config['cutmix_alpha'],
        prob=config['mixup_prob'],
        num_classes=8
    )
    
    # 6. Setup EMA
    ema = EMA(model, decay=config['ema_decay']) if config['use_ema'] else None
    
    return {
        'model': model,
        'optimizer': optimizer,
        'scheduler': scheduler,
        'criterion': criterion,
        'mixup': mixup,
        'ema': ema,
        'config': config
    }


if __name__ == "__main__":
    # Test the module
    print("Anti-Overfitting Module Loaded Successfully!")
    print("\nAvailable functions:")
    print("  - get_anti_overfit_train_transform()")
    print("  - MixupCutmix")
    print("  - FocalLoss")
    print("  - LinearWarmupCosineAnnealingLR")
    print("  - EMA")
    print("  - get_anti_overfit_config()")
    print("  - prepare_anti_overfit_training()")
