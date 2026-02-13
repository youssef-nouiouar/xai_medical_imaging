"""
Enhanced Training Script with Anti-Overfitting Techniques
==========================================================
Implements recommendations from:
1. Gayatri & Aarthy (2023) - "Reduction of overfitting on ISIC-2019"
2. State-of-the-art regularization techniques

Key techniques applied:
- Strong dermoscopy-specific data augmentation
- Mixup & CutMix regularization
- Focal Loss for class imbalance
- Learning rate warmup + cosine annealing
- Exponential Moving Average (EMA)
- Two-stage learning rates (backbone vs head)
- Gradient clipping
- Early stopping with patience

Usage:
    python train_anti_overfit.py --model resnet50 --epochs 50
    python train_anti_overfit.py --model efficientnet_b4 --epochs 50
    python train_anti_overfit.py --model convnext_base --epochs 50
"""

import os
import sys
import argparse
import json
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.isic_dataset import ISICDataset
from utils.anti_overfitting import (
    get_anti_overfit_train_transform,
    get_val_test_transform,
    MixupCutmix,
    FocalLoss,
    LinearWarmupCosineAnnealingLR,
    EMA,
    clip_gradients,
    get_anti_overfit_config,
    get_two_stage_lr_groups
)

import timm


class AntiOverfitTrainer:
    """
    Trainer class with comprehensive anti-overfitting techniques.
    """
    
    def __init__(self, model_name: str, save_dir: str = './results',
                 device: str = 'cuda', num_classes: int = 8):
        self.model_name = model_name
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.num_classes = num_classes
        
        # Get model-specific anti-overfitting config
        self.config = get_anti_overfit_config(model_name)
        
        # Initialize model
        self.model = self._create_model()
        
        # Training components (initialized in setup())
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.mixup = None
        self.ema = None
        self.scaler = None
        
        # Tracking
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': [],
            'val_f1': [], 'val_auc': [],
            'lr': []
        }
        self.best_val_acc = 0.0
        self.best_val_f1 = 0.0
        self.patience_counter = 0
    
    def _create_model(self) -> nn.Module:
        """Create model with timm."""
        # Map model names to timm names
        timm_names = {
            'resnet50': 'resnet50',
            'efficientnet_b4': 'efficientnet_b4',
            'convnext_base': 'convnext_base',
            'vit_base': 'vit_base_patch16_224',
            'deit_base': 'deit_base_patch16_224',
            'swin_base': 'swin_base_patch4_window7_224'
        }
        
        timm_name = timm_names.get(self.model_name, self.model_name)
        
        # Create model with dropout
        model = timm.create_model(
            timm_name,
            pretrained=True,
            num_classes=self.num_classes,
            drop_rate=self.config['dropout_rate']
        )
        
        return model.to(self.device)
    
    def setup_training(self, train_loader: DataLoader):
        """Setup all training components with anti-overfitting techniques."""
        
        # 1. Get class weights for imbalanced dataset
        class_weights = train_loader.dataset.get_class_weights().to(self.device)
        print(f"\nClass weights computed: {class_weights.cpu().numpy().round(3)}")
        
        # 2. Focal Loss with class weights and label smoothing
        if self.config['use_focal_loss']:
            self.criterion = FocalLoss(
                alpha=class_weights,
                gamma=self.config['focal_gamma'],
                label_smoothing=self.config['label_smoothing']
            )
            print(f"Using Focal Loss (gamma={self.config['focal_gamma']}, smoothing={self.config['label_smoothing']})")
        else:
            self.criterion = nn.CrossEntropyLoss(
                weight=class_weights,
                label_smoothing=self.config['label_smoothing']
            )
            print(f"Using CrossEntropy with class weights and label smoothing")
        
        # 3. Optimizer with two-stage learning rates
        param_groups = get_two_stage_lr_groups(
            self.model,
            head_lr=self.config['lr'],
            backbone_lr=self.config['lr'] / 10
        )
        self.optimizer = torch.optim.AdamW(
            param_groups,
            weight_decay=self.config['weight_decay']
        )
        print(f"Optimizer: AdamW (head_lr={self.config['lr']}, backbone_lr={self.config['lr']/10:.1e})")
        print(f"Weight decay: {self.config['weight_decay']}")
        
        # 4. Scheduler with warmup
        # Will be set in train() once we know num_epochs
        
        # 5. Mixup/CutMix
        self.mixup = MixupCutmix(
            mixup_alpha=self.config['mixup_alpha'],
            cutmix_alpha=self.config['cutmix_alpha'],
            prob=self.config['mixup_prob'],
            num_classes=self.num_classes
        )
        print(f"Mixup/CutMix enabled (prob={self.config['mixup_prob']})")
        
        # 6. EMA
        if self.config['use_ema']:
            self.ema = EMA(self.model, decay=self.config['ema_decay'])
            print(f"EMA enabled (decay={self.config['ema_decay']})")
        
        # 7. Mixed precision
        self.scaler = GradScaler()
        print("Mixed precision training enabled")
    
    def train_epoch(self, train_loader: DataLoader, epoch: int) -> tuple:
        """Train for one epoch with all anti-overfitting techniques."""
        self.model.train()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}', leave=False)
        for batch_idx, (images, labels, _) in enumerate(pbar):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Apply Mixup/CutMix
            mixed_images, labels_a, labels_b, lam = self.mixup(images, labels)
            
            self.optimizer.zero_grad()
            
            # Forward pass with mixed precision
            with autocast():
                outputs = self.model(mixed_images)
                
                # Mixed loss
                if lam < 1.0:
                    loss = MixupCutmix.mixup_criterion(
                        self.criterion, outputs, labels_a, labels_b, lam
                    )
                else:
                    loss = self.criterion(outputs, labels_a)
            
            # Backward pass
            self.scaler.scale(loss).backward()
            
            # Gradient clipping
            self.scaler.unscale_(self.optimizer)
            clip_gradients(self.model, max_norm=self.config['gradient_clip'])
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # Update EMA
            if self.ema is not None:
                self.ema.update()
            
            total_loss += loss.item()
            
            # Track predictions (using original labels, not mixed)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
            })
        
        avg_loss = total_loss / len(train_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        
        return avg_loss, accuracy
    
    @torch.no_grad()
    def validate(self, val_loader: DataLoader, use_ema: bool = True) -> dict:
        """Validate the model."""
        # Use EMA weights for validation if available
        if use_ema and self.ema is not None:
            self.ema.apply_shadow()
        
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        all_probs = []
        
        for images, labels, _ in val_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            total_loss += loss.item()
            
            probs = F.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)
            
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        
        # Restore original weights
        if use_ema and self.ema is not None:
            self.ema.restore()
        
        # Compute metrics
        avg_loss = total_loss / len(val_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='macro')
        
        try:
            all_probs_np = np.array(all_probs)
            auc = roc_auc_score(all_labels, all_probs_np, multi_class='ovr', average='macro')
        except:
            auc = 0.0
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'f1_macro': f1,
            'auc_roc': auc,
            'predictions': all_preds,
            'labels': all_labels
        }
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader,
              epochs: int = 50, patience: int = None) -> dict:
        """
        Full training loop with anti-overfitting techniques.
        """
        # Setup training components
        self.setup_training(train_loader)
        
        # Setup scheduler
        self.scheduler = LinearWarmupCosineAnnealingLR(
            self.optimizer,
            warmup_epochs=self.config['warmup_epochs'],
            max_epochs=epochs,
            eta_min=1e-6
        )
        
        if patience is None:
            patience = self.config['early_stopping_patience']
        
        print(f"\n{'='*60}")
        print(f"Training {self.model_name} with Anti-Overfitting Techniques")
        print(f"{'='*60}")
        print(f"Epochs: {epochs}")
        print(f"Warmup epochs: {self.config['warmup_epochs']}")
        print(f"Early stopping patience: {patience}")
        print(f"Batch size: {self.config['batch_size']}")
        print(f"Augmentation strength: {self.config['augmentation_strength']}")
        print(f"{'='*60}\n")
        
        for epoch in range(1, epochs + 1):
            # Train
            train_loss, train_acc = self.train_epoch(train_loader, epoch)
            
            # Validate
            val_metrics = self.validate(val_loader)
            
            # Update scheduler
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Log
            print(f"Epoch {epoch:3d}/{epochs} | "
                  f"Train: {train_loss:.4f}/{train_acc:.4f} | "
                  f"Val: {val_metrics['loss']:.4f}/{val_metrics['accuracy']:.4f} "
                  f"F1:{val_metrics['f1_macro']:.4f} AUC:{val_metrics['auc_roc']:.4f} | "
                  f"LR: {current_lr:.2e}")
            
            # History
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_acc'].append(val_metrics['accuracy'])
            self.history['val_f1'].append(val_metrics['f1_macro'])
            self.history['val_auc'].append(val_metrics['auc_roc'])
            self.history['lr'].append(current_lr)
            
            # Check for improvement (using F1 for imbalanced data)
            improved = False
            if val_metrics['f1_macro'] > self.best_val_f1:
                self.best_val_f1 = val_metrics['f1_macro']
                self.best_val_acc = val_metrics['accuracy']
                self.patience_counter = 0
                improved = True
                
                # Save best model
                self.save_checkpoint(epoch, val_metrics, is_best=True)
                print(f"         ★ New best model! F1: {self.best_val_f1:.4f}")
            else:
                self.patience_counter += 1
            
            # Early stopping
            if self.patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch} (no improvement for {patience} epochs)")
                break
        
        # Save final history
        self.save_history()
        
        return self.history
    
    def save_checkpoint(self, epoch: int, metrics: dict, is_best: bool = False):
        """Save model checkpoint."""
        suffix = 'best' if is_best else f'epoch{epoch}'
        
        # Save main model
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_acc': metrics['accuracy'],
            'val_f1': metrics['f1_macro'],
            'val_auc': metrics['auc_roc'],
            'config': self.config
        }
        
        # Save EMA weights if available
        if self.ema is not None:
            checkpoint['ema_shadow'] = self.ema.shadow
        
        path = self.save_dir / f'{self.model_name}_{suffix}.pth'
        torch.save(checkpoint, path)
    
    def save_history(self):
        """Save training history."""
        # Save as JSON
        history_path = self.save_dir / f'{self.model_name}_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        # Save training curves plot
        self.plot_training_curves()
    
    def plot_training_curves(self):
        """Plot and save training curves."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        epochs = range(1, len(self.history['train_loss']) + 1)
        
        # Loss
        ax = axes[0, 0]
        ax.plot(epochs, self.history['train_loss'], 'b-', label='Train', linewidth=2)
        ax.plot(epochs, self.history['val_loss'], 'r-', label='Validation', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training and Validation Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add gap indicator
        final_gap = self.history['train_loss'][-1] - self.history['val_loss'][-1]
        ax.text(0.02, 0.98, f'Final Gap: {abs(final_gap):.4f}',
               transform=ax.transAxes, fontsize=10, verticalalignment='top')
        
        # Accuracy
        ax = axes[0, 1]
        ax.plot(epochs, self.history['train_acc'], 'b-', label='Train', linewidth=2)
        ax.plot(epochs, self.history['val_acc'], 'r-', label='Validation', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        ax.set_title('Training and Validation Accuracy')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Overfitting gap
        acc_gap = [t - v for t, v in zip(self.history['train_acc'], self.history['val_acc'])]
        ax.fill_between(epochs, self.history['val_acc'], self.history['train_acc'],
                       alpha=0.2, color='gray', label='Overfit Gap')
        
        # F1 and AUC
        ax = axes[1, 0]
        ax.plot(epochs, self.history['val_f1'], 'g-', label='F1 Macro', linewidth=2)
        ax.plot(epochs, self.history['val_auc'], 'm-', label='AUC-ROC', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Score')
        ax.set_title('Validation F1 and AUC-ROC')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Learning rate
        ax = axes[1, 1]
        ax.plot(epochs, self.history['lr'], 'k-', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Learning Rate')
        ax.set_title('Learning Rate Schedule (with Warmup)')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        ax.axvline(x=self.config['warmup_epochs'], color='r', linestyle='--',
                  label=f'Warmup end ({self.config["warmup_epochs"]} epochs)')
        ax.legend()
        
        plt.suptitle(f'{self.model_name} - Training with Anti-Overfitting Techniques',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Save
        plot_path = self.save_dir / f'{self.model_name}_training_curves.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\nTraining curves saved to: {plot_path}")


def create_data_loaders(root_dir: str, config: dict, num_workers: int = 4) -> dict:
    """Create data loaders with enhanced augmentation."""
    
    loaders = {}
    
    for split in ['train', 'val', 'test']:
        # Use strong augmentation for training
        if split == 'train':
            transform = get_anti_overfit_train_transform(
                image_size=224,
                strength=config['augmentation_strength']
            )
        else:
            transform = get_val_test_transform(image_size=224)
        
        dataset = ISICDataset(
            root_dir=root_dir,
            split=split,
            transform=transform,
            image_size=224,
            use_albumentations=False,  # We're using our custom transforms
            use_official_test=True
        )
        
        # Use weighted sampling for training
        if split == 'train':
            class_weights = dataset.get_class_weights()
            sample_weights = class_weights[dataset.labels]
            sampler = torch.utils.data.WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(dataset),
                replacement=True
            )
            loaders[split] = DataLoader(
                dataset,
                batch_size=config['batch_size'],
                sampler=sampler,
                num_workers=num_workers,
                pin_memory=True,
                drop_last=True
            )
        else:
            loaders[split] = DataLoader(
                dataset,
                batch_size=config['batch_size'] * 2,  # Larger batch for eval
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True
            )
        
        print(f"{split.capitalize()} loader: {len(dataset)} images")
    
    return loaders


def main():
    parser = argparse.ArgumentParser(
        description='Train models with anti-overfitting techniques'
    )
    parser.add_argument('--model', type=str, default='resnet50',
                       choices=['resnet50', 'efficientnet_b4', 'convnext_base',
                               'vit_base', 'deit_base', 'swin_base'],
                       help='Model to train')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--data_dir', type=str, default='./data/ISIC2019',
                       help='Path to ISIC2019 dataset')
    parser.add_argument('--save_dir', type=str, default='./results',
                       help='Directory to save results')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loader workers')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')
    
    args = parser.parse_args()
    
    # Verify CUDA
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'
    
    print(f"\n{'#'*70}")
    print(f"# Anti-Overfitting Training: {args.model}")
    print(f"{'#'*70}\n")
    
    # Get config
    config = get_anti_overfit_config(args.model)
    
    # Create data loaders
    print("Creating data loaders with enhanced augmentation...")
    loaders = create_data_loaders(args.data_dir, config, args.num_workers)
    
    # Create trainer
    trainer = AntiOverfitTrainer(
        model_name=args.model,
        save_dir=args.save_dir,
        device=args.device
    )
    
    # Train
    history = trainer.train(
        loaders['train'],
        loaders['val'],
        epochs=args.epochs
    )
    
    # Final evaluation on test set
    print("\n" + "="*60)
    print("Final Test Set Evaluation")
    print("="*60)
    
    test_metrics = trainer.validate(loaders['test'], use_ema=True)
    
    print(f"\nTest Results:")
    print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"  F1 Macro: {test_metrics['f1_macro']:.4f}")
    print(f"  AUC-ROC:  {test_metrics['auc_roc']:.4f}")
    
    # Save test results
    results = {
        'model': args.model,
        'test_accuracy': test_metrics['accuracy'],
        'test_f1': test_metrics['f1_macro'],
        'test_auc': test_metrics['auc_roc'],
        'best_val_acc': trainer.best_val_acc,
        'best_val_f1': trainer.best_val_f1,
        'config': config
    }
    
    results_path = Path(args.save_dir) / f'{args.model}_test_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {results_path}")


if __name__ == '__main__':
    main()
