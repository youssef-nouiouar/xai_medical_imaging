"""
Training Script
===============
Script principal pour entraîner les modèles CNN, ViT et Hybrides
sur le dataset ISIC pour la classification des lésions cutanées.

Auteur: [Votre Nom]
Date: 2025
"""

import os
import argparse
import yaml
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import numpy as np
from sklearn.metrics import (
    accuracy_score, roc_auc_score, classification_report,
    confusion_matrix, f1_score
)
import matplotlib.pyplot as plt
import seaborn as sns

# Imports locaux
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.isic_dataset import ISICDataset, get_data_loaders, download_isic_instructions
from models.architectures import ModelFactory


class Trainer:
    """Classe principale pour l'entraînement des modèles."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict,
        device: str = 'cuda',
        save_dir: str = './results'
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Optimizer
        self.optimizer = self._get_optimizer()
        
        # Scheduler
        self.scheduler = self._get_scheduler()
        
        # Loss function
        self.criterion = self._get_criterion()
        
        # Mixed precision
        self.scaler = GradScaler() if config.get('mixed_precision', True) else None
        
        # Tracking
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': [],
            'val_auc': []
        }
        self.best_val_acc = 0.0
        self.best_val_auc = 0.0
        self.patience_counter = 0
        
    def _get_optimizer(self) -> optim.Optimizer:
        """Configurer l'optimizer."""
        opt_config = self.config.get('optimizer', {})
        opt_name = opt_config.get('name', 'AdamW')
        lr = opt_config.get('lr', 1e-4)
        weight_decay = opt_config.get('weight_decay', 0.01)
        
        if opt_name == 'AdamW':
            return optim.AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        elif opt_name == 'Adam':
            return optim.Adam(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        elif opt_name == 'SGD':
            return optim.SGD(
                self.model.parameters(),
                lr=lr,
                momentum=0.9,
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Optimizer inconnu: {opt_name}")
    
    def _get_scheduler(self) -> Optional[optim.lr_scheduler._LRScheduler]:
        """Configurer le scheduler."""
        sched_config = self.config.get('scheduler', {})
        sched_name = sched_config.get('name', 'CosineAnnealingLR')
        
        if sched_name == 'CosineAnnealingLR':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=sched_config.get('T_max', 50),
                eta_min=sched_config.get('eta_min', 1e-6)
            )
        elif sched_name == 'StepLR':
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=sched_config.get('step_size', 10),
                gamma=sched_config.get('gamma', 0.1)
            )
        elif sched_name == 'ReduceLROnPlateau':
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=0.5,
                patience=5
            )
        else:
            return None
    
    def _get_criterion(self) -> nn.Module:
        """Configurer la fonction de perte."""
        loss_config = self.config.get('loss', {})
        label_smoothing = loss_config.get('label_smoothing', 0.1)
        
        # Poids des classes si spécifié
        if loss_config.get('class_weights', False):
            weights = self.train_loader.dataset.get_class_weights().to(self.device)
            return nn.CrossEntropyLoss(weight=weights, label_smoothing=label_smoothing)
        else:
            return nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    
    def train_epoch(self) -> Tuple[float, float]:
        """Entraîner pendant une epoch."""
        self.model.train()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        pbar = tqdm(self.train_loader, desc='Training')
        for batch_idx, (images, labels, _) in enumerate(pbar):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward avec mixed precision
            if self.scaler is not None:
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
            
            total_loss += loss.item()
            
            # Predictions
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.6f}'
            })
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        
        return avg_loss, accuracy
    
    @torch.no_grad()
    def validate(self) -> Tuple[float, float, float, Dict]:
        """Valider le modèle."""
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        all_probs = []
        
        for images, labels, _ in tqdm(self.val_loader, desc='Validation'):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            total_loss += loss.item()
            
            probs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        
        # AUC (multiclass)
        try:
            all_probs = np.array(all_probs)
            auc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='weighted')
        except:
            auc = 0.0
        
        # F1 Score
        f1 = f1_score(all_labels, all_preds, average='weighted')
        
        metrics = {
            'accuracy': accuracy,
            'auc': auc,
            'f1': f1,
            'confusion_matrix': confusion_matrix(all_labels, all_preds).tolist(),
            'classification_report': classification_report(
                all_labels, all_preds,
                target_names=ISICDataset.CLASS_NAMES,
                output_dict=True
            )
        }
        
        return avg_loss, accuracy, auc, metrics
    
    def train(self, num_epochs: int, patience: int = 10) -> Dict:
        """
        Boucle d'entraînement principale.
        
        Args:
            num_epochs: Nombre d'epochs
            patience: Patience pour early stopping
            
        Returns:
            Historique d'entraînement
        """
        print(f"\n{'='*60}")
        print(f"Début de l'entraînement - {num_epochs} epochs")
        print(f"Device: {self.device}")
        print(f"Batch size: {self.train_loader.batch_size}")
        print(f"{'='*60}\n")
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print("-" * 40)
            
            # Train
            train_loss, train_acc = self.train_epoch()
            
            # Validate
            val_loss, val_acc, val_auc, metrics = self.validate()
            
            # Update scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_acc)
                else:
                    self.scheduler.step()
            
            # Log
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val AUC: {val_auc:.4f}")
            
            # History
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            self.history['val_auc'].append(val_auc)
            
            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_val_auc = val_auc
                self.patience_counter = 0
                
                self._save_checkpoint(epoch, metrics, is_best=True)
                print(f"  ✓ Nouveau meilleur modèle sauvegardé (acc: {val_acc:.4f})")
            else:
                self.patience_counter += 1
                
            # Early stopping
            if self.patience_counter >= patience:
                print(f"\nEarly stopping après {epoch+1} epochs")
                break
        
        # Sauvegarder l'historique
        self._save_history()
        
        # Plots
        self._plot_training_curves()
        
        return self.history
    
    def _save_checkpoint(self, epoch: int, metrics: Dict, is_best: bool = False):
        """Sauvegarder un checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_acc': self.best_val_acc,
            'best_val_auc': self.best_val_auc,
            'metrics': metrics,
            'config': self.config
        }
        
        filename = 'best_model.pth' if is_best else f'checkpoint_epoch_{epoch}.pth'
        torch.save(checkpoint, self.save_dir / filename)
    
    def _save_history(self):
        """Sauvegarder l'historique d'entraînement."""
        with open(self.save_dir / 'training_history.json', 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def _plot_training_curves(self):
        """Tracer les courbes d'apprentissage."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # Loss
        axes[0].plot(self.history['train_loss'], label='Train')
        axes[0].plot(self.history['val_loss'], label='Val')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Loss Curves')
        axes[0].legend()
        axes[0].grid(True)
        
        # Accuracy
        axes[1].plot(self.history['train_acc'], label='Train')
        axes[1].plot(self.history['val_acc'], label='Val')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Accuracy Curves')
        axes[1].legend()
        axes[1].grid(True)
        
        # AUC
        axes[2].plot(self.history['val_auc'], label='Val AUC', color='green')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('AUC')
        axes[2].set_title('Validation AUC')
        axes[2].legend()
        axes[2].grid(True)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'training_curves.png', dpi=150)
        plt.close()
        
        print(f"\nCourbes sauvegardées dans {self.save_dir / 'training_curves.png'}")


def main():
    """Fonction principale."""
    parser = argparse.ArgumentParser(description='Entraînement des modèles')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Chemin vers le fichier de configuration')
    parser.add_argument('--model', type=str, default='resnet50',
                       choices=['resnet50', 'efficientnet_b4', 'vit_base', 
                               'deit_base', 'swin_base', 'convnext_base'],
                       help='Modèle à entraîner')
    parser.add_argument('--data_dir', type=str, default='../data/ISIC2019',
                       help='Chemin vers le dataset')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Nombre d\'epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Taille des batches')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda/cpu)')
    parser.add_argument('--save_dir', type=str, default='./results',
                       help='Dossier de sauvegarde')
    
    args = parser.parse_args()
    
    # Charger la configuration
    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            full_config = yaml.safe_load(f)
        config = full_config.get('training', {})
    else:
        config = {}

    # Override avec les arguments CLI
    config['optimizer'] = config.get('optimizer', {})
    config['optimizer']['lr'] = args.lr
    config['batch_size'] = args.batch_size
    
    # Vérifier le dataset
    if not os.path.exists(args.data_dir):
        print("❌ Dataset non trouvé!")
        download_isic_instructions()
        return
    
    # Device
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Utilisation de: {device}")
    
    # Charger les données
    print("\n[1/4] Chargement des données...")
    loaders = get_data_loaders(
        root_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=224,
        num_workers=4
    )
    
    print(f"  • Train: {len(loaders['train'].dataset)} images")
    print(f"  • Val: {len(loaders['val'].dataset)} images")
    print(f"  • Test: {len(loaders['test'].dataset)} images")
    
    # Créer le modèle
    print(f"\n[2/4] Création du modèle: {args.model}")
    model = ModelFactory.create(
        model_name=args.model,
        num_classes=8,  # ISIC 2019 (UNK excluded — 0 samples)
        pretrained=True
    )
    print(f"  • Type: {model.model_type}")
    
    # Dossier de sauvegarde
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = Path(args.save_dir) / f"{args.model}_{timestamp}"
    
    # Créer le trainer
    print(f"\n[3/4] Configuration du trainer...")
    trainer = Trainer(
        model=model,
        train_loader=loaders['train'],
        val_loader=loaders['val'],
        config=config,
        device=device,
        save_dir=save_dir
    )
    
    # Entraîner
    print(f"\n[4/4] Début de l'entraînement...")
    history = trainer.train(
        num_epochs=args.epochs,
        patience=config.get('early_stopping_patience', 10)
    )
    
    # Résumé final
    print("\n" + "=" * 60)
    print("ENTRAÎNEMENT TERMINÉ")
    print("=" * 60)
    print(f"Meilleure accuracy: {trainer.best_val_acc:.4f}")
    print(f"Meilleur AUC: {trainer.best_val_auc:.4f}")
    print(f"Modèle sauvegardé dans: {save_dir}")


if __name__ == "__main__":
    main()
