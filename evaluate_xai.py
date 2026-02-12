"""
XAI Evaluation Script
=====================
Script pour évaluer et comparer les méthodes XAI sur les modèles entraînés.

Auteur: [Votre Nom]
Date: 2025
"""

import os
import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import cv2

# Imports locaux
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.isic_dataset import ISICDataset, get_data_loaders
from models.architectures import ModelFactory
from xai_methods.explainers import (
    XAIFactory, GradCAMExplainer, AttentionRolloutExplainer,
    GenericAttentionExplainer, LIMEExplainer, IntegratedGradientsExplainer
)


class XAIEvaluator:
    """Classe pour évaluer les méthodes XAI."""
    
    def __init__(
        self,
        model: torch.nn.Module,
        device: str = 'cuda',
        save_dir: str = './results/xai'
    ):
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialiser les méthodes XAI
        self.xai_methods = self._initialize_xai_methods()
        
    def _initialize_xai_methods(self) -> Dict:
        """Initialiser les méthodes XAI compatibles avec le modèle."""
        methods = {}
        model_type = getattr(self.model, 'model_type', 'cnn')
        
        # Grad-CAM (compatible avec tous)
        try:
            target_layer = self.model.get_target_layer()
            methods['gradcam'] = GradCAMExplainer(
                self.model, target_layer, self.device
            )
            print("  ✓ Grad-CAM initialisé")
        except Exception as e:
            print(f"  ✗ Grad-CAM: {e}")
        
        # Méthodes spécifiques aux ViT
        if model_type == 'vit':
            try:
                methods['attention_rollout'] = AttentionRolloutExplainer(
                    self.model, self.device
                )
                print("  ✓ Attention Rollout initialisé")
            except Exception as e:
                print(f"  ✗ Attention Rollout: {e}")
            
            try:
                methods['generic_attention'] = GenericAttentionExplainer(
                    self.model, self.device
                )
                print("  ✓ Generic Attention initialisé")
            except Exception as e:
                print(f"  ✗ Generic Attention: {e}")
        
        # Integrated Gradients (compatible avec tous)
        try:
            methods['integrated_gradients'] = IntegratedGradientsExplainer(
                self.model, self.device, n_steps=50
            )
            print("  ✓ Integrated Gradients initialisé")
        except Exception as e:
            print(f"  ✗ Integrated Gradients: {e}")
        
        # LIME (compatible avec tous)
        try:
            methods['lime'] = LIMEExplainer(
                self.model, self.device, num_samples=500
            )
            print("  ✓ LIME initialisé")
        except Exception as e:
            print(f"  ✗ LIME: {e}")
        
        return methods
    
    def explain_single(
        self,
        image: torch.Tensor,
        target_class: Optional[int] = None
    ) -> Dict[str, np.ndarray]:
        """
        Générer les explications pour une seule image.
        
        Args:
            image: Tensor de l'image (1, C, H, W)
            target_class: Classe cible (None = prédite)
            
        Returns:
            Dict avec les cartes de saillance de chaque méthode
        """
        explanations = {}
        
        for method_name, explainer in self.xai_methods.items():
            try:
                saliency = explainer.explain(image, target_class)
                explanations[method_name] = saliency
            except Exception as e:
                print(f"Erreur {method_name}: {e}")
                explanations[method_name] = np.zeros((224, 224))
        
        return explanations
    
    def visualize_explanations(
        self,
        image: torch.Tensor,
        explanations: Dict[str, np.ndarray],
        prediction: int,
        true_label: int,
        save_path: Optional[Path] = None
    ):
        """Visualiser les explications côte à côte."""
        num_methods = len(explanations)
        fig, axes = plt.subplots(1, num_methods + 1, figsize=(4 * (num_methods + 1), 4))
        
        # Image originale
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        original = image.squeeze(0).cpu() * std + mean
        original = original.permute(1, 2, 0).numpy()
        original = np.clip(original, 0, 1)
        
        axes[0].imshow(original)
        axes[0].set_title(f'Original\nPred: {ISICDataset.CLASS_NAMES[prediction]}\nTrue: {ISICDataset.CLASS_NAMES[true_label]}')
        axes[0].axis('off')
        
        # Explications
        for idx, (method_name, saliency) in enumerate(explanations.items()):
            ax = axes[idx + 1]
            
            # Superposer la saliency map
            heatmap = cv2.applyColorMap(np.uint8(255 * saliency), cv2.COLORMAP_JET)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
            
            # Resize si nécessaire
            if saliency.shape != original.shape[:2]:
                saliency_resized = cv2.resize(saliency, (original.shape[1], original.shape[0]))
                heatmap = cv2.applyColorMap(np.uint8(255 * saliency_resized), cv2.COLORMAP_JET)
                heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
            
            overlay = 0.5 * heatmap + 0.5 * original
            overlay = np.clip(overlay, 0, 1)
            
            ax.imshow(overlay)
            ax.set_title(method_name.replace('_', ' ').title())
            ax.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def compute_insertion_deletion(
        self,
        image: torch.Tensor,
        saliency: np.ndarray,
        target_class: int,
        num_steps: int = 100
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculer les courbes d'insertion et de délétion.
        
        Args:
            image: Image d'entrée
            saliency: Carte de saillance
            target_class: Classe cible
            num_steps: Nombre de pas
            
        Returns:
            Tuple (insertion_scores, deletion_scores)
        """
        image = image.to(self.device)
        
        # Aplatir et trier les indices par importance
        flat_saliency = saliency.flatten()
        sorted_indices = np.argsort(flat_saliency)[::-1]  # Du plus au moins important
        
        # Image baseline (floue ou noire)
        blurred = torch.zeros_like(image).to(self.device)
        
        insertion_scores = []
        deletion_scores = []
        
        pixels_per_step = len(sorted_indices) // num_steps
        
        for step in range(num_steps + 1):
            num_pixels = step * pixels_per_step
            
            # Insertion: partir de baseline, ajouter les pixels importants
            mask_insertion = np.zeros(flat_saliency.shape)
            mask_insertion[sorted_indices[:num_pixels]] = 1
            mask_insertion = mask_insertion.reshape(saliency.shape)
            mask_insertion = cv2.resize(mask_insertion, (224, 224))
            mask_insertion = torch.FloatTensor(mask_insertion).unsqueeze(0).unsqueeze(0).to(self.device)
            
            inserted_image = image * mask_insertion + blurred * (1 - mask_insertion)
            
            # Deletion: partir de l'image, supprimer les pixels importants
            mask_deletion = np.ones(flat_saliency.shape)
            mask_deletion[sorted_indices[:num_pixels]] = 0
            mask_deletion = mask_deletion.reshape(saliency.shape)
            mask_deletion = cv2.resize(mask_deletion, (224, 224))
            mask_deletion = torch.FloatTensor(mask_deletion).unsqueeze(0).unsqueeze(0).to(self.device)
            
            deleted_image = image * mask_deletion + blurred * (1 - mask_deletion)
            
            # Prédictions
            with torch.no_grad():
                ins_output = self.model(inserted_image)
                del_output = self.model(deleted_image)
                
                ins_prob = F.softmax(ins_output, dim=1)[0, target_class].item()
                del_prob = F.softmax(del_output, dim=1)[0, target_class].item()
            
            insertion_scores.append(ins_prob)
            deletion_scores.append(del_prob)
        
        return np.array(insertion_scores), np.array(deletion_scores)
    
    def evaluate_faithfulness(
        self,
        dataloader: torch.utils.data.DataLoader,
        num_samples: int = 100
    ) -> Dict:
        """
        Évaluer la fidélité des méthodes XAI sur un ensemble de données.
        
        Args:
            dataloader: DataLoader avec les données de test
            num_samples: Nombre d'échantillons à évaluer
            
        Returns:
            Dict avec les métriques de fidélité
        """
        results = {method: {'insertion_auc': [], 'deletion_auc': []} 
                   for method in self.xai_methods.keys()}
        
        sample_count = 0
        
        for images, labels, _ in tqdm(dataloader, desc='Évaluation XAI'):
            if sample_count >= num_samples:
                break
            
            for i in range(images.size(0)):
                if sample_count >= num_samples:
                    break
                
                image = images[i:i+1].to(self.device)
                label = labels[i].item()
                
                # Prédiction
                with torch.no_grad():
                    output = self.model(image)
                    pred = output.argmax(dim=1).item()
                
                # Seulement évaluer les prédictions correctes
                if pred != label:
                    continue
                
                # Générer les explications
                explanations = self.explain_single(image, pred)
                
                # Calculer insertion/deletion pour chaque méthode
                for method_name, saliency in explanations.items():
                    try:
                        ins_scores, del_scores = self.compute_insertion_deletion(
                            image, saliency, pred, num_steps=50
                        )
                        
                        # AUC approximative (somme normalisée)
                        ins_auc = np.trapz(ins_scores) / len(ins_scores)
                        del_auc = np.trapz(del_scores) / len(del_scores)
                        
                        results[method_name]['insertion_auc'].append(ins_auc)
                        results[method_name]['deletion_auc'].append(del_auc)
                    except Exception as e:
                        print(f"Erreur {method_name}: {e}")
                
                sample_count += 1
        
        # Calculer les moyennes
        summary = {}
        for method_name, metrics in results.items():
            summary[method_name] = {
                'insertion_auc_mean': np.mean(metrics['insertion_auc']) if metrics['insertion_auc'] else 0,
                'insertion_auc_std': np.std(metrics['insertion_auc']) if metrics['insertion_auc'] else 0,
                'deletion_auc_mean': np.mean(metrics['deletion_auc']) if metrics['deletion_auc'] else 0,
                'deletion_auc_std': np.std(metrics['deletion_auc']) if metrics['deletion_auc'] else 0,
            }
        
        return summary
    
    def sanity_check(
        self,
        image: torch.Tensor,
        target_class: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Sanity check: comparer les explications du modèle entraîné vs aléatoire.
        
        Args:
            image: Image d'entrée
            target_class: Classe cible
            
        Returns:
            Dict avec les scores de différence pour chaque méthode
        """
        # Explications du modèle entraîné
        trained_explanations = self.explain_single(image, target_class)
        
        # Créer un modèle avec poids aléatoires
        import copy
        random_model = copy.deepcopy(self.model)
        for param in random_model.parameters():
            param.data = torch.randn_like(param.data)
        
        # Créer un évaluateur temporaire avec le modèle aléatoire
        temp_evaluator = XAIEvaluator(random_model, self.device, self.save_dir / 'temp')
        random_explanations = temp_evaluator.explain_single(image, target_class)
        
        # Calculer la différence
        differences = {}
        for method_name in trained_explanations.keys():
            if method_name in random_explanations:
                diff = np.abs(trained_explanations[method_name] - random_explanations[method_name]).mean()
                differences[method_name] = diff
        
        return differences
    
    def generate_report(
        self,
        dataloader: torch.utils.data.DataLoader,
        num_samples: int = 50,
        num_visualizations: int = 10
    ):
        """
        Générer un rapport complet d'évaluation XAI.
        
        Args:
            dataloader: DataLoader de test
            num_samples: Nombre d'échantillons pour les métriques
            num_visualizations: Nombre de visualisations à générer
        """
        print("\n" + "=" * 60)
        print("GÉNÉRATION DU RAPPORT XAI")
        print("=" * 60)
        
        # 1. Visualisations
        print("\n[1/3] Génération des visualisations...")
        vis_dir = self.save_dir / 'visualizations'
        vis_dir.mkdir(exist_ok=True)
        
        vis_count = 0
        for images, labels, image_ids in dataloader:
            if vis_count >= num_visualizations:
                break
            
            for i in range(images.size(0)):
                if vis_count >= num_visualizations:
                    break
                
                image = images[i:i+1].to(self.device)
                label = labels[i].item()
                image_id = image_ids[i]
                
                with torch.no_grad():
                    output = self.model(image)
                    pred = output.argmax(dim=1).item()
                
                explanations = self.explain_single(image, pred)
                
                save_path = vis_dir / f'{image_id}_explanations.png'
                self.visualize_explanations(image, explanations, pred, label, save_path)
                
                vis_count += 1
        
        print(f"  ✓ {vis_count} visualisations générées dans {vis_dir}")
        
        # 2. Métriques de fidélité
        print("\n[2/3] Calcul des métriques de fidélité...")
        faithfulness = self.evaluate_faithfulness(dataloader, num_samples)
        
        # 3. Sauvegarder les résultats
        print("\n[3/3] Sauvegarde des résultats...")
        
        results = {
            'faithfulness': faithfulness,
            'timestamp': datetime.now().isoformat(),
            'num_samples': num_samples
        }
        
        with open(self.save_dir / 'xai_evaluation_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        # Afficher le résumé
        print("\n" + "=" * 60)
        print("RÉSULTATS")
        print("=" * 60)
        
        print("\nMétriques de Fidélité:")
        print("-" * 50)
        print(f"{'Méthode':<25} {'Insertion AUC':<15} {'Deletion AUC':<15}")
        print("-" * 50)
        
        for method, metrics in faithfulness.items():
            ins = f"{metrics['insertion_auc_mean']:.4f}±{metrics['insertion_auc_std']:.4f}"
            delete = f"{metrics['deletion_auc_mean']:.4f}±{metrics['deletion_auc_std']:.4f}"
            print(f"{method:<25} {ins:<15} {delete:<15}")
        
        # Plot de comparaison
        self._plot_faithfulness_comparison(faithfulness)
        
        print(f"\nRésultats sauvegardés dans: {self.save_dir}")
    
    def _plot_faithfulness_comparison(self, faithfulness: Dict):
        """Tracer un graphique de comparaison des méthodes XAI."""
        methods = list(faithfulness.keys())
        ins_means = [faithfulness[m]['insertion_auc_mean'] for m in methods]
        ins_stds = [faithfulness[m]['insertion_auc_std'] for m in methods]
        del_means = [faithfulness[m]['deletion_auc_mean'] for m in methods]
        del_stds = [faithfulness[m]['deletion_auc_std'] for m in methods]
        
        x = np.arange(len(methods))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        bars1 = ax.bar(x - width/2, ins_means, width, yerr=ins_stds, 
                       label='Insertion AUC ↑', color='#2ecc71', capsize=5)
        bars2 = ax.bar(x + width/2, del_means, width, yerr=del_stds,
                       label='Deletion AUC ↓', color='#e74c3c', capsize=5)
        
        ax.set_xlabel('Méthode XAI')
        ax.set_ylabel('AUC Score')
        ax.set_title('Comparaison des Méthodes XAI - Fidélité')
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace('_', '\n') for m in methods], fontsize=9)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'faithfulness_comparison.png', dpi=150)
        plt.close()


def main():
    """Fonction principale."""
    parser = argparse.ArgumentParser(description='Évaluation XAI')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Chemin vers le checkpoint du modèle')
    parser.add_argument('--model', type=str, required=True,
                       choices=['resnet50', 'efficientnet_b4', 'vit_base',
                               'deit_base', 'swin_base', 'convnext_base'],
                       help='Type de modèle')
    parser.add_argument('--data_dir', type=str, default='./data/ISIC2019',
                       help='Chemin vers le dataset')
    parser.add_argument('--save_dir', type=str, default='./results/xai',
                       help='Dossier de sauvegarde')
    parser.add_argument('--num_samples', type=int, default=50,
                       help='Nombre d\'échantillons pour l\'évaluation')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device')
    
    args = parser.parse_args()
    
    # Device
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Charger le modèle
    print(f"\n[1/3] Chargement du modèle {args.model}...")
    model = ModelFactory.create(args.model, num_classes=8, pretrained=False)
    
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"  ✓ Checkpoint chargé (epoch {checkpoint.get('epoch', '?')})")
    print(f"  ✓ Best val acc: {checkpoint.get('best_val_acc', '?'):.4f}")
    
    # Charger les données
    print("\n[2/3] Chargement des données...")
    loaders = get_data_loaders(
        root_dir=args.data_dir,
        batch_size=16,
        image_size=224,
        num_workers=4
    )
    
    # Créer l'évaluateur
    print("\n[3/3] Initialisation de l'évaluateur XAI...")
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = Path(args.save_dir) / f"{args.model}_{timestamp}"
    
    evaluator = XAIEvaluator(model, device, save_dir)
    
    # Générer le rapport
    evaluator.generate_report(
        dataloader=loaders['test'],
        num_samples=args.num_samples,
        num_visualizations=10
    )


if __name__ == "__main__":
    main()
