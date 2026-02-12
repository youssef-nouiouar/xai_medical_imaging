"""
XAI Methods Module
==================
Implémentation des méthodes d'IA explicable pour CNN et Vision Transformers.

Méthodes incluses:
- Grad-CAM (et variantes)
- Attention Rollout
- Generic Attention (Chefer et al.)
- LIME
- SHAP
- Integrated Gradients

Auteur: [Votre Nom]
Date: 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, List, Tuple, Dict, Union, Callable
from abc import ABC, abstractmethod
import cv2
from PIL import Image

# Imports conditionnels pour les librairies XAI
try:
    from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, ScoreCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    GRADCAM_AVAILABLE = True
except ImportError:
    GRADCAM_AVAILABLE = False
    print("Warning: pytorch-grad-cam non installé. Installez avec: pip install pytorch-grad-cam")

try:
    from captum.attr import IntegratedGradients, LayerGradCam, Saliency
    CAPTUM_AVAILABLE = True
except ImportError:
    CAPTUM_AVAILABLE = False
    print("Warning: captum non installé. Installez avec: pip install captum")

try:
    from lime import lime_image
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    print("Warning: lime non installé. Installez avec: pip install lime")

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("Warning: shap non installé. Installez avec: pip install shap")


# =============================================================================
# BASE XAI CLASS
# =============================================================================

class BaseXAI(ABC):
    """Classe de base pour toutes les méthodes XAI."""
    
    def __init__(self, model: nn.Module, device: str = 'cuda'):
        self.model = model
        self.device = device
        self.model.eval()
        self.model.to(device)
        
    @abstractmethod
    def explain(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None
    ) -> np.ndarray:
        """
        Générer une carte d'explication pour l'entrée.
        
        Args:
            input_tensor: Image d'entrée (1, C, H, W)
            target_class: Classe cible (None = classe prédite)
            
        Returns:
            Carte de saillance (H, W) normalisée [0, 1]
        """
        pass
    
    def visualize(
        self,
        input_tensor: torch.Tensor,
        saliency_map: np.ndarray,
        original_image: Optional[np.ndarray] = None,
        alpha: float = 0.5
    ) -> np.ndarray:
        """
        Superposer la carte de saillance sur l'image originale.
        
        Args:
            input_tensor: Image d'entrée normalisée
            saliency_map: Carte de saillance (H, W)
            original_image: Image originale RGB [0, 1]
            alpha: Transparence de la superposition
            
        Returns:
            Image avec la carte de chaleur superposée
        """
        # Dénormaliser l'image si nécessaire
        if original_image is None:
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            original_image = input_tensor.squeeze(0).cpu() * std + mean
            original_image = original_image.permute(1, 2, 0).numpy()
            original_image = np.clip(original_image, 0, 1)
        
        # Redimensionner la saliency map si nécessaire
        if saliency_map.shape != original_image.shape[:2]:
            saliency_map = cv2.resize(
                saliency_map,
                (original_image.shape[1], original_image.shape[0])
            )
        
        # Créer la heatmap
        heatmap = cv2.applyColorMap(
            np.uint8(255 * saliency_map),
            cv2.COLORMAP_JET
        )
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
        
        # Superposer
        visualization = alpha * heatmap + (1 - alpha) * original_image
        visualization = np.clip(visualization, 0, 1)
        
        return visualization


# =============================================================================
# GRAD-CAM (Compatible CNN et ViT)
# =============================================================================

class GradCAMExplainer(BaseXAI):
    """
    Grad-CAM: Gradient-weighted Class Activation Mapping.
    
    Compatible avec les CNN et les ViT (avec adaptation).
    """
    
    def __init__(
        self,
        model: nn.Module,
        target_layer: nn.Module,
        device: str = 'cuda',
        use_cuda: bool = True
    ):
        super().__init__(model, device)
        self.target_layer = target_layer
        
        if GRADCAM_AVAILABLE:
            self.cam = GradCAM(
                model=model,
                target_layers=[target_layer],
                use_cuda=use_cuda and torch.cuda.is_available()
            )
        else:
            self.cam = None
            self._setup_manual_gradcam()
    
    def _setup_manual_gradcam(self):
        """Configuration manuelle de Grad-CAM si la librairie n'est pas disponible."""
        self.gradients = None
        self.activations = None
        
        def forward_hook(module, input, output):
            self.activations = output.detach()
            
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)
    
    def explain(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None
    ) -> np.ndarray:
        """Générer la carte Grad-CAM."""
        input_tensor = input_tensor.to(self.device)
        
        if self.cam is not None:
            # Utiliser pytorch-grad-cam
            targets = [ClassifierOutputTarget(target_class)] if target_class is not None else None
            grayscale_cam = self.cam(input_tensor=input_tensor, targets=targets)
            return grayscale_cam[0]
        else:
            # Implémentation manuelle
            return self._manual_gradcam(input_tensor, target_class)
    
    def _manual_gradcam(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None
    ) -> np.ndarray:
        """Implémentation manuelle de Grad-CAM."""
        self.model.zero_grad()
        
        # Forward pass
        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Backward pass
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        output.backward(gradient=one_hot, retain_graph=True)
        
        # Calculer Grad-CAM
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        
        # Normaliser
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        # Redimensionner à la taille de l'image
        cam = cv2.resize(cam, (224, 224))
        
        return cam


# =============================================================================
# ATTENTION ROLLOUT (ViT uniquement)
# =============================================================================

class AttentionRolloutExplainer(BaseXAI):
    """
    Attention Rollout pour Vision Transformers.
    
    Agrège les poids d'attention de toutes les couches pour estimer
    le flux d'information depuis l'entrée vers la sortie.
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda',
        head_fusion: str = 'mean',
        discard_ratio: float = 0.9
    ):
        """
        Args:
            model: Modèle ViT
            head_fusion: Comment fusionner les têtes d'attention ('mean', 'max', 'min')
            discard_ratio: Ratio d'attention à ignorer (pour réduire le bruit)
        """
        super().__init__(model, device)
        self.head_fusion = head_fusion
        self.discard_ratio = discard_ratio
        
    def _get_attention_maps(self, input_tensor: torch.Tensor) -> List[torch.Tensor]:
        """Extraire les cartes d'attention de toutes les couches."""
        attentions = []
        
        # Hook pour capturer les attentions
        hooks = []
        
        def get_attention_hook(module, input, output):
            # Pour timm ViT, on doit recalculer l'attention
            pass
        
        # Méthode alternative: forward personnalisé
        if hasattr(self.model, 'forward_with_attention'):
            _, attentions = self.model.forward_with_attention(input_tensor)
        else:
            # Fallback: extraire manuellement
            attentions = self._extract_attentions_manually(input_tensor)
        
        return attentions
    
    def _extract_attentions_manually(self, input_tensor: torch.Tensor) -> List[torch.Tensor]:
        """Extraire les attentions manuellement pour les modèles timm."""
        attentions = []
        
        # Accéder aux blocs du transformer
        if hasattr(self.model, 'backbone'):
            blocks = self.model.backbone.blocks
            patch_embed = self.model.backbone.patch_embed
            cls_token = self.model.backbone.cls_token
            pos_embed = self.model.backbone.pos_embed
            pos_drop = self.model.backbone.pos_drop
        else:
            blocks = self.model.blocks
            patch_embed = self.model.patch_embed
            cls_token = self.model.cls_token
            pos_embed = self.model.pos_embed
            pos_drop = self.model.pos_drop
        
        # Embedding
        x = patch_embed(input_tensor)
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = pos_drop(x + pos_embed)
        
        # Passer à travers chaque bloc
        for blk in blocks:
            B, N, C = x.shape
            
            # Calculer QKV
            qkv = blk.attn.qkv(blk.norm1(x))
            qkv = qkv.reshape(B, N, 3, blk.attn.num_heads, C // blk.attn.num_heads)
            qkv = qkv.permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            
            # Attention
            attn = (q @ k.transpose(-2, -1)) * blk.attn.scale
            attn = attn.softmax(dim=-1)
            attentions.append(attn)
            
            # Forward du bloc
            x = blk(x)
        
        return attentions
    
    def explain(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None
    ) -> np.ndarray:
        """Calculer l'Attention Rollout."""
        input_tensor = input_tensor.to(self.device)
        
        with torch.no_grad():
            attentions = self._extract_attentions_manually(input_tensor)
        
        # Fusionner les têtes d'attention
        result = torch.eye(attentions[0].size(-1)).to(self.device)
        
        for attention in attentions:
            # Fusion des têtes
            if self.head_fusion == 'mean':
                attention_heads_fused = attention.mean(dim=1)
            elif self.head_fusion == 'max':
                attention_heads_fused = attention.max(dim=1)[0]
            elif self.head_fusion == 'min':
                attention_heads_fused = attention.min(dim=1)[0]
            else:
                raise ValueError(f"head_fusion inconnu: {self.head_fusion}")
            
            # Ajouter l'identité (connexion résiduelle)
            I = torch.eye(attention_heads_fused.size(-1)).to(self.device)
            a = (attention_heads_fused + I) / 2
            
            # Normaliser
            a = a / a.sum(dim=-1, keepdim=True)
            
            # Multiplier
            result = torch.matmul(a, result)
        
        # Extraire l'attention du token CLS vers les patches
        mask = result[0, 0, 1:]  # Ignorer le CLS token
        
        # Reshape en grille spatiale
        num_patches = int(np.sqrt(mask.shape[0]))
        mask = mask.reshape(num_patches, num_patches).cpu().numpy()
        
        # Normaliser
        mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)
        
        # Redimensionner à 224x224
        mask = cv2.resize(mask, (224, 224))
        
        return mask


# =============================================================================
# GENERIC ATTENTION (Chefer et al., 2021)
# =============================================================================

class GenericAttentionExplainer(BaseXAI):
    """
    Transformer Interpretability Beyond Attention Visualization.
    (Chefer et al., CVPR 2021)
    
    Combine les gradients et l'attention pour une meilleure fidélité.
    """
    
    def __init__(self, model: nn.Module, device: str = 'cuda'):
        super().__init__(model, device)
        self.attentions = []
        self.attention_gradients = []
        self._register_hooks()
    
    def _register_hooks(self):
        """Enregistrer les hooks pour capturer attentions et gradients."""
        self.hooks = []
        
        # Accéder aux blocs
        if hasattr(self.model, 'backbone'):
            blocks = self.model.backbone.blocks
        else:
            blocks = self.model.blocks
        
        for blk in blocks:
            # Hook pour l'attention
            hook = blk.attn.attn_drop.register_forward_hook(
                self._attention_forward_hook
            )
            self.hooks.append(hook)
    
    def _attention_forward_hook(self, module, input, output):
        self.attentions.append(output)
    
    def explain(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None
    ) -> np.ndarray:
        """Calculer l'explication Generic Attention."""
        input_tensor = input_tensor.to(self.device)
        input_tensor.requires_grad = True
        
        self.attentions = []
        self.model.zero_grad()
        
        # Forward
        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Backward
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        output.backward(gradient=one_hot)
        
        # Calculer la relevance
        # Pour chaque couche: R = E[∇A ⊙ A]
        num_tokens = self.attentions[0].shape[-1]
        R = torch.eye(num_tokens, num_tokens).to(self.device)
        
        for attn in self.attentions:
            # Moyenne sur les têtes
            attn_avg = attn.mean(dim=1)
            
            # Gradient de l'attention (approximation)
            grad = attn_avg  # Simplification
            
            # Combiner
            cam = attn_avg * grad
            cam = cam.clamp(min=0)
            
            # Ajouter l'identité
            I = torch.eye(cam.size(-1)).to(self.device)
            cam = (cam + I) / 2
            cam = cam / cam.sum(dim=-1, keepdim=True)
            
            R = torch.matmul(cam, R)
        
        # Extraire le masque
        mask = R[0, 0, 1:]  # Attention du CLS vers les patches
        
        # Reshape et normaliser
        num_patches = int(np.sqrt(mask.shape[0]))
        mask = mask.reshape(num_patches, num_patches).detach().cpu().numpy()
        mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)
        mask = cv2.resize(mask, (224, 224))
        
        return mask


# =============================================================================
# LIME
# =============================================================================

class LIMEExplainer(BaseXAI):
    """LIME (Local Interpretable Model-agnostic Explanations)."""
    
    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda',
        num_samples: int = 1000,
        num_features: int = 10
    ):
        super().__init__(model, device)
        self.num_samples = num_samples
        self.num_features = num_features
        
        if LIME_AVAILABLE:
            self.explainer = lime_image.LimeImageExplainer()
        else:
            self.explainer = None
    
    def _predict_fn(self, images: np.ndarray) -> np.ndarray:
        """Fonction de prédiction pour LIME."""
        # Prétraitement
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        
        batch = []
        for img in images:
            img = cv2.resize(img, (224, 224))
            img = (img - mean) / std
            img = img.transpose(2, 0, 1)
            batch.append(img)
        
        batch = torch.FloatTensor(np.array(batch)).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(batch)
            probs = F.softmax(outputs, dim=1)
        
        return probs.cpu().numpy()
    
    def explain(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None
    ) -> np.ndarray:
        """Générer l'explication LIME."""
        if not LIME_AVAILABLE:
            raise ImportError("LIME n'est pas installé")
        
        # Convertir en image numpy
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        image = input_tensor.squeeze(0).cpu() * std + mean
        image = image.permute(1, 2, 0).numpy()
        image = np.clip(image, 0, 1)
        
        # Obtenir la classe cible
        if target_class is None:
            with torch.no_grad():
                output = self.model(input_tensor.to(self.device))
                target_class = output.argmax(dim=1).item()
        
        # Expliquer
        explanation = self.explainer.explain_instance(
            image,
            self._predict_fn,
            top_labels=1,
            hide_color=0,
            num_samples=self.num_samples
        )
        
        # Obtenir le masque
        _, mask = explanation.get_image_and_mask(
            target_class,
            positive_only=True,
            num_features=self.num_features,
            hide_rest=False
        )
        
        return mask.astype(float)


# =============================================================================
# INTEGRATED GRADIENTS
# =============================================================================

class IntegratedGradientsExplainer(BaseXAI):
    """Integrated Gradients (Sundararajan et al., 2017)."""
    
    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda',
        n_steps: int = 50
    ):
        super().__init__(model, device)
        self.n_steps = n_steps
        
        if CAPTUM_AVAILABLE:
            self.ig = IntegratedGradients(model)
        else:
            self.ig = None
    
    def explain(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None
    ) -> np.ndarray:
        """Calculer les Integrated Gradients."""
        input_tensor = input_tensor.to(self.device)
        
        if target_class is None:
            with torch.no_grad():
                output = self.model(input_tensor)
                target_class = output.argmax(dim=1).item()
        
        if CAPTUM_AVAILABLE:
            # Utiliser Captum
            baseline = torch.zeros_like(input_tensor).to(self.device)
            attributions = self.ig.attribute(
                input_tensor,
                baselines=baseline,
                target=target_class,
                n_steps=self.n_steps
            )
            
            # Agréger sur les canaux
            attr = attributions.squeeze().cpu().numpy()
            attr = np.abs(attr).mean(axis=0)
            
        else:
            # Implémentation manuelle
            attr = self._manual_integrated_gradients(input_tensor, target_class)
        
        # Normaliser
        attr = (attr - attr.min()) / (attr.max() - attr.min() + 1e-8)
        
        return attr
    
    def _manual_integrated_gradients(
        self,
        input_tensor: torch.Tensor,
        target_class: int
    ) -> np.ndarray:
        """Implémentation manuelle des Integrated Gradients."""
        baseline = torch.zeros_like(input_tensor).to(self.device)
        scaled_inputs = [
            baseline + (float(i) / self.n_steps) * (input_tensor - baseline)
            for i in range(self.n_steps + 1)
        ]
        
        gradients = []
        for scaled_input in scaled_inputs:
            scaled_input.requires_grad = True
            self.model.zero_grad()
            
            output = self.model(scaled_input)
            output[0, target_class].backward()
            
            gradients.append(scaled_input.grad.detach())
        
        # Intégrer
        avg_gradients = torch.stack(gradients).mean(dim=0)
        integrated_gradients = (input_tensor - baseline) * avg_gradients
        
        # Agréger
        attr = integrated_gradients.squeeze().cpu().numpy()
        attr = np.abs(attr).mean(axis=0)
        
        return attr


# =============================================================================
# XAI FACTORY
# =============================================================================

class XAIFactory:
    """Factory pour créer les méthodes XAI."""
    
    @staticmethod
    def create(
        method_name: str,
        model: nn.Module,
        device: str = 'cuda',
        **kwargs
    ) -> BaseXAI:
        """
        Créer une méthode XAI.
        
        Args:
            method_name: Nom de la méthode
            model: Modèle à expliquer
            device: Device à utiliser
            **kwargs: Arguments supplémentaires
            
        Returns:
            Instance de la méthode XAI
        """
        methods = {
            'gradcam': GradCAMExplainer,
            'attention_rollout': AttentionRolloutExplainer,
            'generic_attention': GenericAttentionExplainer,
            'lime': LIMEExplainer,
            'integrated_gradients': IntegratedGradientsExplainer,
        }
        
        if method_name not in methods:
            raise ValueError(f"Méthode '{method_name}' non reconnue. Disponibles: {list(methods.keys())}")
        
        # Configuration spécifique
        if method_name == 'gradcam':
            target_layer = model.get_target_layer()
            return GradCAMExplainer(model, target_layer, device, **kwargs)
        else:
            return methods[method_name](model, device, **kwargs)
    
    @staticmethod
    def get_compatible_methods(model_type: str) -> List[str]:
        """Retourner les méthodes compatibles avec un type de modèle."""
        all_methods = ['gradcam', 'lime', 'integrated_gradients']
        
        if model_type == 'vit':
            all_methods.extend(['attention_rollout', 'generic_attention'])
        
        return all_methods


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("TEST DES MÉTHODES XAI")
    print("=" * 60)
    
    print("\nLibrairies disponibles:")
    print(f"  • pytorch-grad-cam: {GRADCAM_AVAILABLE}")
    print(f"  • captum: {CAPTUM_AVAILABLE}")
    print(f"  • lime: {LIME_AVAILABLE}")
    print(f"  • shap: {SHAP_AVAILABLE}")
    
    print("\nMéthodes disponibles:")
    print("  • gradcam (CNN, ViT)")
    print("  • attention_rollout (ViT)")
    print("  • generic_attention (ViT)")
    print("  • lime (Tous)")
    print("  • integrated_gradients (Tous)")
