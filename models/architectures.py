"""
Model Architectures Module
==========================
Ce module contient les architectures CNN, Vision Transformer et Hybrides
pour la classification d'images médicales.

Auteur: [Votre Nom]
Date: 2025
"""

import torch
import torch.nn as nn
import timm
from typing import Optional, Dict, Any, List, Tuple
from abc import ABC, abstractmethod


class BaseModel(nn.Module, ABC):
    """Classe de base pour tous les modèles."""
    
    def __init__(self, num_classes: int, pretrained: bool = True):
        super().__init__()
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.model_type = "base"
        
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass
    
    @abstractmethod
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extraire les features avant la couche de classification."""
        pass
    
    def get_attention_maps(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        """Retourner les cartes d'attention (pour ViT uniquement)."""
        return None
    
    def get_target_layer(self):
        """Retourner la couche cible pour Grad-CAM."""
        raise NotImplementedError


# =============================================================================
# CNN MODELS
# =============================================================================

class ResNet50Model(BaseModel):
    """ResNet-50 pour classification d'images médicales."""
    
    def __init__(self, num_classes: int, pretrained: bool = True):
        super().__init__(num_classes, pretrained)
        self.model_type = "cnn"
        
        # Charger ResNet-50 pré-entraîné
        self.backbone = timm.create_model(
            'resnet50',
            pretrained=pretrained,
            num_classes=num_classes
        )
        
        # Stocker la dernière couche conv pour Grad-CAM
        self.target_layer = self.backbone.layer4[-1]
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extraire les features avant le global pooling."""
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.act1(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        return x
    
    def get_target_layer(self):
        return self.target_layer


class EfficientNetB4Model(BaseModel):
    """EfficientNet-B4 pour classification d'images médicales."""
    
    def __init__(self, num_classes: int, pretrained: bool = True):
        super().__init__(num_classes, pretrained)
        self.model_type = "cnn"
        
        self.backbone = timm.create_model(
            'efficientnet_b4',
            pretrained=pretrained,
            num_classes=num_classes
        )
        
        # Dernière couche conv
        self.target_layer = self.backbone.conv_head
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone.forward_features(x)
    
    def get_target_layer(self):
        return self.target_layer


# =============================================================================
# VISION TRANSFORMER MODELS
# =============================================================================

class ViTBaseModel(BaseModel):
    """Vision Transformer Base (ViT-B/16) pour classification."""
    
    def __init__(self, num_classes: int, pretrained: bool = True):
        super().__init__(num_classes, pretrained)
        self.model_type = "vit"
        
        # ViT-Base avec patch 16x16
        self.backbone = timm.create_model(
            'vit_base_patch16_224',
            pretrained=pretrained,
            num_classes=num_classes
        )
        
        # Stocker les attentions
        self.attention_weights = []
        self._register_attention_hooks()
        
    def _register_attention_hooks(self):
        """Enregistrer des hooks pour capturer les poids d'attention."""
        def get_attention_hook(module, input, output):
            # Pour ViT de timm, l'attention est dans output[1] si return_attn=True
            # Sinon, on doit la calculer manuellement
            pass
        
        # Les hooks seront utilisés par les méthodes XAI
        self.hooks = []
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
    
    def forward_with_attention(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Forward pass qui retourne aussi les cartes d'attention."""
        attentions = []
        
        # Embedding
        x = self.backbone.patch_embed(x)
        cls_token = self.backbone.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = self.backbone.pos_drop(x + self.backbone.pos_embed)
        
        # Passer à travers chaque bloc transformer
        for blk in self.backbone.blocks:
            # Calculer l'attention manuellement
            B, N, C = x.shape
            qkv = blk.attn.qkv(blk.norm1(x)).reshape(B, N, 3, blk.attn.num_heads, C // blk.attn.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            
            attn = (q @ k.transpose(-2, -1)) * blk.attn.scale
            attn = attn.softmax(dim=-1)
            attentions.append(attn)
            
            # Forward normal du bloc
            x = x + blk.drop_path1(blk.ls1(blk.attn(blk.norm1(x))))
            x = x + blk.drop_path2(blk.ls2(blk.mlp(blk.norm2(x))))
        
        # Classification
        x = self.backbone.norm(x)
        x = self.backbone.fc_norm(x[:, 0])
        x = self.backbone.head(x)
        
        return x, attentions
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extraire les features avant la tête de classification."""
        return self.backbone.forward_features(x)
    
    def get_attention_maps(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Retourner les cartes d'attention de toutes les couches."""
        _, attentions = self.forward_with_attention(x)
        return attentions
    
    def get_target_layer(self):
        """Pour Grad-CAM adapté aux ViT."""
        return self.backbone.blocks[-1].norm1


class DeiTBaseModel(BaseModel):
    """DeiT-Base (Data-efficient Image Transformer)."""
    
    def __init__(self, num_classes: int, pretrained: bool = True):
        super().__init__(num_classes, pretrained)
        self.model_type = "vit"
        
        self.backbone = timm.create_model(
            'deit_base_patch16_224',
            pretrained=pretrained,
            num_classes=num_classes
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone.forward_features(x)
    
    def get_target_layer(self):
        return self.backbone.blocks[-1].norm1


class SwinBaseModel(BaseModel):
    """Swin Transformer Base."""
    
    def __init__(self, num_classes: int, pretrained: bool = True):
        super().__init__(num_classes, pretrained)
        self.model_type = "vit"
        
        self.backbone = timm.create_model(
            'swin_base_patch4_window7_224',
            pretrained=pretrained,
            num_classes=num_classes
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone.forward_features(x)
    
    def get_target_layer(self):
        return self.backbone.layers[-1].blocks[-1].norm1


# =============================================================================
# HYBRID MODELS
# =============================================================================

class ConvNeXtBaseModel(BaseModel):
    """ConvNeXt-Base (CNN modernisé avec design inspiré des Transformers)."""
    
    def __init__(self, num_classes: int, pretrained: bool = True):
        super().__init__(num_classes, pretrained)
        self.model_type = "hybrid"
        
        self.backbone = timm.create_model(
            'convnext_base',
            pretrained=pretrained,
            num_classes=num_classes
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone.forward_features(x)
    
    def get_target_layer(self):
        return self.backbone.stages[-1].blocks[-1]


# =============================================================================
# MODEL FACTORY
# =============================================================================

class ModelFactory:
    """Factory pour créer les modèles."""
    
    MODELS = {
        # CNN
        'resnet50': ResNet50Model,
        'efficientnet_b4': EfficientNetB4Model,
        # ViT
        'vit_base': ViTBaseModel,
        'deit_base': DeiTBaseModel,
        'swin_base': SwinBaseModel,
        # Hybrid
        'convnext_base': ConvNeXtBaseModel,
    }
    
    @classmethod
    def create(
        cls,
        model_name: str,
        num_classes: int,
        pretrained: bool = True,
        **kwargs
    ) -> BaseModel:
        """
        Créer un modèle par son nom.
        
        Args:
            model_name: Nom du modèle (voir MODELS)
            num_classes: Nombre de classes de sortie
            pretrained: Utiliser les poids pré-entraînés
            
        Returns:
            Instance du modèle
        """
        if model_name not in cls.MODELS:
            available = list(cls.MODELS.keys())
            raise ValueError(f"Modèle '{model_name}' non reconnu. Disponibles: {available}")
        
        model_class = cls.MODELS[model_name]
        return model_class(num_classes=num_classes, pretrained=pretrained)
    
    @classmethod
    def list_models(cls) -> Dict[str, str]:
        """Lister tous les modèles disponibles avec leur type."""
        return {
            name: model_class(num_classes=2, pretrained=False).model_type
            for name, model_class in cls.MODELS.items()
        }


def get_model_summary(model: nn.Module, input_size: Tuple[int, int, int] = (3, 224, 224)):
    """Afficher un résumé du modèle."""
    from torchinfo import summary
    return summary(model, input_size=(1, *input_size), verbose=0)


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("TEST DES MODÈLES")
    print("=" * 60)
    
    # Tester la création de chaque modèle
    for model_name in ModelFactory.MODELS.keys():
        print(f"\n[TEST] Création de {model_name}...")
        try:
            model = ModelFactory.create(model_name, num_classes=8, pretrained=False)
            
            # Test forward pass
            dummy_input = torch.randn(2, 3, 224, 224)
            output = model(dummy_input)
            
            print(f"  ✓ Type: {model.model_type}")
            print(f"  ✓ Output shape: {output.shape}")
            print(f"  ✓ Target layer: {type(model.get_target_layer()).__name__}")
            
        except Exception as e:
            print(f"  ✗ Erreur: {e}")
    
    print("\n" + "=" * 60)
    print("MODÈLES DISPONIBLES:")
    print("=" * 60)
    for name, model_type in ModelFactory.list_models().items():
        print(f"  • {name}: {model_type}")
