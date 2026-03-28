"""
Integrated Grad-CAM for Vision Transformers (IG-CAM-ViT)
=========================================================
Adapts IG-CAM to Vision Transformers by treating patch tokens
as spatial feature maps and integrating gradients along the
interpolation path from baseline to input.

Three strategies:
  - Strategy 1: IG on patch tokens (recommended)
  - Strategy 2: IG on CLS-to-patch attention weights
  - Strategy 3: Hybrid (tokens × attention)

Supports: ViT-B/16, ViT-L/16, DeiT-B, Swin Transformer

Author: [Your name]
"""

import torch
import torch.nn.functional as F
import numpy as np
import math
from typing import Optional, Tuple, Literal, List


class IGCAMViT:
    """
    Integrated Grad-CAM for Vision Transformers.

    Parameters
    ----------
    model : torch.nn.Module
        A ViT model (timm-style or torchvision-style).
    target_block : torch.nn.Module
        The last transformer block to hook into.
        Example (timm): model.blocks[-1]
        Example (torchvision): model.encoder.layers[-1]
    n_steps : int
        Number of interpolation steps (20-50 recommended).
    variant : str
        'A' = scalar weights (GAP of IG)
        'B' = pixel-wise weights (ReLU of IG)
        'C' = top-M dimensions only (reduces noise for high-D)
    strategy : str
        '1' = IG on patch tokens (default, recommended)
        '2' = IG on CLS-to-patch attention
        '3' = hybrid (tokens × attention)
    top_m : int or None
        For variant C: number of embedding dimensions to keep.
        If None, defaults to D // 4.
    num_patches_side : int or None
        sqrt(P) — the spatial grid side. If None, auto-detected.
    has_cls_token : bool
        Whether the model uses a CLS token (True for ViT, False for some variants).
    """

    def __init__(
        self,
        model: torch.nn.Module,
        target_block: torch.nn.Module,
        n_steps: int = 50,
        variant: Literal["A", "B", "C"] = "A",
        strategy: Literal["1", "2", "3"] = "1",
        top_m: Optional[int] = None,
        num_patches_side: Optional[int] = None,
        has_cls_token: bool = True,
    ):
        self.model = model
        self.target_block = target_block
        self.n_steps = n_steps
        self.variant = variant
        self.strategy = strategy
        self.top_m = top_m
        self.num_patches_side = num_patches_side
        self.has_cls_token = has_cls_token
        self.device = next(model.parameters()).device

        # Storage for hooks
        self._block_output: Optional[torch.Tensor] = None
        self._block_grad: Optional[torch.Tensor] = None
        self._attn_weights: Optional[torch.Tensor] = None
        self._attn_grad: Optional[torch.Tensor] = None

        # Register hooks on the target block output
        self._fwd_hook = target_block.register_forward_hook(self._save_block_output)
        self._bwd_hook = target_block.register_full_backward_hook(self._save_block_grad)

        # For strategy 2/3: hook into attention inside the target block
        if strategy in ("2", "3"):
            self._attn_hooks = self._register_attention_hooks(target_block)
        else:
            self._attn_hooks = []

    # ------------------------------------------------------------------
    # Hook callbacks
    # ------------------------------------------------------------------
    def _save_block_output(self, module, input, output):
        # Handle both tuple and tensor outputs
        if isinstance(output, tuple):
            self._block_output = output[0].detach()
        else:
            self._block_output = output.detach()

    def _save_block_grad(self, module, grad_input, grad_output):
        if isinstance(grad_output, tuple):
            self._block_grad = grad_output[0].detach()
        else:
            self._block_grad = grad_output.detach()

    def _register_attention_hooks(self, block) -> list:
        """
        Register hooks to capture attention weights and their gradients.
        Supports timm-style ViTs where block.attn.attn_drop exists.
        """
        hooks = []

        def save_attn(module, input, output):
            # For timm: the softmax output before dropout
            self._attn_weights = output.detach()

        def save_attn_grad(module, grad_input, grad_output):
            self._attn_grad = grad_output[0].detach()

        # Try common architectures for attention weight access
        attn_module = None
        if hasattr(block, 'attn'):
            attn = block.attn
            # timm-style: attn.attn_drop is right after softmax
            if hasattr(attn, 'attn_drop'):
                attn_module = attn.attn_drop
            # torchvision-style: self_attention
            elif hasattr(attn, 'self_attention'):
                attn_module = attn.self_attention

        if attn_module is not None:
            hooks.append(attn_module.register_forward_hook(save_attn))
            hooks.append(attn_module.register_full_backward_hook(save_attn_grad))

        return hooks

    # ------------------------------------------------------------------
    # Utility: detect patch grid size
    # ------------------------------------------------------------------
    def _get_grid_size(self, num_patches: int) -> Tuple[int, int]:
        if self.num_patches_side is not None:
            return (self.num_patches_side, self.num_patches_side)
        side = int(math.sqrt(num_patches))
        assert side * side == num_patches, (
            f"Cannot infer square grid from {num_patches} patches. "
            f"Set num_patches_side manually."
        )
        return (side, side)

    # ------------------------------------------------------------------
    # Core: extract patch tokens from block output
    # ------------------------------------------------------------------
    def _extract_patch_tokens(self, block_output: torch.Tensor) -> torch.Tensor:
        """
        Extract patch tokens from the block output, excluding CLS token.

        Parameters
        ----------
        block_output : (batch, seq_len, D)

        Returns
        -------
        patch_tokens : (batch, P, D) where P = seq_len - 1 if CLS exists
        """
        if self.has_cls_token:
            return block_output[:, 1:, :]  # exclude CLS at position 0
        return block_output

    def _reshape_to_grid(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        """
        Reshape (batch, P, D) → (batch, D, h_p, w_p)

        This is the key mapping: each embedding dimension d becomes
        a spatial "channel", analogous to CNN feature maps.
        """
        B, P, D = patch_tokens.shape
        h_p, w_p = self._get_grid_size(P)
        return patch_tokens.permute(0, 2, 1).reshape(B, D, h_p, w_p)

    # ------------------------------------------------------------------
    # Strategy 1: IG on patch tokens
    # ------------------------------------------------------------------
    def _compute_ig_patch_tokens(
        self,
        input_tensor: torch.Tensor,
        baseline: torch.Tensor,
        target_class: int,
        batch_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Integrated Gradients at the patch token level.

        Returns
        -------
        ig_grid : (D, h_p, w_p) — IG attribution per dimension per spatial position
        final_grid : (D, h_p, w_p) — patch tokens at input, reshaped to grid
        """
        alphas = torch.linspace(0, 1, self.n_steps + 1, device=self.device)
        delta = input_tensor.squeeze(0) - baseline.squeeze(0)

        all_grids = []
        all_grads = []

        for start in range(0, self.n_steps + 1, batch_size):
            end = min(start + batch_size, self.n_steps + 1)
            batch_alphas = alphas[start:end]

            # Interpolated inputs: (batch, C, H, W)
            interp = baseline + batch_alphas.view(-1, 1, 1, 1) * delta.unsqueeze(0)
            interp.requires_grad_(True)

            # Forward
            output = self.model(interp)
            scores = output[:, target_class]

            # Backward
            self.model.zero_grad()
            scores.sum().backward(retain_graph=False)

            # Extract patch tokens and reshape to grid
            patch_tokens = self._extract_patch_tokens(self._block_output)
            grids = self._reshape_to_grid(patch_tokens)  # (batch, D, h_p, w_p)

            patch_grads = self._extract_patch_tokens(self._block_grad)
            grad_grids = self._reshape_to_grid(patch_grads)

            all_grids.append(grids.clone())
            all_grads.append(grad_grids.clone())

        all_grids = torch.cat(all_grids, dim=0)  # (N+1, D, h_p, w_p)
        all_grads = torch.cat(all_grads, dim=0)  # (N+1, D, h_p, w_p)

        # Trapezoidal rule for integral of gradients
        avg_grads = (
            all_grads[0] / 2
            + all_grads[1:-1].sum(dim=0)
            + all_grads[-1] / 2
        ) / self.n_steps

        # IG = delta_tokens * avg_gradients
        delta_grids = all_grids[-1] - all_grids[0]  # (D, h_p, w_p)
        ig_grid = delta_grids * avg_grads  # (D, h_p, w_p)

        final_grid = all_grids[-1]  # (D, h_p, w_p)

        return ig_grid, final_grid

    # ------------------------------------------------------------------
    # Strategy 2: IG on attention weights
    # ------------------------------------------------------------------
    def _compute_ig_attention(
        self,
        input_tensor: torch.Tensor,
        baseline: torch.Tensor,
        target_class: int,
        batch_size: int,
    ) -> torch.Tensor:
        """
        Compute Integrated Gradients on CLS-to-patch attention weights.

        Returns
        -------
        ig_attn : (h_p, w_p) — attention-based attribution map
        """
        alphas = torch.linspace(0, 1, self.n_steps + 1, device=self.device)
        delta = input_tensor.squeeze(0) - baseline.squeeze(0)

        all_attns = []
        all_attn_grads = []

        for start in range(0, self.n_steps + 1, batch_size):
            end = min(start + batch_size, self.n_steps + 1)
            batch_alphas = alphas[start:end]

            interp = baseline + batch_alphas.view(-1, 1, 1, 1) * delta.unsqueeze(0)
            interp.requires_grad_(True)

            output = self.model(interp)
            scores = output[:, target_class]

            self.model.zero_grad()
            scores.sum().backward(retain_graph=False)

            if self._attn_weights is not None:
                # attn_weights: (batch, H_heads, seq, seq)
                # Extract CLS row (index 0), patch columns (index 1:)
                cls_attn = self._attn_weights[:, :, 0, 1:]  # (batch, H, P)
                all_attns.append(cls_attn.clone())

            if self._attn_grad is not None:
                cls_attn_grad = self._attn_grad[:, :, 0, 1:]  # (batch, H, P)
                all_attn_grads.append(cls_attn_grad.clone())

        if not all_attns or not all_attn_grads:
            raise RuntimeError(
                "Attention hooks did not capture data. "
                "Ensure the model architecture is supported."
            )

        all_attns = torch.cat(all_attns, dim=0)       # (N+1, H, P)
        all_attn_grads = torch.cat(all_attn_grads, dim=0)  # (N+1, H, P)

        # Trapezoidal rule
        avg_attn_grads = (
            all_attn_grads[0] / 2
            + all_attn_grads[1:-1].sum(dim=0)
            + all_attn_grads[-1] / 2
        ) / self.n_steps

        delta_attn = all_attns[-1] - all_attns[0]  # (H, P)
        ig_attn_per_head = delta_attn * avg_attn_grads  # (H, P)

        # Average over heads
        ig_attn = ig_attn_per_head.mean(dim=0)  # (P,)

        # Reshape to grid
        P = ig_attn.shape[0]
        h_p, w_p = self._get_grid_size(P)
        ig_attn = ig_attn.reshape(h_p, w_p)

        return ig_attn

    # ------------------------------------------------------------------
    # Weight computation (variants A, B, C)
    # ------------------------------------------------------------------
    def _compute_weights_and_cam(
        self,
        ig_grid: torch.Tensor,
        final_grid: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply variant A/B/C weighting to produce the saliency map.

        Parameters
        ----------
        ig_grid : (D, h_p, w_p)
        final_grid : (D, h_p, w_p)

        Returns
        -------
        cam : (h_p, w_p)
        """
        D = ig_grid.shape[0]

        if self.variant == "C":
            # Select top-M dimensions by total IG magnitude
            m = self.top_m if self.top_m is not None else D // 4
            dim_importance = ig_grid.abs().sum(dim=(-2, -1))  # (D,)
            _, top_indices = dim_importance.topk(m)
            ig_grid = ig_grid[top_indices]
            final_grid = final_grid[top_indices]

        if self.variant in ("A", "C"):
            # Scalar weight per dimension (GAP)
            weights = ig_grid.mean(dim=(-2, -1))  # (D,) or (M,)
            cam = (weights[:, None, None] * final_grid).sum(dim=0)
        else:
            # Variant B: pixel-wise weights
            weights = F.relu(ig_grid)
            cam = (weights * final_grid).sum(dim=0)

        cam = F.relu(cam)
        return cam

    # ------------------------------------------------------------------
    # Main generation method
    # ------------------------------------------------------------------
    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
        baseline: Optional[torch.Tensor] = None,
        batch_size: int = 4,
    ) -> np.ndarray:
        """
        Generate the IG-CAM saliency map for a Vision Transformer.

        Parameters
        ----------
        input_tensor : (1, C, H, W)
        target_class : int or None (auto-detect if None)
        baseline : (1, C, H, W) or None (black image)
        batch_size : int (smaller for ViTs due to memory)

        Returns
        -------
        saliency_map : np.ndarray (H, W), values in [0, 1]
        """
        self.model.eval()
        input_tensor = input_tensor.to(self.device)
        H, W = input_tensor.shape[2], input_tensor.shape[3]

        if baseline is None:
            baseline = torch.zeros_like(input_tensor)
        else:
            baseline = baseline.to(self.device)

        if target_class is None:
            with torch.no_grad():
                output = self.model(input_tensor)
                target_class = output.argmax(dim=1).item()

        # --- Strategy 1: IG on patch tokens ---
        if self.strategy in ("1", "3"):
            ig_grid, final_grid = self._compute_ig_patch_tokens(
                input_tensor, baseline, target_class, batch_size
            )
            cam_tokens = self._compute_weights_and_cam(ig_grid, final_grid)

        # --- Strategy 2/3: IG on attention ---
        if self.strategy in ("2", "3"):
            ig_attn = self._compute_ig_attention(
                input_tensor, baseline, target_class, batch_size
            )

        # --- Combine strategies ---
        if self.strategy == "1":
            cam = cam_tokens
        elif self.strategy == "2":
            cam = F.relu(ig_attn)
        else:  # strategy 3: hybrid
            # Normalize attention map to [0, 1]
            attn_norm = ig_attn - ig_attn.min()
            attn_max = attn_norm.max()
            if attn_max > 1e-8:
                attn_norm = attn_norm / attn_max
            # Multiply: regions must be important in BOTH token and attention space
            cam = cam_tokens * attn_norm

        # Upsample
        cam = cam.unsqueeze(0).unsqueeze(0)
        cam = F.interpolate(cam, size=(H, W), mode="bilinear", align_corners=False)
        cam = cam.squeeze()

        # Min-max normalize
        cam = cam.cpu().numpy()
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam

    # ------------------------------------------------------------------
    # Completeness verification
    # ------------------------------------------------------------------
    def verify_completeness(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
        baseline: Optional[torch.Tensor] = None,
        batch_size: int = 4,
    ) -> dict:
        """
        Verify: sum of IG attributions ≈ y^c(x) - y^c(x').
        Only valid for strategy 1 (patch tokens).
        """
        self.model.eval()
        input_tensor = input_tensor.to(self.device)

        if baseline is None:
            baseline = torch.zeros_like(input_tensor)
        else:
            baseline = baseline.to(self.device)

        if target_class is None:
            with torch.no_grad():
                output = self.model(input_tensor)
                target_class = output.argmax(dim=1).item()

        # Get scores at endpoints
        with torch.no_grad():
            score_input = self.model(input_tensor)[:, target_class].item()
            score_baseline = self.model(baseline)[:, target_class].item()

        output_diff = score_input - score_baseline

        # Compute IG
        ig_grid, _ = self._compute_ig_patch_tokens(
            input_tensor, baseline, target_class, batch_size
        )

        attribution_sum = ig_grid.sum().item()
        relative_error = abs(attribution_sum - output_diff) / (abs(output_diff) + 1e-10)

        return {
            "attribution_sum": attribution_sum,
            "output_diff": output_diff,
            "relative_error": relative_error,
            "n_steps": self.n_steps,
        }

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    def remove_hooks(self):
        self._fwd_hook.remove()
        self._bwd_hook.remove()
        for h in self._attn_hooks:
            h.remove()

    def __del__(self):
        try:
            self.remove_hooks()
        except Exception:
            pass


# ======================================================================
# Unified interface: auto-detect CNN vs ViT
# ======================================================================

def create_igcam(
    model: torch.nn.Module,
    model_type: Literal["cnn", "vit"] = "cnn",
    n_steps: int = 50,
    variant: str = "A",
    strategy: str = "1",
    **kwargs,
):
    """
    Factory function to create IG-CAM for either CNN or ViT.

    Parameters
    ----------
    model : the classification model
    model_type : 'cnn' or 'vit'
    n_steps : interpolation steps
    variant : 'A', 'B', or 'C' (C only for ViT)
    strategy : '1', '2', '3' (only for ViT)

    Returns
    -------
    IGCAM or IGCAMViT instance
    """
    if model_type == "cnn":
        # Auto-detect last conv layer for common architectures
        target_layer = _detect_cnn_target(model)
        # Import CNN version (from ig_cam.py)
        from ig_cam import IGCAM
        return IGCAM(model, target_layer, n_steps=n_steps, variant=variant)
    else:
        target_block = _detect_vit_target(model)
        return IGCAMViT(
            model, target_block,
            n_steps=n_steps, variant=variant, strategy=strategy,
            **kwargs,
        )


def _detect_cnn_target(model) -> torch.nn.Module:
    """Auto-detect the last conv layer for common CNN architectures."""
    name = model.__class__.__name__.lower()
    if "resnet" in name or "resnext" in name:
        return model.layer4[-1].conv3 if hasattr(model.layer4[-1], 'conv3') else model.layer4[-1]
    elif "vgg" in name:
        return model.features[-1]
    elif "densenet" in name:
        return model.features[-1]
    elif "efficientnet" in name:
        return model.features[-1][0]
    else:
        raise ValueError(
            f"Cannot auto-detect target layer for {name}. "
            f"Pass the layer manually."
        )


def _detect_vit_target(model) -> torch.nn.Module:
    """Auto-detect the last transformer block for common ViT architectures."""
    # timm-style ViT
    if hasattr(model, 'blocks'):
        return model.blocks[-1]
    # torchvision-style ViT
    if hasattr(model, 'encoder') and hasattr(model.encoder, 'layers'):
        return model.encoder.layers[-1]
    # DeiT (same as timm)
    if hasattr(model, 'blocks'):
        return model.blocks[-1]
    # Swin Transformer
    if hasattr(model, 'layers'):
        last_stage = model.layers[-1]
        if hasattr(last_stage, 'blocks'):
            return last_stage.blocks[-1]
    raise ValueError(
        f"Cannot auto-detect target block for {model.__class__.__name__}. "
        f"Pass the block manually."
    )


# ======================================================================
# Demo with timm ViT
# ======================================================================

def demo_vit():
    """
    Demo with ViT-B/16 from timm.
    Requires: pip install timm torchvision pillow matplotlib
    """
    import timm
    import torchvision.transforms as transforms
    from PIL import Image

    # --- Load model ---
    model = timm.create_model('vit_base_patch16_224', pretrained=True)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Target: last transformer block
    target_block = model.blocks[-1]

    # --- Preprocessing ---
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # --- Load image ---
    img = Image.open("sample_image.jpg").convert("RGB")
    input_tensor = preprocess(img).unsqueeze(0)

    viz_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    img_np = viz_transform(img).permute(1, 2, 0).numpy()

    # --- Strategy 1: IG on patch tokens ---
    print("=== Strategy 1: IG on patch tokens ===")
    for var_name, var in [("A (scalar)", "A"), ("B (pixel-wise)", "B"), ("C (top-M)", "C")]:
        igcam = IGCAMViT(
            model, target_block,
            n_steps=50, variant=var, strategy="1",
            num_patches_side=14, top_m=192,
        )
        cam = igcam.generate(input_tensor)
        print(f"  Variant {var_name}: cam range [{cam.min():.4f}, {cam.max():.4f}]")

        if var == "A":
            completeness = igcam.verify_completeness(input_tensor)
            print(f"  Completeness: sum={completeness['attribution_sum']:.4f}, "
                  f"diff={completeness['output_diff']:.4f}, "
                  f"error={completeness['relative_error']:.4%}")
        igcam.remove_hooks()

    # --- Strategy 2: IG on attention ---
    print("\n=== Strategy 2: IG on attention ===")
    igcam2 = IGCAMViT(
        model, target_block,
        n_steps=50, variant="A", strategy="2",
        num_patches_side=14,
    )
    cam2 = igcam2.generate(input_tensor)
    print(f"  Attention IG: cam range [{cam2.min():.4f}, {cam2.max():.4f}]")
    igcam2.remove_hooks()

    # --- Strategy 3: Hybrid ---
    print("\n=== Strategy 3: Hybrid (tokens × attention) ===")
    igcam3 = IGCAMViT(
        model, target_block,
        n_steps=50, variant="B", strategy="3",
        num_patches_side=14,
    )
    cam3 = igcam3.generate(input_tensor)
    print(f"  Hybrid: cam range [{cam3.min():.4f}, {cam3.max():.4f}]")
    igcam3.remove_hooks()

    # --- Using the factory function ---
    print("\n=== Factory function ===")
    igcam_auto = create_igcam(model, model_type="vit", n_steps=30, variant="A", strategy="1")
    cam_auto = igcam_auto.generate(input_tensor)
    print(f"  Auto-detect: cam range [{cam_auto.min():.4f}, {cam_auto.max():.4f}]")
    igcam_auto.remove_hooks()

    print("\nDone!")


if __name__ == "__main__":
    demo_vit()
