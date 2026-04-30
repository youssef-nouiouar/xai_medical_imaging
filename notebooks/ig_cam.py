"""
Integrated Grad-CAM (IG-CAM / IG-CAM++)
========================================
A novel explainability method combining Integrated Gradients with Grad-CAM.

Computes integrated gradients at the last convolutional layer, then uses
either GAP-pooled scalar weights (IG-CAM, Variant A) or pixel-wise weights
(IG-CAM++, Variant B) to produce class-discriminative saliency maps.

Inherits axiomatic properties from Integrated Gradients:
  - Sensitivity: non-zero attribution when input differs from baseline
  - Completeness: attributions sum to f(x) - f(x')
  - Implementation invariance

Author: [youssef Nouiouar]
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Literal
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm


class IGCAM:
    """
    Integrated Grad-CAM.

    Parameters
    ----------
    model : torch.nn.Module
        A classification CNN (e.g. ResNet, VGG, EfficientNet).
    target_layer : torch.nn.Module
        The last convolutional layer to hook into.
        Example: model.layer4[-1] for ResNet, model.features[-1] for VGG.
    n_steps : int
        Number of interpolation steps for the integral approximation.
        Higher = more accurate but slower. Recommended: 20-50.
    variant : str
        'A' for IG-CAM (scalar weights via GAP),
        'B' for IG-CAM++ (pixel-wise weights).
    """

    def __init__(
        self,
        model: torch.nn.Module,
        target_layer: torch.nn.Module,
        n_steps: int = 50,
        variant: Literal["A", "B"] = "A",
    ):
        self.model = model
        self.target_layer = target_layer
        self.n_steps = n_steps
        self.variant = variant
        self.device = next(model.parameters()).device

        # Storage for hooks
        self._activations: Optional[torch.Tensor] = None
        self._gradients: Optional[torch.Tensor] = None

        # Register hooks
        self._forward_hook = target_layer.register_forward_hook(self._save_activation)
        self._backward_hook = target_layer.register_full_backward_hook(self._save_gradient)

    # ------------------------------------------------------------------
    # Hook callbacks
    # ------------------------------------------------------------------
    def _save_activation(self, module, input, output):
        self._activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self._gradients = grad_output[0].detach()

    # ------------------------------------------------------------------
    # Core computation
    # ------------------------------------------------------------------
    def _interpolate_inputs(
        self, input_tensor: torch.Tensor, baseline: torch.Tensor
    ) -> torch.Tensor:
        """
        Generate interpolated inputs along the straight-line path.

        x(t) = baseline + t * (input - baseline),  t in [0, 1]

        Returns
        -------
        interpolated : Tensor of shape (n_steps+1, C, H, W)
        """
        alphas = torch.linspace(0, 1, self.n_steps + 1, device=self.device)
        # Shape: (n_steps+1, 1, 1, 1) for broadcasting
        alphas = alphas.view(-1, 1, 1, 1)
        delta = input_tensor - baseline
        interpolated = baseline + alphas * delta  # (n_steps+1, C, H, W)
        return interpolated

    def _compute_gradients_along_path(
        self,
        interpolated_inputs: torch.Tensor,
        target_class: int,
        batch_size: int = 8,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        For each interpolated input x(t), compute:
          - A^k(t): feature maps at the target layer
          - dY^c / dA^k(t): gradients of class score w.r.t. feature maps

        Returns
        -------
        all_activations : (n_steps+1, K, h, w)
        all_gradients   : (n_steps+1, K, h, w)
        all_scores      : (n_steps+1,)
        """
        n_total = interpolated_inputs.shape[0]
        all_activations = []
        all_gradients = []
        all_scores = []

        # Process in batches to manage GPU memory
        for start in range(0, n_total, batch_size):
            end = min(start + batch_size, n_total)
            batch = interpolated_inputs[start:end].requires_grad_(True)

            # Forward pass
            output = self.model(batch)
            scores = output[:, target_class]

            # Backward pass — sum over batch to get gradients for each sample
            self.model.zero_grad()
            scores_sum = scores.sum()
            scores_sum.backward(retain_graph=False)

            all_activations.append(self._activations.clone())
            all_gradients.append(self._gradients.clone())
            all_scores.append(scores.detach())

        all_activations = torch.cat(all_activations, dim=0)  # (N+1, K, h, w)
        all_gradients = torch.cat(all_gradients, dim=0)      # (N+1, K, h, w)
        all_scores = torch.cat(all_scores, dim=0)            # (N+1,)

        return all_activations, all_gradients, all_scores

    def _compute_ig_at_conv_layer(
        self,
        all_activations: torch.Tensor,
        all_gradients: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute Integrated Gradients at the convolutional layer level.

        IG_k(i,j) = (A^k_{ij}(1) - A^k_{ij}(0)) * integral_0^1 dY^c/dA^k_{ij}(t) dt

        The integral is approximated using the trapezoidal rule.

        Parameters
        ----------
        all_activations : (n_steps+1, K, h, w)
        all_gradients   : (n_steps+1, K, h, w)

        Returns
        -------
        ig_conv : (K, h, w) — integrated gradient attribution per channel per pixel
        """
        # Trapezoidal rule for integral of gradients
        # avg_gradients ≈ (1/N) * sum_{n=1}^{N} grad(t_n)
        # Using trapezoidal: (grad[0]/2 + grad[1] + ... + grad[N-1] + grad[N]/2) / N
        avg_gradients = (
            all_gradients[0] / 2
            + all_gradients[1:-1].sum(dim=0)
            + all_gradients[-1] / 2
        ) / self.n_steps  # (K, h, w)

        # Delta of activations: A(input) - A(baseline)
        delta_activations = all_activations[-1] - all_activations[0]  # (K, h, w)

        # IG at conv layer
        ig_conv = delta_activations * avg_gradients  # (K, h, w)

        return ig_conv

    def _variant_a_weights(self, ig_conv: torch.Tensor) -> torch.Tensor:
        """
        Variant A (IG-CAM): Global Average Pooling of IG to get scalar weights.

        w_k = (1/Z) * sum_{i,j} IG_k(i,j)

        Returns
        -------
        weights : (K,) — one scalar weight per channel
        """
        weights = ig_conv.mean(dim=(-2, -1))  # (K,)
        return weights

    def _variant_b_weights(self, ig_conv: torch.Tensor) -> torch.Tensor:
        """
        Variant B (IG-CAM++): Pixel-wise weights with ReLU.

        w_k(i,j) = ReLU(IG_k(i,j))

        Only positive attributions contribute (inspired by Grad-CAM++).

        Returns
        -------
        weights : (K, h, w) — per-pixel per-channel weights
        """
        weights = F.relu(ig_conv)  # (K, h, w)
        return weights

    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
        baseline: Optional[torch.Tensor] = None,
        batch_size: int = 8,
    ) -> np.ndarray:
        """
        Generate the IG-CAM / IG-CAM++ saliency map.

        Parameters
        ----------
        input_tensor : Tensor of shape (1, C, H, W)
            The input image (preprocessed).
        target_class : int or None
            Class index. If None, uses the predicted class.
        baseline : Tensor or None
            Baseline image. If None, uses a black image (zeros).
        batch_size : int
            Batch size for processing interpolation steps.

        Returns
        -------
        saliency_map : np.ndarray of shape (H, W), values in [0, 1]
        """
        self.model.eval()
        input_tensor = input_tensor.to(self.device)
        H, W = input_tensor.shape[2], input_tensor.shape[3]

        # Default baseline: black image
        if baseline is None:
            baseline = torch.zeros_like(input_tensor)
        else:
            baseline = baseline.to(self.device)

        # Default target class: predicted class
        if target_class is None:
            with torch.no_grad():
                output = self.model(input_tensor)
                target_class = output.argmax(dim=1).item()

        # Step 1: Interpolate inputs along the path
        interpolated = self._interpolate_inputs(
            input_tensor.squeeze(0), baseline.squeeze(0)
        )  # (N+1, C, H, W)

        # Step 2: Compute activations and gradients along the path
        all_activations, all_gradients, all_scores = self._compute_gradients_along_path(
            interpolated, target_class, batch_size
        )

        # Step 3: Compute Integrated Gradients at conv layer
        ig_conv = self._compute_ig_at_conv_layer(all_activations, all_gradients)

        # Step 4: Compute weights (Variant A or B)
        final_activations = all_activations[-1]  # (K, h, w) — activations at input

        if self.variant == "A":
            weights = self._variant_a_weights(ig_conv)  # (K,)
            # Weighted sum: sum_k w_k * A^k
            cam = (weights[:, None, None] * final_activations).sum(dim=0)  # (h, w)
        else:
            weights = self._variant_b_weights(ig_conv)  # (K, h, w)
            # Element-wise product then sum: sum_k w_k(i,j) * A^k(i,j)
            cam = (weights * final_activations).sum(dim=0)  # (h, w)

        # ReLU: keep only positive contributions
        cam = F.relu(cam)

        # Step 5: Upsample to input resolution
        cam = cam.unsqueeze(0).unsqueeze(0)  # (1, 1, h, w)
        cam = F.interpolate(cam, size=(H, W), mode="bilinear", align_corners=False)
        cam = cam.squeeze()  # (H, W)

        # Min-max normalization
        cam = cam.cpu().numpy()
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam

    def verify_completeness(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
        baseline: Optional[torch.Tensor] = None,
        batch_size: int = 8,
    ) -> dict:
        """
        Verify the completeness axiom:
            sum_{k,i,j} IG_k(i,j) ≈ y^c(x) - y^c(x')

        Returns a dict with the sum of attributions, the output difference,
        and the relative error.
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

        interpolated = self._interpolate_inputs(
            input_tensor.squeeze(0), baseline.squeeze(0)
        )

        all_activations, all_gradients, all_scores = self._compute_gradients_along_path(
            interpolated, target_class, batch_size
        )

        ig_conv = self._compute_ig_at_conv_layer(all_activations, all_gradients)

        attribution_sum = ig_conv.sum().item()
        output_diff = (all_scores[-1] - all_scores[0]).item()

        relative_error = abs(attribution_sum - output_diff) / (abs(output_diff) + 1e-10)

        return {
            "attribution_sum": attribution_sum,
            "output_diff_yc_x_minus_yc_baseline": output_diff,
            "relative_error": relative_error,
            "n_steps": self.n_steps,
        }

    def remove_hooks(self):
        """Remove all registered hooks. Call when done."""
        self._forward_hook.remove()
        self._backward_hook.remove()

    def __del__(self):
        try:
            self.remove_hooks()
        except Exception:
            pass


# ======================================================================
# Visualization utilities
# ======================================================================

def overlay_cam_on_image(
    image: np.ndarray,
    cam: np.ndarray,
    alpha: float = 0.5,
    colormap: str = "jet",
) -> np.ndarray:
    """
    Overlay the saliency map on the original image.

    Parameters
    ----------
    image : np.ndarray (H, W, 3), values in [0, 1]
    cam : np.ndarray (H, W), values in [0, 1]
    alpha : float — blending factor
    colormap : str — matplotlib colormap name

    Returns
    -------
    overlay : np.ndarray (H, W, 3), values in [0, 1]
    """
    cmap = cm.get_cmap(colormap)
    heatmap = cmap(cam)[:, :, :3]  # (H, W, 3)
    overlay = alpha * heatmap + (1 - alpha) * image
    overlay = np.clip(overlay, 0, 1)
    return overlay


def visualize_comparison(
    image: np.ndarray,
    cam_a: np.ndarray,
    cam_b: np.ndarray,
    cam_gradcam: Optional[np.ndarray] = None,
    cam_ig: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
):
    """
    Side-by-side comparison of IG-CAM variants and optionally Grad-CAM / IG.
    """
    methods = [("IG-CAM (A)", cam_a), ("IG-CAM++ (B)", cam_b)]
    if cam_gradcam is not None:
        methods.append(("Grad-CAM", cam_gradcam))
    if cam_ig is not None:
        methods.append(("Integrated Gradients", cam_ig))

    n = len(methods) + 1
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))

    axes[0].imshow(image)
    axes[0].set_title("Original")
    axes[0].axis("off")

    for i, (name, cam) in enumerate(methods):
        overlay = overlay_cam_on_image(image, cam)
        axes[i + 1].imshow(overlay)
        axes[i + 1].set_title(name)
        axes[i + 1].axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


# ======================================================================
# Example usage with a pretrained ResNet
# ======================================================================

def demo():
    """
    Demonstration with ResNet-50 on a sample image.
    Requires: torchvision, PIL
    """
    import torchvision.models as models
    import torchvision.transforms as transforms

    # --- Load model ---
    model = models.resnet50(pretrained=True)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Target layer: last conv block of ResNet-50
    target_layer = model.layer4[-1].conv3
    # Alternative: model.layer4[-1] for the entire bottleneck block

    # --- Preprocessing ---
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    # --- Load image ---
    # Replace with your image path
    img = Image.open("sample_image.jpg").convert("RGB")
    input_tensor = preprocess(img).unsqueeze(0)  # (1, 3, 224, 224)

    # For visualization: unnormalized image
    viz_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    img_np = viz_transform(img).permute(1, 2, 0).numpy()

    # --- Generate saliency maps ---
    print("Computing IG-CAM (Variant A)...")
    igcam_a = IGCAM(model, target_layer, n_steps=50, variant="A")
    cam_a = igcam_a.generate(input_tensor, target_class=None)

    # Verify completeness axiom
    completeness = igcam_a.verify_completeness(input_tensor)
    print(f"  Completeness check:")
    print(f"    Attribution sum:  {completeness['attribution_sum']:.4f}")
    print(f"    Output diff:     {completeness['output_diff_yc_x_minus_yc_baseline']:.4f}")
    print(f"    Relative error:  {completeness['relative_error']:.6f}")
    igcam_a.remove_hooks()

    print("Computing IG-CAM++ (Variant B)...")
    igcam_b = IGCAM(model, target_layer, n_steps=50, variant="B")
    cam_b = igcam_b.generate(input_tensor, target_class=None)
    igcam_b.remove_hooks()

    # --- Visualize ---
    visualize_comparison(img_np, cam_a, cam_b, save_path="igcam_comparison.png")
    print("Saved to igcam_comparison.png")


if __name__ == "__main__":
    demo()
