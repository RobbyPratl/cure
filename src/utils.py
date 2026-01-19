"""Utility functions for I/O and visualization."""

import torch
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Union, Optional
import matplotlib.pyplot as plt


def load_image(path: Union[str, Path], size: Optional[int] = 256) -> torch.Tensor:
    """
    Load image as (C, H, W) tensor in [0, 1].
    
    Args:
        path: Path to image file
        size: Resize to (size, size). None = no resize.
    
    Returns:
        image: (C, H, W) float tensor in [0, 1]
    """
    img = Image.open(path).convert('RGB')
    if size is not None:
        img = img.resize((size, size), Image.LANCZOS)
    arr = np.array(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)


def save_image(tensor: torch.Tensor, path: Union[str, Path]):
    """Save (C, H, W) tensor as image file."""
    arr = tensor.detach().cpu().permute(1, 2, 0).numpy()
    arr = (arr.clip(0, 1) * 255).astype(np.uint8)
    Image.fromarray(arr).save(path)


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert (C, H, W) tensor to (H, W, C) numpy for plotting."""
    return tensor.detach().cpu().permute(1, 2, 0).numpy().clip(0, 1)


def compute_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Compute PSNR in dB."""
    mse = ((pred - target) ** 2).mean().item()
    if mse < 1e-10:
        return float('inf')
    return 10 * np.log10(1.0 / mse)


def show_images(images: list, titles: list = None, figsize=(15, 5), save_path=None):
    """Display multiple images in a row."""
    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=figsize)
    if n == 1:
        axes = [axes]
    for i, img in enumerate(images):
        if isinstance(img, torch.Tensor):
            img = to_numpy(img)
        axes[i].imshow(img)
        axes[i].axis('off')
        if titles:
            axes[i].set_title(titles[i])
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_frequency_response(kernel: torch.Tensor, image_shape=(256, 256), save_path=None):
    """Visualize blur kernel frequency response."""
    from degradations import get_kernel_frequency_response
    
    H_mag = get_kernel_frequency_response(kernel, image_shape)
    H_centered = torch.fft.fftshift(H_mag)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(kernel.numpy(), cmap='hot')
    axes[0].set_title('Kernel (spatial)')
    axes[0].axis('off')
    
    im = axes[1].imshow(H_centered.numpy(), cmap='hot', vmin=0, vmax=1)
    axes[1].set_title('|H(f)| Frequency Response')
    plt.colorbar(im, ax=axes[1])
    
    nulls = (H_centered < 0.1).float()
    axes[2].imshow(nulls.numpy(), cmap='Reds')
    axes[2].set_title(f'Nulls |H(f)|<0.1: {nulls.mean()*100:.1f}%')
    axes[2].axis('off')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()
