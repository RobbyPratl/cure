"""
Degradation operators for image restoration experiments.

All functions are stateless: image in → degraded image out.
Works on torch tensors with shape (C, H, W) or (B, C, H, W).
Values expected in range [0, 1].

Usage:
    from degradations import degrade_image, create_gaussian_kernel
    
    kernel = create_gaussian_kernel(sigma=2.0)
    result = degrade_image(clean_image, blur_sigma=2.0, noise_sigma=0.05)
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict, Any


# BLUR KERNELS
# =============================================================================

def create_gaussian_kernel(sigma: float, kernel_size: Optional[int] = None) -> torch.Tensor:
    """
    Create 2D Gaussian blur kernel.
    
    Args:
        sigma: Standard deviation of Gaussian (controls blur strength)
        kernel_size: Size of kernel. Default: 6*sigma + 1 (covers 99.7% of distribution)
    
    Returns:
        kernel: (kernel_size, kernel_size) tensor, sums to 1
    
    Example:
        >>> kernel = create_gaussian_kernel(sigma=2.0)
        >>> kernel.shape
        torch.Size([13, 13])
        >>> kernel.sum()
        tensor(1.0000)
    """
    if kernel_size is None:
        kernel_size = int(6 * sigma) + 1
        if kernel_size % 2 == 0:
            kernel_size += 1  # Ensure odd size for symmetric kernel
    
    # Create coordinate grid centered at 0
    x = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
    
    # 1D Gaussian
    gauss_1d = torch.exp(-x**2 / (2 * sigma**2))
    
    # 2D Gaussian via outer product
    kernel = gauss_1d[:, None] * gauss_1d[None, :]
    
    # Normalize to sum to 1
    kernel = kernel / kernel.sum()
    
    return kernel


def create_motion_kernel(length: int, angle: float = 0.0) -> torch.Tensor:
    """
    Create motion blur kernel (linear motion).
    
    Args:
        length: Length of motion blur in pixels (must be odd)
        angle: Angle of motion in degrees (0 = horizontal right)
    
    Returns:
        kernel: (length, length) tensor, sums to 1
    
    Example:
        >>> kernel = create_motion_kernel(length=15, angle=45)
        >>> kernel.shape
        torch.Size([15, 15])
    """
    if length % 2 == 0:
        length += 1  # Ensure odd
    
    kernel = torch.zeros(length, length)
    center = length // 2
    
    # Create line at given angle using Bresenham-like approach
    angle_rad = np.radians(angle)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    
    for i in range(length):
        offset = i - center
        x = int(round(center + offset * cos_a))
        y = int(round(center + offset * sin_a))
        
        if 0 <= x < length and 0 <= y < length:
            kernel[y, x] = 1.0
    
    # Normalize
    kernel = kernel / kernel.sum()
    
    return kernel


def create_box_kernel(size: int) -> torch.Tensor:
    """
    Create box (uniform) blur kernel.
    
    Args:
        size: Size of the box kernel (must be odd)
    
    Returns:
        kernel: (size, size) tensor with uniform weights
    """
    if size % 2 == 0:
        size += 1
    
    kernel = torch.ones(size, size) / (size * size)
    return kernel


# =============================================================================
# BLUR APPLICATION
# =============================================================================

def apply_blur(
    image: torch.Tensor,
    kernel: torch.Tensor,
    mode: str = 'fft'
) -> torch.Tensor:
    """
    Apply blur kernel to image.
    
    Args:
        image: (C, H, W) or (B, C, H, W) tensor in [0, 1]
        kernel: (kH, kW) blur kernel (must sum to 1)
        mode: 
            'fft' - FFT-based circular convolution (default, matches Wiener filter)
            'reflect' - Spatial convolution with reflect padding
            'zero' - Spatial convolution with zero padding
    
    Returns:
        blurred: Same shape as input
    
    Note:
        Use 'fft' mode for consistency with Wiener filter analysis,
        as Wiener assumes circular boundary conditions.
    """
    # Handle batch dimension
    squeeze = False
    if image.dim() == 3:
        image = image.unsqueeze(0)
        squeeze = True
    
    B, C, H, W = image.shape
    kH, kW = kernel.shape
    device = image.device
    kernel = kernel.to(device)
    
    if mode == 'fft':
        # FFT-based convolution (circular boundary)
        blurred = _fft_conv2d(image, kernel)
    
    elif mode in ['reflect', 'zero']:
        # Spatial convolution
        pad_h, pad_w = kH // 2, kW // 2
        pad_mode = 'reflect' if mode == 'reflect' else 'constant'
        
        # Pad image
        image_padded = F.pad(image, (pad_w, pad_w, pad_h, pad_h), mode=pad_mode)
        
        # Reshape kernel for grouped conv: (C, 1, kH, kW)
        kernel_conv = kernel.unsqueeze(0).unsqueeze(0).repeat(C, 1, 1, 1)
        
        # Apply convolution (groups=C means each channel convolved separately)
        blurred = F.conv2d(image_padded, kernel_conv, groups=C)
    
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'fft', 'reflect', or 'zero'.")
    
    if squeeze:
        blurred = blurred.squeeze(0)
    
    return blurred


def _fft_conv2d(image: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """
    FFT-based 2D convolution with circular boundary.
    
    Internal function used by apply_blur.
    """
    B, C, H, W = image.shape
    kH, kW = kernel.shape
    device = image.device
    
    # Pad kernel to image size, centered
    kernel_padded = torch.zeros(H, W, device=device, dtype=image.dtype)
    
    # Place kernel in center
    start_h = (H - kH) // 2
    start_w = (W - kW) // 2
    kernel_padded[start_h:start_h + kH, start_w:start_w + kW] = kernel
    
    # Shift so kernel center is at (0, 0) for proper FFT convolution
    kernel_padded = torch.fft.ifftshift(kernel_padded)
    
    # FFT of kernel (same for all channels)
    K_fft = torch.fft.fft2(kernel_padded)
    
    # FFT of image, multiply, inverse FFT
    I_fft = torch.fft.fft2(image)
    blurred_fft = I_fft * K_fft.unsqueeze(0).unsqueeze(0)  # Broadcast over B, C
    blurred = torch.fft.ifft2(blurred_fft).real
    
    return blurred


# =============================================================================
# NOISE
# =============================================================================

def add_gaussian_noise(
    image: torch.Tensor,
    sigma: float,
    seed: Optional[int] = None,
    clip: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Add Gaussian noise to image.
    
    Args:
        image: (C, H, W) or (B, C, H, W) tensor in [0, 1]
        sigma: Noise standard deviation (e.g., 0.05 = moderate noise)
        seed: Random seed for reproducibility
        clip: Whether to clip output to [0, 1]
    
    Returns:
        noisy: Noisy image
        noise: The noise tensor that was added (useful for debugging)
    
    Example:
        >>> noisy, noise = add_gaussian_noise(image, sigma=0.05, seed=42)
        >>> noise.std()  # Should be approximately sigma
        tensor(0.0500)
    """
    if seed is not None:
        torch.manual_seed(seed)
    
    noise = torch.randn_like(image) * sigma
    noisy = image + noise
    
    if clip:
        noisy = noisy.clamp(0, 1)
    
    return noisy, noise


def add_poisson_noise(
    image: torch.Tensor,
    peak: float = 1.0,
    seed: Optional[int] = None
) -> torch.Tensor:
    """
    Add Poisson noise (shot noise) to image.
    
    Args:
        image: (C, H, W) tensor in [0, 1]
        peak: Peak value (higher = less noise)
        seed: Random seed
    
    Returns:
        noisy: Noisy image in [0, 1]
    """
    if seed is not None:
        torch.manual_seed(seed)
    
    # Scale to counts, apply Poisson, scale back
    scaled = image * peak
    noisy = torch.poisson(scaled) / peak
    noisy = noisy.clamp(0, 1)
    
    return noisy


# =============================================================================
# DOWNSAMPLING
# =============================================================================

def downsample(
    image: torch.Tensor,
    factor: int,
    mode: str = 'bilinear',
    antialias: bool = True
) -> torch.Tensor:
    """
    Downsample image by given factor.
    
    Args:
        image: (C, H, W) or (B, C, H, W) tensor
        factor: Downsampling factor (e.g., 2 = half resolution)
        mode: Interpolation mode ('bilinear', 'nearest', 'bicubic')
        antialias: Apply antialiasing filter before downsampling
    
    Returns:
        downsampled: Image at lower resolution
    """
    squeeze = False
    if image.dim() == 3:
        image = image.unsqueeze(0)
        squeeze = True
    
    B, C, H, W = image.shape
    new_H, new_W = H // factor, W // factor
    
    if antialias and mode != 'nearest':
        # Apply Gaussian blur before downsampling to prevent aliasing
        sigma = factor / 2
        kernel = create_gaussian_kernel(sigma)
        image = apply_blur(image, kernel, mode='fft')
    
    # Downsample
    downsampled = F.interpolate(
        image,
        size=(new_H, new_W),
        mode=mode,
        align_corners=False if mode != 'nearest' else None
    )
    
    if squeeze:
        downsampled = downsampled.squeeze(0)
    
    return downsampled


def upsample(
    image: torch.Tensor,
    factor: int,
    mode: str = 'bilinear'
) -> torch.Tensor:
    """
    Upsample image by given factor.
    
    Args:
        image: (C, H, W) or (B, C, H, W) tensor
        factor: Upsampling factor
        mode: Interpolation mode
    
    Returns:
        upsampled: Image at higher resolution
    """
    squeeze = False
    if image.dim() == 3:
        image = image.unsqueeze(0)
        squeeze = True
    
    B, C, H, W = image.shape
    new_H, new_W = H * factor, W * factor
    
    upsampled = F.interpolate(
        image,
        size=(new_H, new_W),
        mode=mode,
        align_corners=False if mode != 'nearest' else None
    )
    
    if squeeze:
        upsampled = upsampled.squeeze(0)
    
    return upsampled


# =============================================================================
# COMBINED DEGRADATION PIPELINE
# =============================================================================

def degrade_image(
    image: torch.Tensor,
    blur_sigma: float = 2.0,
    noise_sigma: float = 0.05,
    blur_type: str = 'gaussian',
    blur_mode: str = 'fft',
    seed: Optional[int] = None,
    clip_noise: bool = True
) -> Dict[str, Any]:
    """
    Apply full degradation pipeline: blur then noise.
    
    This is the main function you'll use for experiments.
    
    Args:
        image: (C, H, W) clean image tensor in [0, 1]
        blur_sigma: Blur kernel parameter
            - For 'gaussian': standard deviation
            - For 'motion': length in pixels
            - For 'box': kernel size
        noise_sigma: Additive Gaussian noise standard deviation
        blur_type: 'gaussian', 'motion', or 'box'
        blur_mode: 'fft', 'reflect', or 'zero' (boundary handling)
        seed: Random seed for reproducibility
    
    Returns:
        dict with keys:
            - 'degraded': (C, H, W) degraded image
            - 'clean': (C, H, W) original clean image (for convenience)
            - 'kernel': (kH, kW) blur kernel used
            - 'noise': (C, H, W) noise realization
            - 'blurred': (C, H, W) blurred image before noise (for debugging)
            - 'params': dict of degradation parameters
    
    Example:
        >>> result = degrade_image(clean, blur_sigma=3.0, noise_sigma=0.05, seed=42)
        >>> degraded = result['degraded']
        >>> kernel = result['kernel']  # Need this for Wiener filter
    """
    # Create blur kernel
    if blur_type == 'gaussian':
        kernel = create_gaussian_kernel(blur_sigma)
    elif blur_type == 'motion':
        kernel = create_motion_kernel(int(blur_sigma), angle=0)
    elif blur_type == 'box':
        kernel = create_box_kernel(int(blur_sigma))
    else:
        raise ValueError(f"Unknown blur_type: {blur_type}")
    
    # Apply blur (FFT mode for consistency with Wiener filter)
    blurred = apply_blur(image, kernel, mode=blur_mode)
    
    # Add noise
    degraded, noise = add_gaussian_noise(blurred, noise_sigma, seed=seed, clip=clip_noise)
    
    return {
        'degraded': degraded,
        'clean': image,
        'kernel': kernel,
        'noise': noise,
        'blurred': blurred,
        'params': {
            'blur_sigma': blur_sigma,
            'blur_type': blur_type,
            'blur_mode': blur_mode,
            'noise_sigma': noise_sigma,
            'seed': seed,
        }
    }


# =============================================================================
# FREQUENCY DOMAIN UTILITIES
# =============================================================================

def get_kernel_frequency_response(
    kernel: torch.Tensor,
    image_shape: Tuple[int, int]
) -> torch.Tensor:
    """
    Compute frequency response magnitude |H(f)| of blur kernel.
    
    This is essential for CURE analysis: we hypothesize that diffusion
    models are miscalibrated where |H(f)| ≈ 0 (blur nulls).
    
    Args:
        kernel: (kH, kW) blur kernel
        image_shape: (H, W) target image size
    
    Returns:
        H_magnitude: (H, W) tensor of |H(f)| in standard FFT layout
                     (DC at corner, use fftshift for visualization)
    
    Example:
        >>> kernel = create_gaussian_kernel(sigma=3.0)
        >>> H_mag = get_kernel_frequency_response(kernel, (256, 256))
        >>> 
        >>> # For visualization:
        >>> H_mag_centered = torch.fft.fftshift(H_mag)
        >>> plt.imshow(H_mag_centered)
    """
    H, W = image_shape
    kH, kW = kernel.shape
    
    # Pad kernel to image size
    kernel_padded = torch.zeros(H, W, dtype=kernel.dtype, device=kernel.device)
    start_h = (H - kH) // 2
    start_w = (W - kW) // 2
    kernel_padded[start_h:start_h + kH, start_w:start_w + kW] = kernel
    
    # Shift center to origin
    kernel_padded = torch.fft.ifftshift(kernel_padded)
    
    # Compute frequency response
    H_freq = torch.fft.fft2(kernel_padded)
    H_magnitude = torch.abs(H_freq)
    
    return H_magnitude


def get_frequency_grid(image_shape: Tuple[int, int]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create frequency coordinate grids.
    
    Args:
        image_shape: (H, W)
    
    Returns:
        freq_y: (H, W) vertical frequency coordinates
        freq_x: (H, W) horizontal frequency coordinates
    
    Note:
        Frequencies are in cycles per pixel, range [-0.5, 0.5].
        DC component is at (0, 0) corner in standard FFT layout.
    """
    H, W = image_shape
    freq_y = torch.fft.fftfreq(H)[:, None].expand(H, W)
    freq_x = torch.fft.fftfreq(W)[None, :].expand(H, W)
    
    return freq_y, freq_x


def get_frequency_magnitude(image_shape: Tuple[int, int]) -> torch.Tensor:
    """
    Create radial frequency magnitude grid.
    
    Args:
        image_shape: (H, W)
    
    Returns:
        freq_mag: (H, W) tensor of |f| = sqrt(fx² + fy²)
    """
    freq_y, freq_x = get_frequency_grid(image_shape)
    freq_mag = torch.sqrt(freq_x**2 + freq_y**2)
    return freq_mag


# =============================================================================
# TESTING / VALIDATION
# =============================================================================

def _test_degradations():
    """Run basic tests to validate implementations."""
    print("Testing degradations.py...")
    
    # Create test image (gradient)
    H, W = 64, 64
    x = torch.linspace(0, 1, W).unsqueeze(0).expand(H, W)
    y = torch.linspace(0, 1, H).unsqueeze(1).expand(H, W)
    test_image = torch.stack([x, y, (x + y) / 2], dim=0)  # (3, 64, 64)
    
    # Test 1: Gaussian kernel sums to 1
    kernel = create_gaussian_kernel(sigma=2.0)
    assert abs(kernel.sum().item() - 1.0) < 1e-6, "Kernel should sum to 1"
    print("Gaussian kernel sums to 1")
    
    # Test 2: Motion kernel sums to 1
    kernel_m = create_motion_kernel(length=11, angle=45)
    assert abs(kernel_m.sum().item() - 1.0) < 1e-6, "Motion kernel should sum to 1"
    print("Motion kernel sums to 1")
    
    # Test 3: Blur preserves mean (approximately)
    kernel = create_gaussian_kernel(sigma=2.0)
    blurred = apply_blur(test_image, kernel, mode='fft')
    assert blurred.shape == test_image.shape, "Blur should preserve shape"
    assert abs(blurred.mean() - test_image.mean()) < 0.01, "Blur should preserve mean"
    print("Blur preserves shape and mean")
    
    # Test 4: Noise has correct std
    noisy, noise = add_gaussian_noise(test_image, sigma=0.1, seed=42)
    measured_std = noise.std().item()
    assert abs(measured_std - 0.1) < 0.01, f"Noise std should be ~0.1, got {measured_std}"
    print("Noise has correct standard deviation")
    
    # Test 5: Full pipeline
    result = degrade_image(test_image, blur_sigma=2.0, noise_sigma=0.05, seed=42)
    assert 'degraded' in result, "Should return degraded image"
    assert 'kernel' in result, "Should return kernel"
    assert result['degraded'].shape == test_image.shape, "Shape should be preserved"
    print("Full degradation pipeline works")
    
    # Test 6: Frequency response
    H_mag = get_kernel_frequency_response(kernel, (64, 64))
    assert H_mag.shape == (64, 64), "Frequency response should match image size"
    assert H_mag[0, 0].item() > 0.99, "DC component should be ~1 for normalized kernel"
    print("Frequency response computation works")
    
    print("\nAll tests passed! ✓")


if __name__ == "__main__":
    _test_degradations()