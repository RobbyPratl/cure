"""Frequency domain utilities for spectral analysis."""

import torch
import numpy as np
from typing import Tuple, List


class FrequencyBands:
    """
    Partition Fourier space into radial frequency bands.
    
    Used for frequency-resolved calibration analysis.
    """
    
    def __init__(self, image_shape: Tuple[int, int], n_bands: int = 8):
        """
        Args:
            image_shape: (H, W)
            n_bands: Number of radial bands
        """
        H, W = image_shape
        self.image_shape = image_shape
        self.n_bands = n_bands
        
        # Frequency magnitude grid
        freq_y = torch.fft.fftfreq(H)[:, None].expand(H, W)
        freq_x = torch.fft.fftfreq(W)[None, :].expand(H, W)
        self.freq_mag = torch.sqrt(freq_x**2 + freq_y**2)
        
        # Band edges (linear spacing)
        max_freq = self.freq_mag.max().item()
        self.band_edges = np.linspace(0, max_freq, n_bands + 1)
        self.band_centers = (self.band_edges[:-1] + self.band_edges[1:]) / 2
        
        # Precompute masks
        self.masks = []
        for i in range(n_bands):
            mask = (self.freq_mag >= self.band_edges[i]) & (self.freq_mag < self.band_edges[i+1])
            self.masks.append(mask)
    
    def get_mask(self, band_idx: int) -> torch.Tensor:
        """Get boolean mask for frequency band."""
        return self.masks[band_idx]
    
    def extract_band(self, freq_data: torch.Tensor, band_idx: int) -> torch.Tensor:
        """Extract values from frequency data at given band."""
        return freq_data[self.masks[band_idx]]
    
    def apply_per_band(self, freq_data: torch.Tensor, func) -> List:
        """Apply function to each frequency band, return list of results."""
        results = []
        for i in range(self.n_bands):
            band_data = self.extract_band(freq_data, i)
            if len(band_data) > 0:
                results.append(func(band_data))
            else:
                results.append(None)
        return results


def to_frequency(image: torch.Tensor) -> torch.Tensor:
    """
    Transform image to frequency domain.
    
    Args:
        image: (C, H, W) or (H, W) spatial image
    
    Returns:
        freq: Complex tensor of same shape
    """
    if image.dim() == 3:
        return torch.fft.fft2(image)
    return torch.fft.fft2(image)


def to_spatial(freq: torch.Tensor) -> torch.Tensor:
    """Transform frequency data back to spatial domain."""
    return torch.fft.ifft2(freq).real


def get_magnitude(freq: torch.Tensor) -> torch.Tensor:
    """Get magnitude of complex frequency data."""
    return torch.abs(freq)


def get_phase(freq: torch.Tensor) -> torch.Tensor:
    """Get phase of complex frequency data."""
    return torch.angle(freq)
