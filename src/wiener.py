"""
Wiener filter with closed-form posterior uncertainty.

This is the calibration reference for CURE - it's provably correct
under Gaussian assumptions.
"""

import torch
import numpy as np
from typing import Tuple, Optional
try:
    from .degradations import get_kernel_frequency_response
except ImportError:
    from degradations import get_kernel_frequency_response


class WienerFilter:
    """
    Wiener filter for image deblurring with exact posterior variance.
    
    Under linear-Gaussian model:
        y = Hx + n,  x ~ N(0, Cx),  n ~ N(0, σ²I)
    
    Posterior variance per frequency:
        Σ(f) = σ² · Sx(f) / [σ² + |H(f)|² · Sx(f)]
    
    Key insight: Variance is HIGH where |H(f)| ≈ 0 (information destroyed).
    """
    
    def __init__(
        self,
        kernel: torch.Tensor,
        noise_sigma: float,
        image_shape: Tuple[int, int],
        prior_type: str = 'empirical',
        signal_power: float = 0.1
    ):
        """
        Args:
            kernel: (kH, kW) blur kernel
            noise_sigma: Noise standard deviation σ
            image_shape: (H, W)
            prior_type: 'empirical' (1/f² natural image) or 'flat'
            signal_power: Approximate signal variance (power) for prior scaling
        """
        self.noise_sigma = noise_sigma
        self.noise_var = noise_sigma ** 2
        self.image_shape = image_shape
        self.signal_power = signal_power
        H, W = image_shape
        
        # Blur frequency response
        self.H_freq = self._kernel_to_freq(kernel, image_shape)
        self.H_mag_sq = torch.abs(self.H_freq) ** 2
        
        # Prior PSD - scale to represent actual signal power
        if prior_type == 'empirical':
            self.prior_psd = self._natural_image_prior(image_shape) * signal_power
        else:
            self.prior_psd = torch.ones(H, W) * signal_power
        
        # Wiener filter: W(f) = H*(f)·Sx(f) / [|H(f)|²·Sx(f) + σ²]
        denom = self.H_mag_sq * self.prior_psd + self.noise_var
        self.W_freq = torch.conj(self.H_freq) * self.prior_psd / (denom + 1e-10)
        
        # Posterior variance: Σ(f) = σ²·Sx(f) / [σ² + |H(f)|²·Sx(f)]
        self.posterior_var_freq = (self.noise_var * self.prior_psd) / (denom + 1e-10)
    
    def _kernel_to_freq(self, kernel: torch.Tensor, shape: Tuple[int, int]) -> torch.Tensor:
        """Convert spatial kernel to frequency response."""
        H, W = shape
        kH, kW = kernel.shape
        padded = torch.zeros(H, W, dtype=torch.float32)
        sh, sw = (H - kH) // 2, (W - kW) // 2
        padded[sh:sh+kH, sw:sw+kW] = kernel
        padded = torch.fft.ifftshift(padded)
        return torch.fft.fft2(padded)
    
    def _natural_image_prior(self, shape: Tuple[int, int]) -> torch.Tensor:
        """1/f² prior typical of natural images."""
        H, W = shape
        fy = torch.fft.fftfreq(H)[:, None]
        fx = torch.fft.fftfreq(W)[None, :]
        freq_mag = torch.sqrt(fx**2 + fy**2)
        freq_mag[0, 0] = 1e-10
        psd = 1.0 / (freq_mag ** 2 + 1e-6)
        return psd / psd.max()
    
    def restore(self, degraded: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Restore image using Wiener filter.
        
        Args:
            degraded: (C, H, W) degraded image
        
        Returns:
            restored: (C, H, W) restored image
            var_spatial: (H, W) spatial variance (approximate)
            var_freq: (H, W) frequency variance (exact)
        """
        C, H, W = degraded.shape
        
        restored = []
        for c in range(C):
            Y = torch.fft.fft2(degraded[c])
            X = self.W_freq * Y
            restored.append(torch.fft.ifft2(X).real)
        
        restored = torch.stack(restored, dim=0)
        var_spatial = torch.fft.ifft2(self.posterior_var_freq).real.abs()
        
        return restored, var_spatial, self.posterior_var_freq
    
    def sample_posterior(self, degraded: torch.Tensor, n_samples: int = 50) -> torch.Tensor:
        """
        Draw samples from Gaussian posterior.
        
        The posterior is Gaussian with diagonal covariance in the FREQUENCY domain.
        So we sample noise in frequency domain, scale by sqrt(posterior_var_freq),
        then transform back to spatial domain.
        
        Key scaling: ifft2 divides by sqrt(H*W), so we must scale up by sqrt(H*W)
        to preserve variance in spatial domain (Parseval's theorem).
        
        Args:
            degraded: (C, H, W) degraded image
            n_samples: Number of samples
        
        Returns:
            samples: (n_samples, C, H, W)
        """
        C, H, W = degraded.shape
        mean, _, var_freq = self.restore(degraded)
        
        # Standard deviation in frequency domain (where covariance is diagonal)
        std_freq = torch.sqrt(var_freq.clamp(min=1e-10))
        
        # Scale factor to preserve variance after ifft2 (Parseval's theorem)
        scale = np.sqrt(H * W)
        
        samples = []
        for _ in range(n_samples):
            # Sample complex Gaussian noise in frequency domain
            # Real and imaginary parts each have variance std_freq²/2 
            # so total variance is std_freq²
            noise_real = torch.randn(C, H, W) * (std_freq * scale / np.sqrt(2)).unsqueeze(0)
            noise_imag = torch.randn(C, H, W) * (std_freq * scale / np.sqrt(2)).unsqueeze(0)
            noise_freq = torch.complex(noise_real, noise_imag)
            
            # Transform noise to spatial domain (gives spatially correlated noise)
            noise_spatial = torch.fft.ifft2(noise_freq).real
            
            sample = mean + noise_spatial
            samples.append(sample)
        
        return torch.stack(samples, dim=0)
    
    def get_snr_per_frequency(self) -> torch.Tensor:
        """SNR(f) = |H(f)|²·Sx(f) / σ²"""
        return self.H_mag_sq * self.prior_psd / self.noise_var


def test_wiener():
    """Test Wiener filter implementation."""
    try:
        from .degradations import create_gaussian_kernel, degrade_image
    except ImportError:
        from degradations import create_gaussian_kernel, degrade_image
    
    print("Testing wiener.py...")
    
    # Create test image
    clean = torch.rand(3, 64, 64)
    result = degrade_image(clean, blur_sigma=2.0, noise_sigma=0.05, seed=42)
    
    # Create filter
    wf = WienerFilter(result['kernel'], 0.05, (64, 64))
    
    # Test restore
    restored, var_s, var_f = wf.restore(result['degraded'])
    assert restored.shape == clean.shape, "Shape mismatch"
    print("  ✓ restore() works")
    
    # Test sampling
    samples = wf.sample_posterior(result['degraded'], n_samples=10)
    assert samples.shape == (10, 3, 64, 64), "Sample shape wrong"
    print("  ✓ sample_posterior() works")
    
    # Test SNR
    snr = wf.get_snr_per_frequency()
    assert snr.shape == (64, 64), "SNR shape wrong"
    print("  ✓ get_snr_per_frequency() works")
    
    print("All Wiener tests passed! ✓")


if __name__ == "__main__":
    test_wiener()
