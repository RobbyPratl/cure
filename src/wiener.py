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
        """1/f² prior typical of natural images, normalized so mean = 1."""
        H, W = shape
        fy = torch.fft.fftfreq(H)[:, None]
        fx = torch.fft.fftfreq(W)[None, :]
        freq_mag = torch.sqrt(fx**2 + fy**2)
        freq_mag[0, 0] = 1e-10
        psd = 1.0 / (freq_mag ** 2 + 1e-6)
        return psd / psd.mean()  # Normalize by mean so signal_power scales average variance
    
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
        Draw samples from Gaussian posterior in frequency domain.
        
        The Wiener posterior is Gaussian with DIAGONAL covariance in frequency domain:
            mean(f) = W(f) · Y(f)
            var(f) = posterior_var_freq
        
        Key insight: ifft2 scales variance by 1/(H*W) due to Parseval's theorem.
        So to get spatial variance V, we need freq variance V * H * W.
        
        Args:
            degraded: (C, H, W) degraded image
            n_samples: Number of samples
        
        Returns:
            samples: (n_samples, C, H, W)
        """
        C, H, W = degraded.shape
        N = H * W  # Total number of pixels
        
        # Scale factor for frequency variance:
        # 1. Multiply by N because ifft2 divides by N, squaring to 1/N² for variance
        # 2. Multiply by 2 because we add non-Hermitian complex noise then take .real,
        #    which discards the imaginary part and loses half the variance
        freq_var_scaled = self.posterior_var_freq * N * 2
        
        samples = []
        for _ in range(n_samples):
            sample_channels = []
            for c in range(C):
                # Compute posterior mean in frequency domain
                Y_freq = torch.fft.fft2(degraded[c])
                mean_freq = self.W_freq * Y_freq
                
                # Sample complex Gaussian noise with scaled variance
                # For complex Gaussian: var(real) = var(imag) = total_var / 2
                noise_real = torch.randn(H, W) * torch.sqrt(freq_var_scaled / 2)
                noise_imag = torch.randn(H, W) * torch.sqrt(freq_var_scaled / 2)
                noise_freq = torch.complex(noise_real, noise_imag)
                
                # Sample = mean + noise in frequency domain
                sample_freq = mean_freq + noise_freq
                
                # Transform back to spatial domain
                sample_spatial = torch.fft.ifft2(sample_freq).real
                sample_channels.append(sample_spatial)
            
            samples.append(torch.stack(sample_channels, dim=0))
        
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
