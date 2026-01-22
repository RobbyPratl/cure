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
        signal_power: float = 0.1,
        prior_psd: Optional[torch.Tensor] = None,
        center_mode: str = 'none',
        variance_scale: float = 1.0
    ):
        """
        Args:
            kernel: (kH, kW) blur kernel
            noise_sigma: Noise standard deviation σ
            image_shape: (H, W)
            prior_type: 'empirical' (1/f² natural image) or 'flat'
            signal_power: Approximate signal variance (power) for prior scaling
            prior_psd: Optional PSD override (H, W) or (C, H, W). If provided,
                this is used directly instead of the analytic prior.
            center_mode: 'none' or 'channel_mean'. If 'channel_mean', the
                per-channel spatial mean of the degraded image is subtracted
                before filtering and added back to the posterior mean.
            variance_scale: Optional multiplicative calibration factor for
                posterior variance used in sampling. Default 1.0 (no scaling).
        """
        self.noise_sigma = noise_sigma
        self.noise_var = noise_sigma ** 2
        self.image_shape = image_shape
        self.signal_power = signal_power
        if center_mode not in {'none', 'channel_mean'}:
            raise ValueError("center_mode must be 'none' or 'channel_mean'")
        self.center_mode = center_mode
        self.variance_scale = float(variance_scale)
        H, W = image_shape
        
        # Blur frequency response
        self.H_freq = self._kernel_to_freq(kernel, image_shape)
        self.H_mag_sq = torch.abs(self.H_freq) ** 2
        
        # Prior PSD - scale to represent actual signal power
        if prior_psd is not None:
            psd = prior_psd
            if psd.dim() == 3:
                if psd.shape[1:] != (H, W):
                    raise ValueError(f"prior_psd must have shape (C, {H}, {W})")
                self.prior_psd = psd.real.to(self.H_freq.device)
            else:
                if psd.shape != (H, W):
                    raise ValueError(f"prior_psd must have shape {(H, W)} or (C, H, W)")
                self.prior_psd = psd.real.to(self.H_freq.device)
        elif prior_type == 'empirical':
            self.prior_psd = self._natural_image_prior(image_shape) * signal_power
        else:
            self.prior_psd = torch.ones(H, W, device=self.H_freq.device) * signal_power

        self.per_channel_prior = (self.prior_psd.dim() == 3)
        
        # Wiener filter: W(f) = H*(f)·Sx(f) / [|H(f)|²·Sx(f) + σ²]
        if self.per_channel_prior:
            H_mag_sq = self.H_mag_sq.unsqueeze(0)
            H_freq = self.H_freq.unsqueeze(0)
            denom = H_mag_sq * self.prior_psd + self.noise_var
            self.W_freq = torch.conj(H_freq) * self.prior_psd / (denom + 1e-10)
            # Posterior variance: Σ(f) = σ²·Sx(f) / [σ² + |H(f)|²·Sx(f)]
            self.posterior_var_freq = (self.noise_var * self.prior_psd) / (denom + 1e-10)
        else:
            denom = self.H_mag_sq * self.prior_psd + self.noise_var
            self.W_freq = torch.conj(self.H_freq) * self.prior_psd / (denom + 1e-10)
            # Posterior variance: Σ(f) = σ²·Sx(f) / [σ² + |H(f)|²·Sx(f)]
            self.posterior_var_freq = (self.noise_var * self.prior_psd) / (denom + 1e-10)
    
    def _kernel_to_freq(self, kernel: torch.Tensor, shape: Tuple[int, int]) -> torch.Tensor:
        """Convert spatial kernel to frequency response.
        
        For proper FFT alignment, we place the kernel center at the origin (0,0)
        of the padded array. This is the standard convention for convolution
        kernels in frequency domain - no ifftshift needed.
        """
        H, W = shape
        kH, kW = kernel.shape
        padded = torch.zeros(H, W, dtype=torch.float32)
        
        # Place kernel with its center at origin (0,0) using wrap-around indexing.
        # For a kernel of size (kH, kW), the center is at (kH//2, kW//2).
        # We want this center pixel to land at padded[0, 0].
        center_h, center_w = kH // 2, kW // 2
        for i in range(kH):
            for j in range(kW):
                # Destination indices with wrap-around
                di = (i - center_h) % H
                dj = (j - center_w) % W
                padded[di, dj] = kernel[i, j]
        
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

        if self.center_mode == 'channel_mean':
            mean = degraded.mean(dim=(1, 2), keepdim=True)
            degraded = degraded - mean
        else:
            mean = None
        
        restored = []
        for c in range(C):
            Y = torch.fft.fft2(degraded[c])
            Wc = self.W_freq[c] if self.per_channel_prior else self.W_freq
            X = Wc * Y
            restored.append(torch.fft.ifft2(X).real)
        
        restored = torch.stack(restored, dim=0)
        if mean is not None:
            restored = restored + mean
        # For a shift-invariant blur and stationary prior/noise, the posterior
        # covariance is also stationary. The per-pixel variance is the average
        # power of the posterior PSD (zero-lag of the autocovariance), which is
        # simply its mean value.
        posterior_var = self.posterior_var_freq
        if posterior_var.dim() == 3:
            posterior_var = posterior_var.mean(dim=0)
        var_scalar = torch.mean(posterior_var.real)
        var_spatial = torch.full((H, W), float(var_scalar))
        
        return restored, var_spatial, posterior_var
    
    def sample_posterior(self, degraded: torch.Tensor, n_samples: int = 50) -> torch.Tensor:
        """
        Draw samples from the exact Gaussian posterior using frequency-domain sampling.

        For a linear-Gaussian model y = Hx + n with stationary prior and noise,
        the posterior is Gaussian with:
          - Mean: W(f) * Y(f) in frequency domain
          - Covariance: stationary with PSD = posterior_var_freq

        The posterior_var_freq is the power spectral density (PSD) of the error.
        For a stationary process, the spatial variance at any pixel equals the
        mean of the PSD. To sample correctly in frequency domain:
        
        1. The PSD tells us E[|X(f)|²] for each frequency
        2. For fft2 with "backward" normalization (default), irfft2 divides by N=H*W
        3. So we need to scale the frequency-domain noise by sqrt(H*W) to get
           the correct spatial variance after irfft2.

        Args:
            degraded: (C, H, W) degraded image
            n_samples: Number of samples

        Returns:
            samples: (n_samples, C, H, W)
        """
        C, H, W = degraded.shape
        N = H * W

        if self.center_mode == 'channel_mean':
            mean = degraded.mean(dim=(1, 2), keepdim=True)
            degraded = degraded - mean
        else:
            mean = None

        # Posterior mean per channel in spatial domain
        means = []
        for c in range(C):
            Y = torch.fft.fft2(degraded[c])
            X = self.W_freq * Y
            means.append(torch.fft.ifft2(X).real)
        mean_spatial = torch.stack(means, dim=0)
        if mean is not None:
            mean_spatial = mean_spatial + mean

        # For sampling from a stationary Gaussian with PSD S(f):
        # - Generate complex Gaussian Z(f) with E[|Z(f)|²] = S(f) * N (for fft normalization)
        # - Apply irfft2 to get spatial samples with correct variance
        #
        # Using rfft2/irfft2 for efficiency (handles Hermitian symmetry automatically)
        posterior_var = self.posterior_var_freq * self.variance_scale
        if posterior_var.dim() == 3:
            var_freq_rfft = posterior_var[:, :, :W//2 + 1]
        else:
            var_freq_rfft = posterior_var[:, :W//2 + 1]
        
        # Scale by N for FFT normalization (irfft2 divides by N)
        std_freq = torch.sqrt(var_freq_rfft.real * N + 1e-12)
        
        samples = []
        for _ in range(n_samples):
            sample_channels = []
            for c in range(C):
                # Generate complex Gaussian noise in rfft domain
                # Real and imag parts each have variance var/2 to get total var
                std_f = std_freq[c] if std_freq.dim() == 3 else std_freq
                noise_real = torch.randn(H, W//2 + 1) * (std_f / np.sqrt(2))
                noise_imag = torch.randn(H, W//2 + 1) * (std_f / np.sqrt(2))
                noise_freq = torch.complex(noise_real, noise_imag)
                
                # DC and Nyquist frequencies (if W is even) must be real
                # These frequencies have no imaginary counterpart, so full variance goes to real
                noise_freq[0, 0] = noise_freq[0, 0].real * np.sqrt(2)
                if W % 2 == 0:
                    noise_freq[0, W//2] = noise_freq[0, W//2].real * np.sqrt(2)
                if H % 2 == 0:
                    noise_freq[H//2, 0] = noise_freq[H//2, 0].real * np.sqrt(2)
                    if W % 2 == 0:
                        noise_freq[H//2, W//2] = noise_freq[H//2, W//2].real * np.sqrt(2)
                
                # Transform to spatial domain
                noise_spatial = torch.fft.irfft2(noise_freq, s=(H, W))
                sample_channels.append(mean_spatial[c] + noise_spatial)
            samples.append(torch.stack(sample_channels, dim=0))
        
        return torch.stack(samples, dim=0)
    
    def get_snr_per_frequency(self) -> torch.Tensor:
        """SNR(f) = |H(f)|²·Sx(f) / σ²"""
        if self.per_channel_prior:
            return self.H_mag_sq.unsqueeze(0) * self.prior_psd / self.noise_var
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
