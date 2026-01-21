import sys
from pathlib import Path

import torch
import numpy as np

# Ensure src/ is importable when running tests directly
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.wiener import WienerFilter
from src.degradations import create_gaussian_kernel, degrade_image


def test_sampling_matches_posterior_variance_identity_kernel():
    torch.manual_seed(0)
    H = W = 32
    C = 1
    noise_sigma = 0.05
    signal_power = 0.08  # match typical clean variance used in notebook

    # Identity kernel (delta)
    kernel = torch.zeros(5, 5)
    kernel[2, 2] = 1.0

    # Synthetic clean image from the assumed prior scale
    clean = torch.randn(C, H, W) * np.sqrt(signal_power)
    degraded = clean + torch.randn_like(clean) * noise_sigma

    wf = WienerFilter(kernel, noise_sigma=noise_sigma, image_shape=(H, W), prior_type="flat", signal_power=signal_power)

    # Analytical posterior variance for identity kernel is constant across frequencies
    # posterior_var = sigma^2 * Sx / (sigma^2 + |H|^2 * Sx) with |H|=1
    expected_var = (noise_sigma ** 2 * signal_power) / (noise_sigma ** 2 + signal_power)

    # Sample posterior and estimate spatial variance
    samples = wf.sample_posterior(degraded, n_samples=400)
    sample_var = samples.var(dim=0).mean().item()

    assert np.isclose(sample_var, expected_var, rtol=0.15, atol=1e-3), (
        f"Sample variance {sample_var:.4f} deviates from expected {expected_var:.4f}")


def test_kernel_normalization_in_frequency_domain_matches_spatial_sum():
    torch.manual_seed(1)
    kernel = create_gaussian_kernel(sigma=1.2)
    wf = WienerFilter(kernel, noise_sigma=0.05, image_shape=(64, 64))

    # DC gain of H should equal kernel sum (which is 1.0)
    H_dc = wf.H_freq[0, 0].abs().item()
    assert np.isclose(H_dc, 1.0, atol=1e-3), f"DC gain should be 1, got {H_dc}" 
