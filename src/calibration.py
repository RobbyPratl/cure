"""
Calibration metrics for uncertainty quantification.

Frequency-resolved calibration is the KEY NOVEL CONTRIBUTION of CURE.
"""

import torch
import numpy as np
from typing import Tuple, Optional
from .frequency import FrequencyBands


def compute_ece(
    pred_variance: torch.Tensor,
    true_error: torch.Tensor,
    n_bins: int = 10
) -> Tuple[float, np.ndarray]:
    """
    Expected Calibration Error for regression.
    
    For calibrated predictions: E[error² | pred_var] ≈ pred_var
    
    Args:
        pred_variance: Predicted variance
        true_error: Actual squared error (pred - truth)²
        n_bins: Number of calibration bins
    
    Returns:
        ece: Expected Calibration Error (lower = better)
        reliability: (n_bins, 3) of [avg_pred, avg_true, count]
    """
    pred = pred_variance.flatten().numpy()
    err = true_error.flatten().numpy()
    
    bin_edges = np.percentile(pred, np.linspace(0, 100, n_bins + 1))
    bin_edges[-1] += 1e-10
    
    reliability = np.zeros((n_bins, 3))
    ece = 0.0
    
    for i in range(n_bins):
        mask = (pred >= bin_edges[i]) & (pred < bin_edges[i+1])
        count = mask.sum()
        if count > 0:
            avg_pred = pred[mask].mean()
            avg_err = err[mask].mean()
            reliability[i] = [avg_pred, avg_err, count]
            ece += count * abs(avg_pred - avg_err)
    
    ece /= len(pred)
    return float(ece), reliability


def compute_coverage(
    samples: torch.Tensor,
    ground_truth: torch.Tensor,
    alpha: float = 0.9
) -> float:
    """
    Compute interval coverage.
    
    For calibrated uncertainty, α CI should contain truth α% of time.
    
    Args:
        samples: (n_samples, ...) posterior samples
        ground_truth: (...) true values
        alpha: Target coverage (0.9 = 90% CI)
    
    Returns:
        coverage: Fraction where truth is in interval
    """
    lower_q = (1 - alpha) / 2
    upper_q = 1 - lower_q
    
    lower = torch.quantile(samples, lower_q, dim=0)
    upper = torch.quantile(samples, upper_q, dim=0)
    
    in_interval = (ground_truth >= lower) & (ground_truth <= upper)
    return in_interval.float().mean().item()


def compute_frequency_resolved_calibration(
    samples: torch.Tensor,
    ground_truth: torch.Tensor,
    n_bands: int = 8
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    ⭐ NOVEL: Compute calibration metrics per frequency band.
    
    This is the KEY CONTRIBUTION of CURE.
    
    Args:
        samples: (n_samples, C, H, W) posterior samples
        ground_truth: (C, H, W) true image
        n_bands: Number of frequency bands
    
    Returns:
        ece_per_band: (n_bands,)
        coverage_per_band: (n_bands,)
        band_centers: (n_bands,)
    """
    # Convert to grayscale for analysis
    if samples.dim() == 4:
        samples_gray = samples.mean(dim=1)  # (n_samples, H, W)
    else:
        samples_gray = samples
    
    if ground_truth.dim() == 3:
        truth_gray = ground_truth.mean(dim=0)  # (H, W)
    else:
        truth_gray = ground_truth
    
    H, W = truth_gray.shape
    bands = FrequencyBands((H, W), n_bands)
    
    # To frequency domain
    samples_freq = torch.fft.fft2(samples_gray)
    truth_freq = torch.fft.fft2(truth_gray)
    
    mean_freq = samples_freq.mean(dim=0)
    var_freq = samples_freq.var(dim=0).real
    error_freq = torch.abs(mean_freq - truth_freq) ** 2
    
    ece_per_band = np.zeros(n_bands)
    coverage_per_band = np.zeros(n_bands)
    
    for i in range(n_bands):
        mask = bands.masks[i]
        if mask.sum() == 0:
            continue
        
        pred_var = var_freq[mask].real
        true_err = error_freq[mask]
        
        ece_per_band[i], _ = compute_ece(pred_var, true_err)
        
        # Coverage
        samples_band = torch.abs(samples_freq[:, mask])
        truth_band = torch.abs(truth_freq[mask])
        coverage_per_band[i] = compute_coverage(samples_band, truth_band, alpha=0.9)
    
    return ece_per_band, coverage_per_band, bands.band_centers


def compute_calibration_vs_H(
    samples: torch.Tensor,
    ground_truth: torch.Tensor,
    H_magnitude: torch.Tensor,
    n_bins: int = 10
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    ⭐ NOVEL: Analyze calibration as function of |H(f)|.
    
    Tests hypothesis: miscalibration worst where |H(f)| ≈ 0.
    
    Args:
        samples: (n_samples, C, H, W)
        ground_truth: (C, H, W)
        H_magnitude: (H, W) blur frequency response
        n_bins: Number of |H| bins
    
    Returns:
        h_centers: (n_bins,) bin centers
        ece_per_bin: (n_bins,)
        coverage_per_bin: (n_bins,)
    """
    samples_gray = samples.mean(dim=1) if samples.dim() == 4 else samples
    truth_gray = ground_truth.mean(dim=0) if ground_truth.dim() == 3 else ground_truth
    
    samples_freq = torch.fft.fft2(samples_gray)
    truth_freq = torch.fft.fft2(truth_gray)
    
    mean_freq = samples_freq.mean(dim=0)
    var_freq = samples_freq.var(dim=0).real
    error_freq = torch.abs(mean_freq - truth_freq) ** 2
    
    H_flat = H_magnitude.flatten().numpy()
    var_flat = var_freq.flatten().numpy()
    err_flat = error_freq.flatten().numpy()
    
    bin_edges = np.percentile(H_flat, np.linspace(0, 100, n_bins + 1))
    
    h_centers = np.zeros(n_bins)
    ece_per_bin = np.zeros(n_bins)
    coverage_per_bin = np.zeros(n_bins)
    
    for i in range(n_bins):
        mask = (H_flat >= bin_edges[i]) & (H_flat < bin_edges[i+1])
        if mask.sum() == 0:
            continue
        
        h_centers[i] = H_flat[mask].mean()
        
        ece_per_bin[i], _ = compute_ece(
            torch.tensor(var_flat[mask]),
            torch.tensor(err_flat[mask])
        )
        
        samples_bin = torch.abs(samples_freq.flatten(1)[:, mask])
        truth_bin = torch.abs(truth_freq.flatten()[mask])
        coverage_per_bin[i] = compute_coverage(samples_bin, truth_bin, 0.9)
    
    return h_centers, ece_per_bin, coverage_per_bin


def plot_calibration_results(
    wiener_ece: np.ndarray,
    dps_ece: np.ndarray,
    wiener_cov: np.ndarray,
    dps_cov: np.ndarray,
    band_centers: np.ndarray,
    save_path: str = None
):
    """Plot frequency-resolved calibration comparison."""
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    n = len(band_centers)
    x = np.arange(n)
    w = 0.35
    
    axes[0].bar(x - w/2, wiener_ece, w, label='Wiener', color='steelblue')
    axes[0].bar(x + w/2, dps_ece, w, label='DPS', color='coral')
    axes[0].set_xlabel('Frequency Band')
    axes[0].set_ylabel('ECE (↓ better)')
    axes[0].set_title('Calibration Error per Frequency')
    axes[0].legend()
    axes[0].set_xticks(x)
    
    axes[1].bar(x - w/2, wiener_cov, w, label='Wiener', color='steelblue')
    axes[1].bar(x + w/2, dps_cov, w, label='DPS', color='coral')
    axes[1].axhline(0.9, color='black', linestyle='--', label='Target')
    axes[1].set_xlabel('Frequency Band')
    axes[1].set_ylabel('90% Coverage')
    axes[1].set_title('Coverage per Frequency')
    axes[1].legend()
    axes[1].set_xticks(x)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()
