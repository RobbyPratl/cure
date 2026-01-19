"""
CURE: Calibrated Uncertainty in Restoration via Spectral Estimation.

Post-hoc recalibration using Wiener filter uncertainty as reference.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple


class SpectralRecalibrator(nn.Module):
    """
    MLP that recalibrates diffusion variance.
    
    Input: (σ²_diff, |H(f)|, σ_noise, freq_mag)
    Output: σ²_calibrated
    Target: Wiener variance (provably correct)
    """
    
    def __init__(self, input_dim: int = 4, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Softplus()  # Ensure positive
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def prepare_recalibration_data(
    dps_variance: torch.Tensor,
    wiener_variance: torch.Tensor,
    H_magnitude: torch.Tensor,
    noise_sigma: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Prepare training data for recalibrator.
    
    Args:
        dps_variance: (H, W) DPS variance in frequency domain
        wiener_variance: (H, W) Wiener variance (target)
        H_magnitude: (H, W) blur frequency response
        noise_sigma: Noise level
    
    Returns:
        X: (N, 4) features
        y: (N, 1) targets
    """
    H, W = dps_variance.shape
    
    # Frequency magnitude
    fy = torch.fft.fftfreq(H)[:, None]
    fx = torch.fft.fftfreq(W)[None, :]
    freq_mag = torch.sqrt(fx**2 + fy**2)
    
    X = torch.stack([
        dps_variance.flatten(),
        H_magnitude.flatten(),
        torch.full((H*W,), noise_sigma),
        freq_mag.flatten()
    ], dim=1)
    
    y = wiener_variance.flatten().unsqueeze(1)
    
    return X, y


def train_recalibrator(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    n_epochs: int = 100,
    lr: float = 1e-3,
    device: str = 'cuda'
) -> SpectralRecalibrator:
    """Train recalibration model."""
    model = SpectralRecalibrator().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    X_train, y_train = X_train.to(device), y_train.to(device)
    X_val, y_val = X_val.to(device), y_val.to(device)
    
    best_val = float('inf')
    best_state = None
    
    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()
        pred = model(X_train)
        loss = criterion(pred, y_train)
        loss.backward()
        optimizer.step()
        
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val)
            val_loss = criterion(val_pred, y_val).item()
        
        if val_loss < best_val:
            best_val = val_loss
            best_state = model.state_dict().copy()
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}: train={loss.item():.6f}, val={val_loss:.6f}")
    
    model.load_state_dict(best_state)
    return model
