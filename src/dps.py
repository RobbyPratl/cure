"""
Diffusion Posterior Sampling (DPS).

Based on Chung et al., "Diffusion Posterior Sampling for General 
Noisy Inverse Problems", ICLR 2023.
"""

import torch
import torch.nn as nn
from typing import Callable, Tuple, Optional
from tqdm import tqdm


class DPSSampler:
    """
    Diffusion Posterior Sampling for image restoration.
    
    Samples from p(x|y) by guiding pretrained diffusion with likelihood.
    """
    
    def __init__(self, model, scheduler, device: str = 'cuda'):
        """
        Args:
            model: Pretrained UNet (from diffusers)
            scheduler: DDPM scheduler
            device: 'cuda' or 'cpu'
        """
        self.model = model.to(device)
        self.scheduler = scheduler
        self.device = device
        self.model.eval()
    
    @torch.no_grad()
    def _predict_x0(self, xt: torch.Tensor, t: int) -> torch.Tensor:
        """Predict clean x0 from noisy xt."""
        t_tensor = torch.tensor([t], device=self.device)
        noise_pred = self.model(xt, t_tensor).sample
        
        alpha_prod = self.scheduler.alphas_cumprod[t]
        x0 = (xt - torch.sqrt(1 - alpha_prod) * noise_pred) / torch.sqrt(alpha_prod)
        return x0.clamp(-1, 1)
    
    def _likelihood_gradient(
        self,
        x0: torch.Tensor,
        y: torch.Tensor,
        forward_op: Callable,
        noise_sigma: float
    ) -> torch.Tensor:
        """Compute ∇_x0 log p(y|x0)."""
        x0 = x0.detach().requires_grad_(True)
        Hx0 = forward_op(x0)
        residual = y - Hx0
        log_lik = -0.5 * (residual ** 2).sum() / (noise_sigma ** 2)
        grad = torch.autograd.grad(log_lik, x0)[0]
        return grad
    
    def sample(
        self,
        y: torch.Tensor,
        forward_op: Callable,
        noise_sigma: float,
        guidance_scale: float = 1.0,
        num_steps: int = 1000,
        show_progress: bool = True
    ) -> torch.Tensor:
        """
        Generate one posterior sample.
        
        Args:
            y: (C, H, W) degraded observation
            forward_op: Degradation operator H(x)
            noise_sigma: Observation noise level
            guidance_scale: Likelihood guidance strength
            num_steps: Diffusion steps
        
        Returns:
            sample: (C, H, W) posterior sample
        """
        shape = (1, *y.shape)
        xt = torch.randn(shape, device=self.device)
        
        self.scheduler.set_timesteps(num_steps)
        timesteps = self.scheduler.timesteps
        
        iterator = tqdm(timesteps, desc='DPS') if show_progress else timesteps
        
        for t in iterator:
            x0_pred = self._predict_x0(xt, t)
            
            grad = self._likelihood_gradient(
                x0_pred.squeeze(0), y, forward_op, noise_sigma
            )
            
            noise_pred = self.model(xt, torch.tensor([t], device=self.device)).sample
            xt = self.scheduler.step(noise_pred, t, xt).prev_sample
            
            alpha_prod = self.scheduler.alphas_cumprod[t]
            xt = xt + guidance_scale * torch.sqrt(alpha_prod) * grad.unsqueeze(0)
        
        return xt.squeeze(0).clamp(0, 1)
    
    def sample_multiple(
        self,
        y: torch.Tensor,
        forward_op: Callable,
        noise_sigma: float,
        n_samples: int = 50,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate multiple posterior samples.
        
        Returns:
            samples: (n_samples, C, H, W)
            mean: (C, H, W)
            variance: (C, H, W)
        """
        samples = []
        for i in tqdm(range(n_samples), desc='Sampling'):
            s = self.sample(y, forward_op, noise_sigma, show_progress=False, **kwargs)
            samples.append(s.cpu())
        
        samples = torch.stack(samples)
        return samples, samples.mean(0), samples.var(0)


def create_blur_forward_op(kernel: torch.Tensor, device: str = 'cuda') -> Callable:
    """Create forward operator for blur degradation."""
    kernel = kernel.to(device)
    
    def forward_op(x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(0)
        B, C, H, W = x.shape
        kH, kW = kernel.shape
        
        padded = torch.zeros(H, W, device=device)
        sh, sw = (H - kH) // 2, (W - kW) // 2
        padded[sh:sh+kH, sw:sw+kW] = kernel
        padded = torch.fft.ifftshift(padded)
        
        K = torch.fft.fft2(padded)
        X = torch.fft.fft2(x)
        out = torch.fft.ifft2(X * K).real
        
        return out.squeeze(0) if B == 1 else out
    
    return forward_op


def load_pretrained_diffusion(model_name: str = "google/ddpm-celebahq-256"):
    """Load pretrained diffusion model from HuggingFace."""
    from diffusers import DDPMPipeline
    
    pipe = DDPMPipeline.from_pretrained(model_name)
    return pipe.unet, pipe.scheduler
