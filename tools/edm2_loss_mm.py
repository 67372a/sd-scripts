import torch
import torch.nn as nn
import numpy as np
from diffusers import DDPMScheduler

def normalize(x: torch.Tensor, dim=None, eps=1e-4) -> torch.Tensor:
    if dim is None:
        dim = list(range(1, x.ndim))
    norm = torch.linalg.vector_norm(x, dim=dim, keepdim=True, dtype=torch.float32) # type: torch.Tensor
    norm = torch.add(eps, norm, alpha=np.sqrt(norm.numel() / x.numel()))
    return x / norm.to(x.dtype)

class FourierFeatureExtractor(torch.nn.Module):
    def __init__(self, num_channels, bandwidth=1):
        super().__init__()
        self.register_buffer('freqs', 2 * np.pi * torch.randn(num_channels) * bandwidth)
        self.register_buffer('phases', 2 * np.pi * torch.rand(num_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x.to(torch.float32)
        y = y.ger(self.freqs.to(torch.float32))
        y = y + self.phases.to(torch.float32) # type: torch.Tensor
        y = y.cos() * np.sqrt(2)
        return y.to(x.dtype)

class NormalizedLinearLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel):
        super().__init__()
        self.out_channels = out_channels
        self.weight = torch.nn.Parameter(torch.randn(out_channels, in_channels, *kernel))

    def forward(self, x: torch.Tensor, gain=1) -> torch.Tensor:
        w = self.weight.to(torch.float32)
        if self.training:
            with torch.no_grad():
                self.weight.copy_(normalize(w)) # forced weight normalization
        w = normalize(w) # traditional weight normalization
        w = w * (gain / np.sqrt(w[0].numel())) # type: torch.Tensor # magnitude-preserving scaling
        w = w.to(x.dtype)
        if w.ndim == 2:
            return x @ w.t()
        assert w.ndim == 4
        return torch.nn.functional.conv2d(x, w, padding=(w.shape[-1]//2,))
    
class AdaptiveLossWeightMLP(nn.Module):
    def __init__(
            self,
            noise_scheduler: DDPMScheduler,
            logvar_channels=128,
            lambda_weights: torch.Tensor = None,
            device='cuda'
        ):
        super().__init__()
        self.alphas_cumprod = noise_scheduler.alphas_cumprod.to(device=device)
        #self.a_bar_mean = noise_scheduler.alphas_cumprod.mean()
        #self.a_bar_std = noise_scheduler.alphas_cumprod.std()
        self.a_bar_mean = self.alphas_cumprod.mean()
        self.a_bar_std = self.alphas_cumprod.std()
        self.logvar_fourier = FourierFeatureExtractor(logvar_channels)
        self.logvar_linear = NormalizedLinearLayer(logvar_channels, 1, kernel=[]) # kernel = []? (not in code given, added matching edm2)
        self.lambda_weights = lambda_weights.to(device=device) if lambda_weights is not None else torch.ones(1000, device=device)
        self.noise_scheduler = noise_scheduler

    def _forward(self, timesteps: torch.Tensor):
        #a_bar = self.noise_scheduler.alphas_cumprod[timesteps]
        a_bar = self.alphas_cumprod[timesteps]
        c_noise = a_bar.sub(self.a_bar_mean).div_(self.a_bar_std)
        return self.logvar_linear(self.logvar_fourier(c_noise)).squeeze()

    def forward(self, loss: torch.Tensor, timesteps):
        adaptive_loss_weights = self._forward(timesteps)
        loss_scaled = loss * (self.lambda_weights[timesteps] / torch.exp(adaptive_loss_weights)) # type: torch.Tensor
        loss = loss_scaled + adaptive_loss_weights # type: torch.Tensor

        return loss, loss_scaled
    
def create_weight_MLP(noise_scheduler, 
                      logvar_channels=128, 
                      lambda_weights=None, 
                      optimizer=torch.optim.AdamW, 
                      lr=2e-2,
                      optimizer_args={'weight_decay': 0, 'betas': (0.9,0.99)},
                      device='cuda'):
    print("creating weight MLP")
    lossweightMLP = AdaptiveLossWeightMLP(noise_scheduler, logvar_channels, lambda_weights, device)
    MLP_optim = optimizer(lossweightMLP.parameters(), lr=lr, **optimizer_args)
    return lossweightMLP, MLP_optim