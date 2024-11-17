import torch
import torch.nn as nn
import numpy as np
import schedulefree

from safetensors.torch import load_model, save_model


def normalize(x: torch.Tensor, dim=None, eps=1e-4) -> torch.Tensor:
    if dim is None:
        dim = list(range(1, x.ndim))
    norm = torch.linalg.vector_norm(
        x, dim=dim, keepdim=True, dtype=torch.float32)  # type: torch.Tensor
    norm = torch.add(eps, norm, alpha=np.sqrt(norm.numel() / x.numel()))
    norm_detached = norm.detach().to(x.dtype)  # Detach and cast to x's dtype
    return x / norm_detached
    # return x / norm.to(x.dtype)


class FourierFeatureExtractor(nn.Module):
    def __init__(self, num_channels, bandwidth=1):
        super().__init__()
        self.register_buffer('freqs', 2 * torch.pi *
                             torch.randn(num_channels) * bandwidth)
        self.register_buffer('phases', 2 * torch.pi * torch.rand(num_channels))
        self.sqrt_two = torch.sqrt(torch.tensor(2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x.to(torch.float32)
        y = y.ger(self.freqs.to(torch.float32))
        y = y + self.phases.to(torch.float32)  # type: torch.Tensor
        y = y.cos() * self.sqrt_two
        return y.to(x.dtype)


class NormalizedLinearLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel):
        super().__init__()
        self.out_channels = out_channels
        self.weight = nn.Parameter(torch.randn(
            out_channels, in_channels, *kernel))

    def forward(self, x: torch.Tensor, gain=1) -> torch.Tensor:
        w = self.weight.to(torch.float32)
        if self.training:
            with torch.no_grad():
                self.weight.copy_(normalize(w))  # forced weight normalization
        w = normalize(w)  # traditional weight normalization
        # type: torch.Tensor # magnitude-preserving scaling
        w = w * (gain / np.sqrt(w[0].numel()))
        w = w.to(x.dtype)
        if w.ndim == 2:
            return x @ w.t()
        assert w.ndim == 4
        return nn.functional.conv2d(x, w, padding=(w.shape[-1]//2,))


class AdaptiveLossWeightMLP(nn.Module):
    def __init__(
        self,
        noise_scheduler,
        logvar_channels=128,
        device='cuda',
        **kwargs
    ):
        super().__init__()
        self.a_bar_mean = noise_scheduler.alphas_cumprod.mean()
        self.a_bar_std = noise_scheduler.alphas_cumprod.std()
        self.logvar_fourier = FourierFeatureExtractor(logvar_channels)
        # kernel = []? (not in code given, added matching edm2)
        self.logvar_linear = NormalizedLinearLayer(
            logvar_channels, 1, kernel=[])
        self.noise_scheduler = noise_scheduler

    def _forward(self, timesteps: torch.Tensor):
        return self.compute_variance(timesteps)

    def forward(self, loss: torch.Tensor, timesteps):
        adaptive_loss_weights = self.compute_variance(timesteps)
        # type: torch.Tensor
        loss_scaled = loss / torch.exp(adaptive_loss_weights)
        # loss = loss_scaled + adaptive_loss_weights  # type: torch.Tensor

        # stdev, mean = torch.std_mean(loss)
        # print(f"{mean=:.4f} {stdev=:.4f}")

        return loss_scaled

    def compute_variance(self, timesteps: torch.Tensor):
        return self._compute_ddpm_variance(timesteps)

    def _compute_ddpm_variance(self, timesteps: torch.Tensor):
        a_bar = self.noise_scheduler.alphas_cumprod[timesteps]
        c_noise = a_bar.sub(self.a_bar_mean).div_(self.a_bar_std)
        return self.logvar_linear(self.logvar_fourier(c_noise)).squeeze()


class EDM2WeightingWrapper:
    def __init__(self,
                 noise_scheduler,
                 optimizer=torch.optim.AdamW,
                 lr=5e-3, 
                 optimizer_args={'weight_decay': 0},
                 logvar_channels=128,
                 device="cuda",
                 ):
        """
        Initialize EDM2Loss with Fourier features for training with dynamic loss weighting.

        :param optimizer: Optimizer class to use.
        :param lr: Learning rate for the optimizer.
        :param optimizer_args: Additional arguments for the optimizer.
        :param device: Device to run computations on.
        :param logvar_channels: Fourier decomposition complexity.
        """
        self.device = device
        self.model = AdaptiveLossWeightMLP(
            noise_scheduler=noise_scheduler,
            logvar_channels=logvar_channels,
            device=device
        ).to(device)
        self.model.train(mode=True)  # Ensure the model is in training mode
        self.optimizer = optimizer(
            self.model.parameters(), lr=lr, **optimizer_args)
        
        if 'schedulefree' in self.optimizer.__class__.__name__.lower():
            self.optimizer.train()

    def __call__(self, loss, timesteps):
        """
        Compute the weighted loss and backpropagate it through the loss_module.

        :param timesteps: Tensor of timesteps (shape: [batch_size]).
        :param loss: Tensor of individual losses (shape: [batch_size]).
        :return: Scalar tensor representing the total weighted loss.
        """
        timesteps = timesteps.to(self.device)
        loss = loss.to(self.device)

        # Forward pass through the loss_module
        weighted_losses = self.model(loss, timesteps)
        weighted_loss = weighted_losses.mean()

        # Backward pass for loss_module
        # Only compute gradients for self.model, don't touch anything else
        weighted_loss.backward(
            retain_graph=True, inputs=list(self.model.parameters()))

        self.optimizer.step()
        self.optimizer.zero_grad()

        return weighted_losses
    
    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)

    def state_dict(self):
        return {
            key: value for key, value in self.__dict__.items()
        }

    def save_model(self, path):
        save_model(self.model, path)

    def load_model(self, path):
        load_model(self.model, path)
