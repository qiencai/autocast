import math

import torch
import torch.nn as nn

from auto_cast.processors.base import Processor
from auto_cast.types import Batch, EncodedBatch, RolloutOutput, Tensor


class NoiseSchedule(nn.Module):
    """Noise Schedule Module."""

    def forward(self, t: Tensor) -> tuple[Tensor, Tensor]:
        """Get alpha and sigma for given time steps t."""
        msg = "Subclasses should implement this method."
        raise NotImplementedError(msg)
    

class LogLinearSchedule(NoiseSchedule):
    """Log-Linear Noise Schedule.
    
    Implements a log-linear schedule for alpha and sigma.
    """

    def __init__(self, sigma_min: float = 0.002, sigma_max: float = 80.0):
        super().__init__()
        self.log_sigma_min = math.log(sigma_min)
        self.log_sigma_max = math.log(sigma_max)

    def forward(self, t: Tensor) -> tuple[Tensor, Tensor]:
        alpha = torch.ones_like(t)
        sigma = torch.exp(self.log_sigma_min * (1 - t) + self.log_sigma_max * t)
        return alpha, sigma


class DiffusionProcessor(Processor):
    """Diffusion Processor."""

    def __init__(self, denoiser_nn, loss, schedule: NoiseSchedule, **kwargs):
        """Initialize the DiffusionProcessor.

        denoiser_nn: The neural network used for denoising.
        loss: The loss function.
        schedule: Noise schedule from azula.noise (e.g., LogLinearSchedule).
                  Defines how signal (alpha) and noise (sigma) scale over time.
        """
        super().__init__()
        self.denoiser_nn = denoiser_nn
        self.schedule = schedule
        self.loss_func = loss

    def map(self, x: Tensor) -> Tensor:
        """Map input window of states/times to output window using denoiser."""
        return self.denoiser_nn(x)
    
    def forward(self, x: Tensor) -> Tensor:
        return self.map(x)




