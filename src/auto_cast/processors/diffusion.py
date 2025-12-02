import torch
import torch.nn as nn
from auto_cast.types import Batch, EncodedBatch, RolloutOutput, Tensor
from auto_cast.processors.base import Processor

from azula.noise import Schedule



class DiffusionProcessor(Processor):
    """Diffusion Processor."""

    def __init__(self, denoiser_nn, loss, schedule: Schedule, **kwargs):
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
