import torch
from auto_cast.types import Batch, EncodedBatch, RolloutOutput, Tensor
from auto_cast.processors.base import Processor


class DiffusionProcessor(Processor):
    """Diffusion Processor."""

    def __init__(self, denoiser_nn, loss, **kwargs):
        """
        denoiser_nn: The neural network used for denoising.
        loss: The loss function.
        """
        super().__init__()
        self.denoiser_nn = denoiser_nn
        self.loss_func = loss

    def map(self, x: Tensor) -> Tensor:
        """Map input window of states/times to output window using denoiser."""
        return self.denoiser_nn(x)
