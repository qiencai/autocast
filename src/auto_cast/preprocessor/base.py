from torch import nn

from auto_cast.types import Batch, Tensor


class Preprocessor(nn.Module):
    """Base Preprocessor."""

    def forward(self, x: Batch) -> Tensor:
        """Forward Pass through the Preprocessor."""
        msg = "To implement."
        raise NotImplementedError(msg)
