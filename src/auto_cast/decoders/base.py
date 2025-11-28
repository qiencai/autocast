from torch import nn

from auto_cast.types import Tensor


class Decoder(nn.Module):
    """Base Decoder."""

    # Q: Should decoder handle all these input types
    def forward(self, x: Tensor) -> Tensor: ...
