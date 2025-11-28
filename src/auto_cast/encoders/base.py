from typing import Any

from torch import nn


class Encoder(nn.Module):
    """Base encoder."""

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Forward Pass through the Encoder."""
        msg = "To implement."
        raise NotImplementedError(msg)
