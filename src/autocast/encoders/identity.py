from autocast.encoders.base import Encoder
from autocast.types.batch import Batch
from autocast.types.types import TensorBNC


class IdentityEncoder(Encoder):
    """Permute and concatenate Encoder."""

    def __init__(self) -> None:
        super().__init__()

    def encode(self, batch: Batch) -> TensorBNC:
        return batch.input_fields


class IdentityEncoderWithCond(IdentityEncoder):
    """Permute and concatenate Encoder."""
