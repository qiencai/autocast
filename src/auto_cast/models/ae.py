import torch
from torch import nn

from auto_cast.decoders import Decoder
from auto_cast.encoders import Encoder
from auto_cast.models.encoder_decoder import EncoderDecoder
from auto_cast.types import Batch, Tensor


class AELoss(nn.Module):
    """Autoencoder Loss Function."""

    def __init__(
        self, losses: list[nn.Module] | None = None, weights: list[float] | None = None
    ):
        super().__init__()
        losses = losses or [nn.MSELoss()]
        weights = weights or [1.0] * len(losses)
        self.losses = losses
        self.weights = weights

    def forward(self, model: EncoderDecoder, batch: Batch) -> Tensor:
        output = model(batch)
        total_loss = torch.tensor(0.0)
        for loss, weight in zip(self.losses, self.weights, strict=True):
            total_loss += loss(output, batch.output_fields) * weight
        return total_loss


class AE(EncoderDecoder):
    """Autoencoder Model."""

    encoder: Encoder
    decoder: Decoder

    def __init__(self, encoder: Encoder, decoder: Decoder, loss_func: AELoss):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.loss_func = loss_func

    def forward(self, batch: Batch) -> Tensor:
        return self.forward_with_latent(batch)[0]

    def forward_with_latent(self, batch: Batch) -> tuple[Tensor, Tensor]:
        encoded = self.encode(batch)
        decoded = self.decode(encoded)
        return decoded, encoded
