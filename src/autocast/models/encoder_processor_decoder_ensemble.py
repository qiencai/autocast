from einops import rearrange

from autocast.models.encoder_processor_decoder import EncoderProcessorDecoder
from autocast.types import Batch, TensorBTSCM


class EncoderProcessorDecoderEnsemble(EncoderProcessorDecoder):
    """Encoder-Processor-Decoder Ensemble Model."""

    def forward(self, batch: Batch) -> TensorBTSCM:
        """Forward pass through the ensemble model.

        The input batch is encoded, processed, and decoded, with the ensemble
        dimension handled appropriately.

        An ensemble metrics can be passed as the loss func in the same way as in
        the base EncoderProcessorDecoder model.

        TODO: Handling of the other metrics (e.g. MSE) needs to be considered.
        """
        encoded = self.encoder_decoder.encoder.encode(batch)
        mapped: TensorBTSCM = self.processor.map(encoded)
        B, *_, M = mapped.shape
        mapped = rearrange(mapped, "b t ... c m -> (b m) t ... c")
        decoded = self.encoder_decoder.decoder.decode(mapped)
        decoded = rearrange(decoded, "(b m) t ... c -> b t ... c m", b=B, m=M)
        return decoded
