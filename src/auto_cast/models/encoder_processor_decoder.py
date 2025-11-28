from typing import Any

import lightning as L

from auto_cast.models.encoder_decoder import EncoderDecoder
from auto_cast.preprocessor.base import Preprocessor
from auto_cast.processors.base import Processor
from auto_cast.types import Batch, Tensor


class EncoderProcessorDecoder(L.LightningModule):
    """Encoder-Processor-Decoder Model."""

    encoder_decoder: EncoderDecoder
    processor: Processor
    preprocessor: Preprocessor

    def __init__(self): ...

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.encoder_decoder.decoder(
            self.processor(self.encoder_decoder.encoder(*args, **kwargs))
        )

    def training_step(self, batch: Batch, batch_idx: int) -> Tensor:  # noqa: ARG002
        x = self.preprocessor(batch)
        output = self(x)
        loss = self.processor.loss_func(output, batch["output_fields"])
        return loss  # noqa: RET504

    def configure_optmizers(self): ...
