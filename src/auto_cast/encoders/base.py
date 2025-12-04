from abc import ABC, abstractmethod

from torch import nn

from auto_cast.types import Batch, TensorBMultiL


class Encoder(nn.Module, ABC):
    """Base encoder."""

    encoder_model: nn.Module
    latent_dim: int

    def preprocess(self, batch: Batch) -> Batch:
        """Optionally transform a batch before encoding.

        Subclasses can override to implement pre-encoding steps that still
        return a fully-populated `Batch` instance. Default is identity.
        """
        return batch

    @abstractmethod
    def encode(self, batch: Batch) -> TensorBMultiL:
        """Encode the input tensor into the latent space.

        Parameters
        ----------
        x: Batch
            Input batch to be encoded.

        Returns
        -------
        TensorBMultiL
            Encoded tensor in the latent space with shape (B, *, C_latent).
        """

    def __call__(self, batch: Batch) -> TensorBMultiL:
        return self.encode(batch)
