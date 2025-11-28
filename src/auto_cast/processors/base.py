from abc import ABC, abstractmethod
from typing import Any

import lightning as L
import torch
from torch import nn

from auto_cast.preprocessor import Preprocessor
from auto_cast.types import Batch, RolloutOutput, Tensor


class Processor(L.LightningModule):
    """Processor Base Class."""

    teacher_forcing_ratio: float
    stride: int
    max_rollout_steps: int
    preprocessor: Preprocessor
    loss_func: nn.Module

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Forward pass through the Processor."""
        msg = "To implement."
        raise NotImplementedError(msg)

    def training_step(self, batch: Batch, batch_idx: int) -> Tensor:  # noqa: ARG002
        x = self.preprocessor(batch)
        output = self(x)
        loss = self.loss_func(output, batch["output_fields"])
        return loss  # noqa: RET504

    def configure_optmizers(self):
        pass

    def rollout(self, batch: Batch) -> RolloutOutput:
        """Rollout over multiple time steps."""
        pred_outs = []
        gt_outs = []
        for _time_step in range(0, self.max_rollout_steps, self.stride):
            x = self.preprocessor(batch)
            pred_outs.append(self(x))
            gt_outs.append(batch["output_fields"])  # This assumes we have output fields
        return torch.stack(pred_outs), torch.stack(gt_outs)


class DiscreteProcessor(Processor, ABC):
    """DiscreteProcessor."""

    @abstractmethod
    def map(self, x: Batch) -> Tensor:
        ...
        # Map input window of states/times to output window

    def rollout(self, batch: Batch) -> RolloutOutput:
        ...

        # Use self.map to generate trajectory


class FlowBasedGenerativeProcessor(DiscreteProcessor):
    """Flow-based generative processor."""

    def map(self, x: Batch) -> Tensor:
        ...
        # Sample generative model    def loss(self, ...):...
        # Flow matc
