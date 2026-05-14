from collections.abc import Sequence
from typing import Any

import math

import lightning as L
import torch
from omegaconf import DictConfig
from the_well.data.normalization import ZScoreNormalization
from torch import nn
from torchmetrics import Metric, MetricCollection

from autocast.metrics.utils import MetricsMixin
from autocast.models.denorm_mixin import DenormMixin
from autocast.models.encoder_decoder import EncoderDecoder
from autocast.models.optimizer_mixin import OptimizerMixin
from autocast.nn.noise.noise_injector import NoiseInjector
from autocast.processors.base import Processor
from autocast.processors.rollout import RolloutMixin
from autocast.types import Batch, Tensor, TensorBTSC
from autocast.types.types import TensorBTSCM


class EncoderProcessorDecoder(
    DenormMixin, OptimizerMixin, RolloutMixin[Batch], L.LightningModule, MetricsMixin
):
    """Encoder-Processor-Decoder Model."""

    encoder_decoder: EncoderDecoder
    processor: Processor
    train_metrics: MetricCollection | None
    val_metrics: MetricCollection | None
    test_metrics: MetricCollection | None

    def __init__(
        self,
        encoder_decoder: EncoderDecoder,
        processor: Processor,
        optimizer_config: DictConfig | dict[str, Any] | None = None,
        stride: int = 1,
        rollout_stride: int | None = None,
        teacher_forcing_ratio: float = 0.5,
        max_rollout_steps: int = 10,
        train_in_latent_space: bool = False,
        freeze_encoder_decoder: bool = False,
        loss_func: nn.Module | None = None,
        autoregressive_train_steps: int = 0,
        predict_incremental_diff: bool = False,
        train_metrics: Sequence[Metric] | None = [],
        val_metrics: Sequence[Metric] | None = None,
        test_metrics: Sequence[Metric] | None = None,
        input_noise_injector: NoiseInjector | None = None,
        norm: ZScoreNormalization | None = None,
        apply_land_mask: bool = False,
        land_mask_channel: int = 0,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.encoder_decoder = encoder_decoder
        self.processor = processor
        self.optimizer_config = optimizer_config
        self.stride = stride
        self.rollout_stride = rollout_stride if rollout_stride is not None else stride
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.max_rollout_steps = max_rollout_steps
        self.train_in_latent_space = train_in_latent_space
        self.freeze_encoder_decoder = freeze_encoder_decoder
        self.input_noise_injector = input_noise_injector
        self.norm = norm
        self.apply_land_mask = apply_land_mask
        self.land_mask_channel = land_mask_channel

        if self.train_in_latent_space or self.freeze_encoder_decoder:
            self.encoder_decoder.freeze()
        self.loss_func = loss_func
        self.autoregressive_train_steps = autoregressive_train_steps
        self.predict_incremental_diff = predict_incremental_diff

        self.train_metrics = self._build_metrics(train_metrics, "train_")
        self.val_metrics = self._build_metrics(val_metrics, "val_")
        self.test_metrics = self._build_metrics(test_metrics, "test_")

        for key, value in kwargs.items():
            setattr(self, key, value)

    def _apply_input_noise(self, batch: Batch) -> Batch:
        """Apply input noise if self.input_noise_injector is set."""
        if self.input_noise_injector is not None:
            noisy_input = self.input_noise_injector(batch.input_fields)
            batch = Batch(
                input_fields=noisy_input,
                output_fields=batch.output_fields,
                constant_scalars=batch.constant_scalars,
                constant_fields=batch.constant_fields,
                boundary_conditions=batch.boundary_conditions,
                constant_doy_scalars=batch.constant_doy_scalars,
            )
        return batch

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        # PyTorch can serialise _metadata as a real key in the state_dict rather
        # than as an OrderedDict private attribute.  Strict load_state_dict then
        # rejects it as an unexpected key.  Remove it before Lightning loads.
        state_dict = checkpoint.get("state_dict", checkpoint)
        if isinstance(state_dict, dict):
            state_dict.pop("_metadata", None)

    def forward(self, batch: Batch) -> TensorBTSC | TensorBTSCM:
        batch = self._apply_input_noise(batch)
        encoded, global_cond = self.encoder_decoder.encoder.encode_with_cond(batch)
        mapped = self.processor.map(encoded, global_cond)
        decoded = self.encoder_decoder.decoder.decode(mapped)
        if self.predict_incremental_diff:
            # Model predicts increment; add last input frame as residual.
            # last_input: (B, 1, *spatial, C) — broadcasts over T_out steps.
            last_input = batch.input_fields[:, -1:, ...]
            decoded = decoded + last_input
        return decoded

    def loss(self, batch: Batch) -> tuple[Tensor, Tensor | None]:
        if self.autoregressive_train_steps > 0:
            if self.loss_func is None:
                msg = "loss_func must be provided for autoregressive multi-step loss."
                raise ValueError(msg)
            current_batch = self._clone_batch(batch)
            total_loss = torch.zeros(1, device=batch.input_fields.device)
            steps_computed = 0
            y_pred = None
            for _ in range(self.autoregressive_train_steps):
                y_pred = self(current_batch)
                true_slice, should_record = self._true_slice(current_batch, self.stride)
                if not should_record:
                    break
                total_loss = total_loss + self.loss_func(y_pred, true_slice)
                steps_computed += 1
                rand_val = torch.rand(1, device=y_pred.device).item()
                if true_slice.numel() > 0 and rand_val < self.teacher_forcing_ratio:
                    next_input = true_slice
                else:
                    next_input = y_pred  # no detach — gradients flow through rollout
                if next_input.shape[1] < self.stride:
                    break
                current_batch = self._advance_batch(current_batch, next_input, self.stride)
            loss = total_loss / max(steps_computed, 1)
            return loss, y_pred
        if self.train_in_latent_space:
            batch = self._apply_input_noise(batch)
            encoded_batch = self.encoder_decoder.encoder.encode_batch(batch)
            loss = self.processor.loss(encoded_batch)
            y_pred = None
        else:
            if self.loss_func is None:
                msg = "loss_func must be provided when training full EPD model."
                raise ValueError(msg)
            # Otherwise, train full EPD model
            y_pred = self(batch)
            y_true = batch.output_fields
            loss = self.loss_func(y_pred, y_true)
        return loss, y_pred

    def training_step(self, batch: Batch, batch_idx: int) -> Tensor:  # noqa: ARG002
        loss, y_pred = self.loss(batch)
        self.log(
            "train_loss",
            loss,
            prog_bar=True,
            sync_dist=True,
            batch_size=batch.input_fields.shape[0],
        )
        if self.train_metrics is not None:
            if y_pred is None:
                y_pred = self(batch)
            y_true = batch.output_fields
            self._update_and_log_metrics(
                self, self.train_metrics, y_pred, y_true, batch.input_fields.shape[0]
            )
        return loss

    def validation_step(self, batch: Batch, batch_idx: int) -> Tensor:  # noqa: ARG002
        loss, y_pred = self.loss(batch)
        self.log(
            "val_loss",
            loss,
            prog_bar=True,
            sync_dist=True,
            batch_size=batch.input_fields.shape[0],
        )
        if self.val_metrics is not None:
            if y_pred is None:
                y_pred = self(batch)
            y_true = batch.output_fields
            y_pred = self.denormalize_tensor(y_pred)
            y_true = self.denormalize_tensor(y_true)
            self._update_and_log_metrics(
                self, self.val_metrics, y_pred, y_true, batch.input_fields.shape[0]
            )
        return loss

    def test_step(self, batch: Batch, batch_idx: int) -> Tensor:  # noqa: ARG002
        loss, y_pred = self.loss(batch)
        self.log(
            "test_loss",
            loss,
            prog_bar=True,
            sync_dist=True,
            batch_size=batch.input_fields.shape[0],
        )
        if self.test_metrics is not None:
            if y_pred is None:
                y_pred = self(batch)
            y_true = batch.output_fields
            y_pred = self.denormalize_tensor(y_pred)
            y_true = self.denormalize_tensor(y_true)
            self._update_and_log_metrics(
                self, self.test_metrics, y_pred, y_true, batch.input_fields.shape[0]
            )
        return loss

    def _clone_batch(self, batch: Batch) -> Batch:
        return Batch(
            input_fields=batch.input_fields.clone(),
            output_fields=batch.output_fields.clone(),
            constant_scalars=(
                batch.constant_scalars.clone()
                if batch.constant_scalars is not None
                else None
            ),
            constant_fields=(
                batch.constant_fields.clone()
                if batch.constant_fields is not None
                else None
            ),
            boundary_conditions=(
                batch.boundary_conditions.clone()
                if batch.boundary_conditions is not None
                else None
            ),
            constant_doy_scalars=(
                batch.constant_doy_scalars.clone()
                if batch.constant_doy_scalars is not None
                else None
            ),
            forcing_fields_seq=(
                batch.forcing_fields_seq.clone()
                if batch.forcing_fields_seq is not None
                else None
            ),
            doy_seq=(
                batch.doy_seq.clone()
                if batch.doy_seq is not None
                else None
            ),
        )

    def _predict(self, batch: Batch) -> Tensor:
        output = self(batch)
        # To be fixed: note that this assumes landmask is channel 0 in constant_fields. If more constant fields are added or if the land mask is not binary, this logic will need to be updated to make sure the correct channel is used.
        if self.apply_land_mask and batch.constant_fields is not None:
            # Extract the land mask channel from constant_fields: (B, W, H, C_const)
            # Result: (B, W, H) -> (B, 1, W, H, 1) to broadcast over T and C_out.
            mask = batch.constant_fields[..., self.land_mask_channel].unsqueeze(1).unsqueeze(-1)
            output = output * mask
        return output

    def _true_slice(
        self,
        batch: Batch,
        stride: int,
    ) -> tuple[Tensor, bool]:
        if batch.output_fields.shape[1] >= stride:
            return batch.output_fields[:, :stride, ...], True
        return batch.output_fields, False

    def _advance_batch(self, batch: Batch, next_inputs: Tensor, stride: int) -> Batch:
        """Shift the input/output windows forward by `stride` using `next_inputs`.

        Note: stride parameter overrides self.stride to allow different strides
        for training vs evaluation.
        """
        # Get the original number of input time steps to maintain consistency
        n_steps_input = batch.input_fields.shape[1]

        # Concatenate remaining inputs with new predictions
        remaining_inputs = batch.input_fields[:, stride:, ...]
        new_predictions = next_inputs[:, :stride, ...]

        if remaining_inputs.shape[1] == 0:
            # No remaining inputs, use most recent n_steps_input from predictions
            combined = new_predictions[:, -n_steps_input:, ...]
        else:
            combined = torch.cat([remaining_inputs, new_predictions], dim=1)
            # Keep only the most recent n_steps_input time steps
            combined = combined[:, -n_steps_input:, ...]

        next_outputs = (
            batch.output_fields[:, stride:, ...]
            if batch.output_fields.shape[1] > stride
            else batch.output_fields[:, 0:0, ...]  # Empty tensor with correct shape
        )

        return Batch(
            input_fields=combined,
            output_fields=next_outputs,
            constant_scalars=batch.constant_scalars,
            constant_fields=batch.constant_fields,
            boundary_conditions=batch.boundary_conditions,
            constant_doy_scalars=self._advance_doy_scalars(batch.constant_doy_scalars, stride),
            forcing_fields_seq=(
                batch.forcing_fields_seq[:, stride:, ...]
                if batch.forcing_fields_seq is not None and batch.forcing_fields_seq.shape[1] > stride
                else None
            ),
            doy_seq=(
                batch.doy_seq[:, stride:, :]
                if batch.doy_seq is not None and batch.doy_seq.shape[1] > stride
                else None
            ),
        )

    @staticmethod
    def _advance_doy_scalars(
        constant_doy_scalars: torch.Tensor | None, stride: int
    ) -> torch.Tensor | None:
        """Advance cyclic date-of-year scalars by `stride` days.
        
        Assumes constant_doy_scalars are [sin(phase), cos(phase)] and updates them
        to reflect the new date after advancing by `stride` days.
        """
        if constant_doy_scalars is None:
            return None
        sin_val = constant_doy_scalars[..., 0]
        cos_val = constant_doy_scalars[..., 1]
        old_phase = torch.atan2(sin_val, cos_val)
        new_phase = old_phase + 2.0 * math.pi * stride / 365.25
        updated = constant_doy_scalars.clone()
        updated[..., 0] = torch.sin(new_phase)
        updated[..., 1] = torch.cos(new_phase)
        return updated
