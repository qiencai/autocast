from typing import Generic, TypeVar

import torch
from torchmetrics import Metric

from autocast.types import TensorBTC
from autocast.types.types import ArrayLike, Tensor

TPred = TypeVar("TPred", bound=Tensor)
TTrue = TypeVar("TTrue", bound=Tensor)


class BaseMetric(Metric, Generic[TPred, TTrue]):
    """Shared template for spatial metrics that reduce over spatial axes."""

    def __init__(
        self,
        reduce_all: bool = True,
        dist_sync_on_step: bool = False,
    ) -> None:
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.reduce_all = reduce_all

        # States shared by all derived metrics
        self.add_state("sum_score", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_samples", default=torch.tensor(0), dist_reduce_fx="sum")

        # Internal flag to set shape of sum_score
        self._initialized = False

    def _check_input(self, y_pred: ArrayLike, y_true: ArrayLike) -> tuple[TPred, TTrue]:
        """Validate / convert inputs. Concrete subclasses must implement."""
        raise NotImplementedError

    def score(self, y_pred: TPred, y_true: TTrue) -> TensorBTC:
        """Compute spatially reduced score. Concrete subclasses must implement."""
        raise NotImplementedError

    def update(self, y_pred: ArrayLike, y_true: ArrayLike) -> None:
        """Update metric state with a batch of predictions and targets."""
        y_pred_t, y_true_t = self._check_input(y_pred, y_true)

        # (B, T, *S, C) -> (B, T, C)
        score_spatial = self.score(y_pred_t, y_true_t)

        if score_spatial.ndim != 3:
            raise ValueError(
                f"score must return shape (B, T, C), got {score_spatial.shape}"
            )

        batch_size = score_spatial.shape[0]

        # Sum over batch dimension: (B, T, C) -> (T, C)
        score_summed = torch.sum(score_spatial, dim=0)

        # Lazily set correct shape for sum_score on first batch
        if not self._initialized:
            self.sum_score = torch.zeros_like(score_summed)
            self._initialized = True

        self.sum_score += score_summed
        self.total_samples += batch_size

    def compute(self) -> torch.Tensor:
        """Compute final metric value."""
        if self.total_samples == 0:
            msg = "No samples were provided to the metric"
            raise RuntimeError(msg)

        score = self.sum_score / self.total_samples

        if self.reduce_all:
            # Average over time and channels
            return score.mean()

        return score

    def reset(self) -> None:
        """Reset metric state and initialization flag."""
        super().reset()
        self._initialized = False

    def _infer_n_spatial_dims(self, tensor: torch.Tensor) -> int:
        """Infer number of spatial dimensions from tensor shape."""
        return tensor.ndim - 3  # Subtract B, T, C
