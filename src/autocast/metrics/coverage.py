import torch
from torch.nn import ModuleList
from torchmetrics import Metric

from autocast.metrics.ensemble import BTSCMMetric
from autocast.types import Tensor, TensorBTC, TensorBTSC, TensorBTSCM


class Coverage(BTSCMMetric):
    """
    Coverage probability for a fixed coverage level.

    Calculates the proportion of true values that fall within the symmetric
    prediction interval defined by the coverage level.
    """

    name: str = "coverage"

    def __init__(self, coverage_level: float = 0.95):
        """Initialize Coverage metric.

        Args:
            coverage_level: nominal coverage probability (e.g. 0.95 for 95% interval).
                Must be between 0 and 1.
        """
        super().__init__()
        if not (0 < coverage_level < 1):
            raise ValueError(f"coverage_level must be in (0, 1), got {coverage_level}")
        self.coverage_level = coverage_level

    def _score(self, y_pred: TensorBTSCM, y_true: TensorBTSC) -> TensorBTC:
        """
        Compute coverage reduced over spatial dims.

        Args:
            y_pred: (B, T, S, C, M)
            y_true: (B, T, S, C)

        Returns
        -------
            coverage: (B, T, C)
        """
        # Calculate quantiles of the ensemble distribution
        # e.g. coverage_level=0.95 -> 0.025 and 0.975 quantiles
        q_low = 0.5 - self.coverage_level / 2
        q_high = 0.5 + self.coverage_level / 2

        # Calculate quantiles
        q_tensor = torch.tensor(
            [q_low, q_high], device=y_pred.device, dtype=y_pred.dtype
        )
        quantiles = torch.quantile(y_pred, q_tensor, dim=-1)
        # quantiles shape: (2, B, T, S, C)

        lower_q = quantiles[0]
        upper_q = quantiles[1]

        # Calculate coverage (1 if inside, 0 otherwise)
        is_covered = ((y_true >= lower_q) & (y_true <= upper_q)).float()

        # Reduce over spatial dimensions: (B, T, S, C) -> (B, T, C)
        n_spatial_dims = self._infer_n_spatial_dims(is_covered)
        spatial_dims = tuple(range(2, 2 + n_spatial_dims))
        coverage_reduced = is_covered.mean(dim=spatial_dims)

        return coverage_reduced


class MultiCoverage(Metric):
    """
    Computes coverage for multiple coverage levels at once.

    This is a wrapper around multiple Coverage metrics. It inherits from Metric
    to integrate with PyTorch Lightning and TorchMetrics.
    """

    def __init__(self, coverage_levels: list[float] | None = None):
        super().__init__()
        if coverage_levels is None:
            coverage_levels = [
                round(x, 2) for x in torch.linspace(0.05, 0.95, steps=19).tolist()
            ]

        self.coverage_levels = coverage_levels
        # Create a list of Coverage metrics
        self.metrics = ModuleList(
            [Coverage(coverage_level=cl) for cl in coverage_levels]
        )

    def update(self, y_pred, y_true):
        for metric in self.metrics:
            assert isinstance(metric, Coverage)
            metric.update(y_pred, y_true)

    def compute(self) -> Tensor:
        """Compute the Average Calibration Error."""
        errors = []
        for cl, metric in zip(self.coverage_levels, self.metrics, strict=True):
            assert isinstance(metric, Coverage)
            # Calibration error: |observed - expected|
            errors.append(torch.abs(metric.compute() - cl))

        return torch.stack(errors).mean()

    def compute_detailed(self) -> dict[str, Tensor]:
        """Return a dict of results, keys formatted as 'coverage_{coverage_level}'."""
        results = {}
        for cl, metric in zip(self.coverage_levels, self.metrics, strict=True):
            assert isinstance(metric, Coverage)
            results[f"coverage_{cl}"] = metric.compute()
        return results

    # TODO: consider re-adding plot method for directly logging to wandb later
    # def plot(self) -> Any:
    #     try:
    #         import matplotlib.pyplot as plt

    #         import wandb
    #     except ImportError:
    #         print("MultiCoverage.plot: wandb or matplotlib import failed")
    #         return None

    #     # Gather computed values from sub-metrics
    #     results = []
    #     for metric in self.metrics:
    #         assert isinstance(metric, Coverage)
    #         val = metric.compute()
    #         if val.numel() == 1:
    #             results.append(val.item())
    #         else:
    #             results.append(val.mean().item())

    #     # Create matplotlib figure
    #     step = wandb.run.step
    #     fig, ax = plt.subplots(figsize=(8, 6))

    #     # Plot observed coverage
    #     ax.plot(
    #         self.coverage_levels, results, marker="o", label="Observed", linewidth=2
    #     )

    #     # Plot ideal coverage (y=x line)
    #     ax.plot(
    #         self.coverage_levels,
    #         self.coverage_levels,
    #         linestyle="--",
    #         label="Ideal",
    #         linewidth=2,
    #     )

    #     ax.set_xlabel("Expected Coverage")
    #     ax.set_ylabel("Observed Coverage")
    #     ax.set_title(f"Coverage (step={step})")
    #     ax.legend()
    #     ax.grid(True, alpha=0.3)
    #     ax.set_xlim(0, 1)
    #     ax.set_ylim(0, 1)

    #     print(f"MultiCoverage.plot: Created plot with step={step}")

    #     # Convert to wandb Image
    #     plot = wandb.Image(fig)
    #     plt.close(fig)

    #     return plot

    def reset(self):
        # Reset all sub-metrics
        for metric in self.metrics:
            assert isinstance(metric, Coverage)
            metric.reset()
