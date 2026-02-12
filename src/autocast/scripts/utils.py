"""Utility functions for AutoCast scripts."""

import logging
import re
from pathlib import Path
from typing import Any, ClassVar

import pandas as pd
import yaml


def get_default_config_path() -> str:
    """Find the configs directory by searching upward for project root.

    Searches upward from this file for pyproject.toml (project root marker),
    then returns the path to the configs directory.

    Returns
    -------
    str
        Absolute path to the configs directory.

    Raises
    ------
    FileNotFoundError
        If project root (pyproject.toml) cannot be found.
    """
    current = Path(__file__).resolve().parent
    while current != current.parent:  # Stop at filesystem root
        if (current / "pyproject.toml").exists():
            config_dir = current / "configs"
            if not config_dir.exists():
                msg = f"Project root found at {current}, but configs directory missing"
                raise FileNotFoundError(msg)
            return str(config_dir)
        current = current.parent

    msg = "Could not find project root (pyproject.toml)"
    raise FileNotFoundError(msg)


class RunCollator:
    """Collate results from multiple runs in an outputs directory.

    Parameters
    ----------
    outputs_dir : str | Path, default="outputs"
        Path to the outputs directory containing runs.
    config_params : dict[str, str] | None, default=None
        Dictionary mapping output column names to config paths.
        If None, uses default parameters. Example:
        {"encoder": "model.encoder._target_"}
    """

    # Default parameters to extract from config files
    DEFAULT_CONFIG_PARAMS: ClassVar[dict[str, str]] = {
        "encoder": "model.encoder._target_",
        "processor": "model.processor._target_",
        "decoder": "model.decoder._target_",
        "hidden_size": "model.processor.hidden_dim",
        "n_members": "model.n_members",
        "batch_size": "datamodule.batch_size",
        "use_normalization": "datamodule.use_normalization",
        "learning_rate": "optimizer.learning_rate",
        "loss_function": "model.loss_func._target_",
        "train_in_latent": "model.train_in_latent_space",
    }

    def __init__(
        self,
        outputs_dir: str | Path = "outputs",
        config_params: dict[str, str] | None = None,
    ):
        """Initialize the RunCollator."""
        self.log = logging.getLogger(__name__)

        # Resolve outputs directory
        self.outputs_path = Path(outputs_dir).expanduser().resolve()
        if not self.outputs_path.exists():
            msg = f"Outputs directory not found: {self.outputs_path}"
            raise FileNotFoundError(msg)

        # Set config parameters
        self.config_params = config_params or self.DEFAULT_CONFIG_PARAMS

    def _discover_runs(self) -> list[Path]:
        """Discover all valid run directories.

        A valid run must have:
        - resolved_config.yaml
        - eval/evaluation_metrics.csv

        Returns
        -------
        list[Path]
            List of run directory paths, sorted by modification time.
        """
        runs = []
        for config_file in self.outputs_path.rglob("resolved_config.yaml"):
            run_dir = config_file.parent
            eval_metrics = run_dir / "eval" / "evaluation_metrics.csv"
            if eval_metrics.exists():
                runs.append(run_dir)
        return sorted(runs, key=lambda p: p.stat().st_mtime, reverse=True)

    def _extract_run_metadata(self, run_dir: Path) -> dict[str, str]:
        """Extract run metadata (name, date, path, category).

        Parameters
        ----------
        run_dir : Path
            Path to the run directory.

        Returns
        -------
        dict[str, str]
            Dictionary with run_path, run_name, date, and category.
        """
        metadata = {}

        # Relative path from outputs
        metadata["run_path"] = str(run_dir.relative_to(self.outputs_path))

        # Run name (directory name)
        metadata["run_name"] = run_dir.name

        # Try to extract date from parent directory
        parent_name = run_dir.parent.name
        date_match = re.match(r"(\d{4}-\d{2}-\d{2})", parent_name)
        if date_match:
            metadata["date"] = date_match.group(1)
            metadata["category"] = "date-based"
        elif parent_name == "processor":
            # Extract from run name (YYYYMMDD_HHMMSS)
            timestamp_match = re.match(r"(\d{8})_(\d{6})", run_dir.name)
            if timestamp_match:
                yyyymmdd = timestamp_match.group(1)
                metadata["date"] = f"{yyyymmdd[:4]}-{yyyymmdd[4:6]}-{yyyymmdd[6:8]}"
            else:
                metadata["date"] = "unknown"
            metadata["category"] = "processor"
        elif parent_name == "autoencoder":
            # Extract from run name (YYYY-MM-DD_HH-MM-SS)
            date_match = re.match(r"(\d{4}-\d{2}-\d{2})", run_dir.name)
            if date_match:
                metadata["date"] = date_match.group(1)
            else:
                metadata["date"] = "unknown"
            metadata["category"] = "autoencoder"
        else:
            metadata["date"] = "unknown"
            metadata["category"] = "other"

        return metadata

    def _get_nested_value(self, config: dict, path: str, default: Any = "N/A") -> Any:
        """Get a value from nested dictionary using dot notation.

        Parameters
        ----------
        config : dict
            Configuration dictionary.
        path : str
            Dot-separated path (e.g., "model.encoder._target_").
        default : Any, default="N/A"
            Default value if path not found.

        Returns
        -------
        Any
            Value at the specified path, or default if not found.
        """
        keys = path.split(".")
        value = config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value

    def _simplify_target_name(self, target: str) -> str:
        """Simplify a _target_ value by extracting the last component.

        Parameters
        ----------
        target : str
            Full target path (e.g., "autocast.models.encoder.Encoder").

        Returns
        -------
        str
            Simplified name (e.g., "Encoder"), or "Unknown" if empty.
        """
        if not target or target == "N/A":
            return "Unknown"
        return target.split(".")[-1]

    def _extract_config_params(self, config: dict) -> dict[str, Any]:
        """Extract specified parameters from config using configured paths.

        Parameters
        ----------
        config : dict
            Loaded configuration dictionary.

        Returns
        -------
        dict[str, Any]
            Dictionary with extracted parameter values.
        """
        info = {}

        for param_name, config_path in self.config_params.items():
            value = self._get_nested_value(config, config_path)

            # Simplify _target_ values
            if config_path.endswith("._target_") and value != "N/A":
                value = self._simplify_target_name(value)

            info[param_name] = value

        return info

    def _parse_metrics(self, run_dir: Path) -> dict[str, Any]:
        """Parse both evaluation_metrics.csv and rollout_metrics.csv.

        Parameters
        ----------
        run_dir : Path
            Path to the run directory.

        Returns
        -------
        dict[str, Any]
            Dictionary with overall metrics and windowed metrics.
        """
        metrics: dict[str, Any] = {}

        # Parse overall metrics
        eval_csv = run_dir / "eval" / "evaluation_metrics.csv"
        if eval_csv.exists():
            try:
                df_eval = pd.read_csv(eval_csv)
                # Get row where window='all' and batch_idx='all'
                overall_row = df_eval[
                    (df_eval["window"] == "all") & (df_eval["batch_idx"] == "all")
                ]
                if not overall_row.empty:
                    for col in ["mse", "mae", "rmse", "vrmse", "coverage"]:
                        if col in overall_row.columns:
                            # Get the scalar value from the first row
                            value = overall_row.iloc[0][col]
                            metrics[f"overall_{col}"] = value
            except Exception as e:
                self.log.warning(
                    "Failed to parse evaluation metrics from %s: %s", eval_csv, e
                )

        # Parse windowed metrics
        rollout_csv = run_dir / "eval" / "rollout_metrics.csv"
        if rollout_csv.exists():
            try:
                df_rollout = pd.read_csv(rollout_csv)
                # Get rows where batch_idx='all'
                windowed = df_rollout[df_rollout["batch_idx"] == "all"]
                for _, row in windowed.iterrows():
                    window = str(row["window"])
                    for col in ["mse", "mae", "rmse", "vrmse", "coverage"]:
                        if col in row.index:
                            metrics[f"{col}_{window}"] = float(row[col])
            except Exception as e:
                self.log.warning(
                    "Failed to parse rollout metrics from %s: %s", rollout_csv, e
                )

        return metrics

    def _process_single_run(self, run_dir: Path) -> dict[str, Any] | None:
        """Process a single run directory, handling errors gracefully.

        Parameters
        ----------
        run_dir : Path
            Path to the run directory.

        Returns
        -------
        dict[str, Any] | None
            Dictionary of run data, or None if processing fails.
        """
        try:
            # Extract metadata
            run_data = self._extract_run_metadata(run_dir)

            # Load and parse config
            config_file = run_dir / "resolved_config.yaml"
            try:
                with open(config_file) as f:
                    config = yaml.safe_load(f)
                config_info = self._extract_config_params(config)
                run_data.update(config_info)
            except Exception as e:
                self.log.warning("Failed to parse config for %s: %s", run_dir.name, e)
                # Continue with missing config info

            # Parse metrics
            try:
                metrics = self._parse_metrics(run_dir)
                run_data.update(metrics)
            except Exception as e:
                self.log.warning("Failed to parse metrics for %s: %s", run_dir.name, e)
                # Continue with missing metrics

            return run_data

        except Exception as e:
            self.log.error("Failed to process run %s: %s", run_dir.name, e)
            return None

    def _save_results(self, df: pd.DataFrame, output_path: Path) -> None:
        """Save DataFrame to CSV following existing patterns.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to save.
        output_path : Path
            Path where the CSV file will be saved.
        """
        if df.empty:
            self.log.warning("No runs to save; DataFrame is empty.")
            return

        # Sort by date (descending) and run name
        if "date" in df.columns:
            df = df.sort_values(["date", "run_name"], ascending=[False, True])

        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        self.log.info("Saved collated results to %s", output_path)

    def collate(
        self,
        output_csv: str | Path = "collated_results.csv",
        save_csv: bool = True,
    ) -> pd.DataFrame:
        """Collate results from multiple runs in the outputs directory.

        Parameters
        ----------
        output_csv : str | Path, default="collated_results.csv"
            Path where the CSV file will be saved.
        save_csv : bool, default=True
            Whether to save the DataFrame to CSV.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns for run metadata and metrics.
        """
        # Discover runs
        runs = self._discover_runs()
        self.log.info("Found %d valid runs", len(runs))

        # Process each run
        results = []
        for run_dir in runs:
            run_data = self._process_single_run(run_dir)
            if run_data is not None:
                results.append(run_data)

        # Create DataFrame
        df = pd.DataFrame(results)
        self.log.info("Successfully processed %d runs", len(df))

        # Save to CSV
        if save_csv:
            output_path = Path(output_csv).expanduser().resolve()
            self._save_results(df, output_path)

        return df


def collate_run_results(
    outputs_dir: str | Path = "outputs",
    output_csv: str | Path = "collated_results.csv",
    save_csv: bool = True,
    config_params: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Collate results from multiple runs in the outputs directory.

    This is a convenience function that creates a RunCollator instance
    and calls its collate method.

    Parameters
    ----------
    outputs_dir : str | Path, default="outputs"
        Path to the outputs directory containing runs.
    output_csv : str | Path, default="collated_results.csv"
        Path where the CSV file will be saved.
    save_csv : bool, default=True
        Whether to save the DataFrame to CSV.
    config_params : dict[str, str] | None, default=None
        Dictionary mapping output column names to config paths (dot notation).
        If None, uses default parameters.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns for run metadata and metrics.
    """
    collator = RunCollator(outputs_dir=outputs_dir, config_params=config_params)
    return collator.collate(output_csv=output_csv, save_csv=save_csv)
