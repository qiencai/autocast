# AutoCast

## Installation

For development, install with [`uv`](https://github.com/astral-sh/uv):
```bash
uv sync --extra dev
```

If contributing to the codebase, you can run 
```bash 
 pre-commit install 
 ```
This will setup the pre-commit checks so any pushed commits will pass the CI. 

## Quickstart

Train an encoder-decoder stack and evaluate the resulting checkpoint:

```bash
# Train
uv run python -m autocast.train.autoencoder --config-path=configs/
```

Train an encoder-processor-decoder stack and evaluate the resulting checkpoint:

```bash
# Train
uv run train_encoder_processor_decoder \
    --config-path=configs/ \
	--work-dir=outputs/encoder_processor_decoder_run
	
# Evaluate
uv run evaluate_encoder_processor_decoder \
	--config-path=configs/ \
	--work-dir=outputs/processor_eval \
	--checkpoint=outputs/encoder_processor_decoder_run/encoder_processor_decoder.ckpt \
	--batch-index=0 --batch-index=3 \
	--video-dir=outputs/encoder_processor_decoder_run/videos
```

Evaluation writes a CSV of aggregate metrics to `--csv-path` (defaults to
`<work-dir>/evaluation_metrics.csv`) and, when `--batch-index` is provided,
stores rollout animations for the specified test batches.

## Experiment Tracking with Weights & Biases

AutoCast now ships with an optional [Weights & Biases](https://wandb.ai/) integration that is
fully driven by the Hydra config under `configs/logging/wandb.yaml`.

- Enable logging for CLI workflows by passing Hydra config overrides as positional arguments:

	```bash
	uv run train_encoder_processor_decoder \
		--config-path=configs \
		logging.wandb.enabled=true \
		logging.wandb.project=autocast-experiments \
		logging.wandb.name=processor-baseline
	```

- The autoencoder/processor training CLIs pass the configured `WandbLogger` directly into Lightning so that metrics, checkpoints, and artifacts are synchronized automatically.
- The evaluation CLI reports aggregate test metrics to the same run when logging is enabled, making it easy to compare training and evaluation outputs in one dashboard.
- All notebooks contain a dedicated cell that instantiates a `wandb_logger` via `autocast.logging.create_wandb_logger`. Toggle the `enabled` flag in that cell to control tracking when experimenting interactively.

When `enabled` remains `false` (the default), the logger is skipped entirely, so the stack can
be used without a W&B account.

## Running on HPC 

In the [Scripts](/scripts/) folders, slurm scripts can be found for running training and evaluation on the baskerville HPC. 

### Single Job 

To run, simply navigate to the top level of this repository, and run: 

`sbatch scripts/train_and_eval_encoder-processor-decoder.sh` or 
`sbatch scripts/train_and_eval_autoencoder.sh` depending on which model you would like to run.

This will train and evaluate the model using the settings in the corresponding config (found in the configs folder). Outputs from both train and eval will be written out to an outputs folder with the following naming convention: 

`outputs/{job_name}/{$date +%Y%m%d_%H%M%S}`. 

### Multiple Jobs

`scripts/encoder-processor-decoder-parameter_sweep.sh` is an example parameter sweep. 

It uses slurm arrays and hydra override functionality to sweep through combinations of parameters. The resulting output structure looks like this: 

- outputs	
	- {job_name}
		- Job-{job_id} # Unique for each sweep run 
			- parameter_lookup.csv # csv file mapping task id to parameter values. 
			- Task-0 # 0 is the slurm array task id. It is unique for each set of parameters
			- Task-1
			- etc. 

A checklist of things to change in the example script:

- `--array=0-8` : This is the number of parallel jobs to run. This should be equal to the number of parameter combinations you want to run. 
- `JOB_NAME="encoder_processor_decoder_sweep"` : Name of the Job. This is the top level directory. 
- The whole Define Parameter Grid section. 
- The columns to be writte to the parameter csv file 
- The flags in the python script to overright the hydra config. e.g. `trainer.max_epochs=${MAX_EPOCH}`
