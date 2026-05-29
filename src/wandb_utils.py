"""Weights & Biases logging helpers.

Thin wrapper around the global ``wandb`` API so the rest of the codebase can log
metrics / artifacts without repeating boilerplate.  Every function is a no-op
when there is no active run (``wandb.run is None``), so importing modules stay
usable in contexts where W&B was never initialised (e.g. unit tests).

``wandb.login()`` is assumed to have already been performed in the environment.
"""

import logging
import os
import re
from typing import Optional, Sequence

import wandb
from datasets import Dataset
from omegaconf import OmegaConf

logger = logging.getLogger(__name__)

# All training / validation curves share this value as their x-axis instead of
# W&B's internal monotonic step counter.  This avoids "step must be monotonically
# increasing" issues caused by validation (logged at step N) and training
# (logged at step N too) landing on the same step.
_X_AXIS = "train_step"

_ARTIFACT_NAME_RE = re.compile(r"[^a-zA-Z0-9_.\-]+")


def _sanitize(name: str) -> str:
    """Coerce an arbitrary string into a valid W&B artifact name."""
    return _ARTIFACT_NAME_RE.sub("-", name).strip("-") or "artifact"


def init_run(config, extra_config: Optional[dict] = None) -> "wandb.sdk.wandb_run.Run":
    """Start a W&B run, recording the full (resolved) Hydra config.

    Project, entity and run name are all read from ``config.base`` so they can
    be overridden on the CLI (e.g. ``base.wandb_name=my-run``).
    """
    run = wandb.init(
        project=config.base.wandb_project,
        entity=config.base.wandb_entity,
        name=config.base.wandb_name,
        config=OmegaConf.to_container(config, resolve=True, throw_on_missing=False),
    )
    if extra_config:
        wandb.config.update(extra_config, allow_val_change=True)

    # Use train_step as the shared x-axis for training and validation curves.
    wandb.define_metric(_X_AXIS)
    wandb.define_metric("train.*", step_metric=_X_AXIS)
    wandb.define_metric("valid.*", step_metric=_X_AXIS)
    return run


def log_metrics(metrics: dict, step: Optional[int] = None) -> None:
    """Log scalar metrics. When ``step`` is given it becomes the x-axis value."""
    if wandb.run is None:
        return
    payload = dict(metrics)
    if step is not None:
        payload[_X_AXIS] = step
    wandb.log(payload)


def log_path_artifact(
    path: str,
    name: str,
    type: str,
    metadata: Optional[dict] = None,
) -> None:
    """Log a file or directory as a versioned W&B artifact."""
    if wandb.run is None:
        return
    if not os.path.exists(path):
        logger.warning(f"Artifact path does not exist, skipping: {path}")
        return

    artifact = wandb.Artifact(name=_sanitize(name), type=type, metadata=metadata)
    if os.path.isdir(path):
        artifact.add_dir(path)
    else:
        artifact.add_file(path)
    wandb.run.log_artifact(artifact)


def log_generation_examples(
    dataset_list: Sequence[Dataset],
    sentence_keys: Sequence[str],
    label_dict: Optional[dict],
    step: int,
    max_per_dataset: int = 10,
) -> None:
    """Log a few generated examples as a W&B Table for in-UI inspection."""
    if wandb.run is None or not dataset_list:
        return

    columns = [_X_AXIS, "dataset_idx", "label", *sentence_keys]
    table = wandb.Table(columns=columns)
    for di, dataset in enumerate(dataset_list):
        for i in range(min(max_per_dataset, len(dataset))):
            example = dataset[i]
            label = example.get("labels")
            label_name = label_dict.get(label, label) if label_dict else label
            table.add_data(
                step,
                di,
                label_name,
                *[example.get(key, "") for key in sentence_keys],
            )
    wandb.log({"generations": table})


def finish() -> None:
    if wandb.run is not None:
        wandb.finish()
