"""Generate expert trajectory buffers for Trajectory Matching (objective=TM).

Trains several bert-* "experts" on the real dataset by linear probing -- the
pretrained body is frozen and only the classifier head is updated with plain
SGD (so the trajectory is matchable by the SGD student rollout in
TrainerDC._tm_inner_step). The classifier-head parameters are snapshotted every
`snapshot_interval` steps and saved to disk.

Run once before a TM distillation run, e.g.:

    python src/buffer.py --config-name=buffer \
        data.task_name=sst2 learner.model_name=prajjwal1/bert-tiny \
        learner.gradient_checkpointing=False

The output path must match train.tm_buffer_path used by the DC run.
"""

import logging
import os
from dataclasses import dataclass

import hydra
import torch
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from torch.cuda import amp
from transformers import set_seed

from data import DataConfig, DataModule
from generator import GeneratorConfig, GeneratorModel
from learner import LearnerConfig, LearnerModel
from utils import batch_to_cuda, endless_dataloader

logger = logging.getLogger(__name__)


@dataclass
class BaseConfig:
    experiment_name: str
    method: str
    run_name: str
    save_dir_root: str
    save_method_dir: str
    save_dir: str
    data_dir_root: str
    seed: int = 42


@dataclass
class BufferConfig:
    num_experts: int = 10
    expert_train_steps: int = 200
    snapshot_interval: int = 10
    expert_lr: float = 1.0e-2
    batch_size: int = 64
    fp16: bool = False
    bf16: bool = True
    save_path: str = "path/to/tm_buffer.pt"


@dataclass
class Config:
    base: BaseConfig
    data: DataConfig
    generator: GeneratorConfig
    learner: LearnerConfig
    buffer: BufferConfig


cs = ConfigStore.instance()
cs.store(name="config", node=Config)


@hydra.main(config_path="../configs/train", config_name="buffer", version_base=None)
def main(config: Config):
    logger.info(f"Config:\n{OmegaConf.to_yaml(config)}")
    set_seed(config.base.seed)

    # The DataModule needs a generator for its tokenization plumbing even though
    # the buffer only uses the learner side.
    generator = GeneratorModel(config.generator, task_name=config.data.task_name)
    learner = LearnerModel(config.learner, task_name=config.data.task_name)
    data_module = DataModule(config.data, generator=generator, learner=learner)

    learner.cuda()
    head_names = list(learner.classifier_param_names())
    head_set = set(head_names)
    logger.info(f"Classifier head parameters: {head_names}")

    use_amp = config.buffer.fp16 or config.buffer.bf16
    amp_dtype = torch.float16 if config.buffer.fp16 else torch.bfloat16

    snapshot_steps = list(
        range(0, config.buffer.expert_train_steps + 1, config.buffer.snapshot_interval)
    )
    snapshot_set = set(snapshot_steps)

    def snapshot_head() -> dict:
        return {
            n: p.detach().float().cpu().clone()
            for n, p in learner.named_parameters()
            if n in head_set
        }

    experts = []
    for e in range(config.buffer.num_experts):
        logger.info(f"Training expert {e + 1}/{config.buffer.num_experts}")

        # fresh head + pretrained (deterministic) body; freeze everything but head
        learner.init_weights()
        for name, p in learner.named_parameters():
            p.requires_grad = name in head_set
        head_params = [p for n, p in learner.named_parameters() if n in head_set]

        optimizer = torch.optim.SGD(head_params, lr=config.buffer.expert_lr)
        learner.train()

        train_loader = data_module.get_train_loader(
            batch_size=config.buffer.batch_size
        )
        train_loader = endless_dataloader(
            train_loader, max_iteration=config.buffer.expert_train_steps + 1
        )

        snapshots = [snapshot_head()]  # step 0
        for step in range(1, config.buffer.expert_train_steps + 1):
            batch = next(train_loader)
            with amp.autocast(enabled=use_amp, dtype=amp_dtype):
                outputs = learner(**batch_to_cuda(batch["learner"]))
                loss = outputs.loss.mean()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            if step in snapshot_set:
                snapshots.append(snapshot_head())

        logger.info(
            f"Expert {e + 1}: {len(snapshots)} snapshots, final loss={loss.item():.4f}"
        )
        experts.append(snapshots)

    save_path = config.buffer.save_path
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(
        {
            "head_param_names": head_names,
            "snapshot_steps": snapshot_steps,
            "experts": experts,
        },
        save_path,
    )
    logger.info(
        "Saved TM buffer: {} experts x {} snapshots -> {}".format(
            len(experts), len(experts[0]), save_path
        )
    )


if __name__ == "__main__":
    main()
