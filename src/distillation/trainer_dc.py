import logging
import os
import random
from contextlib import contextmanager, nullcontext
from typing import Generator

import torch
import wandb_utils
from datasets import Dataset, concatenate_datasets
from torch import nn
from torch.cuda import amp
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import trange

from coreset import CoresetModule
from data import DataModule
from evaluator import EvaluateConfig, Evaluator
from generator import GeneratorModel
from learner import LearnerModel
from utils import average, batch_to_cuda, configure_optimizer, endless_dataloader

from .trainer_base import TrainerBase

logger = logging.getLogger(__name__)


@contextmanager
def math_sdp_attention():
    """Force the math scaled-dot-product-attention backend.

    Full gradient matching double-backprops through the learner (the cosine
    loss is differentiated through ``torch.func.grad``).  The fused flash /
    mem-efficient SDPA kernels do not implement a second-order derivative
    (``derivative for aten::_scaled_dot_product_efficient_attention_backward
    is not implemented``).  The math backend decomposes attention into
    differentiable matmul/softmax ops that support higher-order autograd.
    """
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel

        with sdpa_kernel(SDPBackend.MATH):
            yield
        return
    except ImportError:
        pass

    # Fallback for torch < 2.3 without torch.nn.attention.
    cuda = torch.backends.cuda
    prev = (
        cuda.flash_sdp_enabled(),
        cuda.mem_efficient_sdp_enabled(),
        cuda.math_sdp_enabled(),
    )
    cuda.enable_flash_sdp(False)
    cuda.enable_mem_efficient_sdp(False)
    cuda.enable_math_sdp(True)
    try:
        yield
    finally:
        cuda.enable_flash_sdp(prev[0])
        cuda.enable_mem_efficient_sdp(prev[1])
        cuda.enable_math_sdp(prev[2])


class TrainerDC(TrainerBase):
    def _build_dm_projection_heads(
        self, hidden_dim: int, device: torch.device
    ) -> nn.ModuleList:
        projection_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim, self.config.dm_projection_hidden_dim),
                    nn.GELU(),
                    nn.Linear(
                        self.config.dm_projection_hidden_dim,
                        self.config.dm_projection_dim,
                    ),
                )
                for _ in range(self.config.dm_projection_heads)
            ]
        ).to(device)

        for projection_head in projection_heads:
            projection_head.eval()
            for param in projection_head.parameters():
                param.requires_grad = False

        return projection_heads

    def fit(
        self,
        generator: GeneratorModel,
        learner: LearnerModel,
        data_module: DataModule,
        evaluator: Evaluator,
        repset_teachers: list[Dataset] | None,
        coreset_module: CoresetModule,
    ):
        generator.cuda()
        learner.cuda()

        num_labels = data_module.num_labels

        assert self.config.objective in ("GM", "HM", "TM", "OGM"), self.config.objective
        is_tm = self.config.objective == "TM"
        if self.config.objective == "HM":
            assert self.config.num_hvp_vectors >= 1, self.config.num_hvp_vectors
            logger.info(
                "Objective=HM: matching gradients + {} Hessian-vector product(s)".format(
                    self.config.num_hvp_vectors
                )
            )
        if self.config.objective == "OGM":
            assert (
                self.config.classifier_grad_only
            ), "OGM requires classifier_grad_only=True (PCA over full grads is infeasible)"
            assert self.config.ogm_num_pcs >= 1, self.config.ogm_num_pcs
            logger.info(
                "Objective=OGM: GM mean + variance profile over {} real-gradient PCs"
                " (lambda={})".format(
                    self.config.ogm_num_pcs, self.config.ogm_lambda
                )
            )
        if is_tm:
            assert self.config.tm_syn_steps >= 1, self.config.tm_syn_steps
            assert self.config.tm_expert_snapshot_gap >= 1
            expert_buffer = self._load_tm_buffer(learner)
            logger.info(
                "Objective=TM: {} experts x {} snapshots, N={} student steps".format(
                    len(expert_buffer["experts"]),
                    len(expert_buffer["experts"][0]),
                    self.config.tm_syn_steps,
                )
            )

        assert self.config.total_train_step % self.config.inner_loop == 0
        outer_loop = self.config.total_train_step // self.config.inner_loop
        assert self.config.val_interval % self.config.inner_loop == 0
        assert outer_loop % (self.config.val_interval // self.config.inner_loop) == 0
        assert self.config.log_interval % self.config.inner_loop == 0
        assert outer_loop % (self.config.log_interval // self.config.inner_loop) == 0

        # setup data loader for gm loss (real-data gradients; not used by TM)
        if not is_tm:
            gm_real_loaders = self.get_gm_real_loaders(
                data_module, learner=learner, repset_teachers=repset_teachers
            )

        # setup data loader for lm loss
        if self.config.lm_lambda > 0:
            lm_loader = self.get_lm_loader(data_module)

        # setup data loader for updating learner (GM/HM only; TM freezes the body)
        if self.config.inner_loop > 1 and not is_tm:
            learner_train_loader = self.get_learner_train_loader(data_module)

        if not self.config.use_generated_data:
            raise NotImplementedError
            # if self.config.n_clusters_for_syn_sampler > 1:
            #     gm_syn_loaders = self.cluster_wise_dataloader(
            #         data_module.preprocessed_datasets["train"],
            #         data_module=data_module,
            #         learner=learner,
            #         dpc=self.config.gm_syn_dpc,
            #         n_clusters=self.config.n_clusters_for_syn_sampler,
            #         max_iteration=self.config.inner_loop
            #         * self.config.generate_dataset_interval,
            #     )
            # else:
            #     gm_syn_loaders = {
            #         label: endless_dataloader(
            #             data_module.get_train_loader(
            #                 batch_size=self.config.gm_syn_dpc,
            #                 shuffle=False,
            #                 drop_last=False,
            #                 label=label,
            #             ),
            #             max_iteration=self.config.total_train_step,
            #         )
            #         for label in range(num_labels)
            #     }

        # setup optimizer
        optimizer, scheduler = self.generator_optimizer(generator)
        scaler = amp.GradScaler(enabled=self.use_amp)
        optimizer.zero_grad(set_to_none=True)

        # setup for torch.functional_call
        params = {k: v.detach() for k, v in learner.named_parameters()}
        buffers = {k: v for k, v in learner.named_buffers()}

        # save tokenizer
        tokenizer_path = os.path.join(self.config.save_model_dir, "tokenizer")
        generator.save_tokenizer(tokenizer_path)

        # save best model path
        best_ckpt_path = os.path.join(self.config.save_model_dir, "best-ckpt")

        train_logs = []
        best_val_score = float("-inf")
        dm_projection_heads = None
        logger.info("Start training!!")
        for ol in trange(
            outer_loop, dynamic_ncols=True, leave=False, desc="Outer Loop"
        ):
            # train_step = ol * self.config.inner_loop
            # evaluate before training
            if (
                (ol * self.config.inner_loop) % self.config.val_interval == 0
                and ol * self.config.inner_loop >= self.config.val_skip_step
            ):
                results = self.evaluate(
                    generator,
                    learner,
                    evaluator,
                    data_module,
                    coreset_module,
                    step=ol * self.config.inner_loop,
                )
                if results[f"valid.{evaluator.metric_key}"] > best_val_score:
                    best_val_score = results[f"valid.{evaluator.metric_key}"]
                    generator.save_model(best_ckpt_path)
                    logger.info(f"Save best checkpoint at `{best_ckpt_path}`")

            # generate dataset
            if (
                ol % self.config.generate_dataset_interval == 0
                and self.config.use_generated_data
            ):
                logger.info(
                    "TRAIN [{:>{}}/{}]: Generate synthetic data".format(
                        ol * self.config.inner_loop,
                        len(str(self.config.total_train_step)),
                        self.config.total_train_step,
                    )
                )
                gm_syn_loaders = self.get_gm_syn_loaders(
                    generator, learner, data_module
                )

            if not is_tm:
                # GM/HM: reset the learner each outer loop and (optionally) train
                # it on real data between inner steps. TM keeps a fixed frozen
                # body and rolls out only the classifier head per step.
                learner.init_weights()
                if self.config.inner_loop > 1:
                    learner_optimizer, learner_scheduler = self.learner_optimizer(
                        learner, evaluate_config=evaluator.config
                    )
                    learner_scaler = amp.GradScaler(enabled=self.use_amp)

            generator.train()

            outer_loop_train_logs = []
            for outer_step in range(self.config.inner_loop):
                # compute DC loss
                grad_sim = 0.0
                hvp_sim = 0.0
                ogm_sim = 0.0
                loss_dm = 0.0
                loss_tm = 0.0
                if is_tm:
                    # trajectory matching: roll out the classifier head on
                    # synthetic data and match the expert head trajectory.
                    loss_tm_tensor = self._tm_inner_step(
                        generator=generator,
                        learner=learner,
                        params=params,
                        buffers=buffers,
                        gm_syn_loaders=gm_syn_loaders,
                        expert_buffer=expert_buffer,
                        num_labels=num_labels,
                    )
                    scaler.scale(
                        loss_tm_tensor * (1 - self.config.lm_lambda)
                    ).backward()
                    loss_tm = loss_tm_tensor.item()
                    loss_dc = loss_tm
                elif self.config.lm_lambda < 1:
                    is_hm = self.config.objective == "HM"
                    is_ogm = self.config.objective == "OGM"
                    for label in range(num_labels):
                        if is_ogm:
                            # orthogonal GM (classifier-head only): matching loss
                            # from per-sample gradients, fp32 (no autocast). The
                            # backward is first-order through loss_weights, so no
                            # DM term is combined here.
                            (
                                loss_dc_label,
                                grad_sim_label,
                                ogm_sim_label,
                            ) = self._ogm_label(
                                generator=generator,
                                learner=learner,
                                params=params,
                                buffers=buffers,
                                gm_real_loaders=gm_real_loaders,
                                gm_syn_loaders=gm_syn_loaders,
                                label=label,
                                num_labels=num_labels,
                            )
                            scaler.scale(
                                loss_dc_label * (1 - self.config.lm_lambda)
                            ).backward()
                            grad_sim += grad_sim_label.item()
                            ogm_sim += ogm_sim_label.item()
                            continue
                        # shared random probe directions for real/syn HVPs (HM only)
                        hvp_vectors = (
                            self.sample_hvp_vectors(
                                learner, params, self.config.num_hvp_vectors
                            )
                            if is_hm
                            else []
                        )
                        with amp.autocast(enabled=self.use_amp, dtype=self.amp_dtype):
                            # compute gradient (+ HVPs) with real samples
                            # sample size: gm_real_dpc * gm_real_grad_accum_step
                            with torch.no_grad():
                                grad_real_list = []
                                hvp_real_lists = [[] for _ in hvp_vectors]
                                for _ in range(self.config.gm_real_grad_accum_step):
                                    batch_gm_real = next(gm_real_loaders[label])
                                    real_learner = batch_to_cuda(
                                        batch_gm_real["learner"]
                                    )
                                    if is_hm:
                                        g_real, h_real = self.compute_grad_and_hvps(
                                            learner=learner,
                                            params=params,
                                            buffers=buffers,
                                            vectors=hvp_vectors,
                                            **real_learner,
                                        )
                                        for k, h in enumerate(h_real):
                                            hvp_real_lists[k].append(h)
                                    else:
                                        g_real = self.compute_grad(
                                            learner=learner,
                                            params=params,
                                            buffers=buffers,
                                            **real_learner,
                                        )
                                    grad_real_list.append(g_real)

                                grad_real = torch.stack(grad_real_list).mean(0)
                                if is_hm:
                                    hvp_real = [
                                        torch.stack(hl).mean(0) for hl in hvp_real_lists
                                    ]

                            # compute generation probability
                            batch_gm_syn = next(gm_syn_loaders[label])
                            gen_losses = generator.compute_loss(
                                **batch_to_cuda(batch_gm_syn["generator"])
                            )
                            loss_weights = F.softmax(
                                -gen_losses / self.config.normalize_temperature,
                                dim=-1,
                            )
                            # compute gradient (+ HVPs) with loss weights
                            syn_learner = batch_to_cuda(batch_gm_syn["learner"])
                            if is_hm:
                                grad_syn, hvp_syn = self.compute_grad_and_hvps(
                                    learner=learner,
                                    params=params,
                                    buffers=buffers,
                                    vectors=hvp_vectors,
                                    **syn_learner,
                                    loss_weights=loss_weights,
                                )
                            else:
                                grad_syn = self.compute_grad(
                                    learner=learner,
                                    params=params,
                                    buffers=buffers,
                                    **syn_learner,
                                    loss_weights=loss_weights,
                                )

                            grad_sim_label = F.cosine_similarity(
                                grad_real, grad_syn, dim=0
                            )
                            if is_hm:
                                # cosine distance averaged over the HVP probes
                                hvp_sim_label = torch.stack(
                                    [
                                        F.cosine_similarity(hr, hs, dim=0)
                                        for hr, hs in zip(hvp_real, hvp_syn)
                                    ]
                                ).mean()
                                loss_dc_label = (
                                    (1 - grad_sim_label) + (1 - hvp_sim_label)
                                ) / num_labels
                            else:
                                hvp_sim_label = None
                                loss_dc_label = (1 - grad_sim_label) / num_labels

                        loss_label = loss_dc_label * (1 - self.config.lm_lambda)

                        if self.config.dm_lambda > 0:
                            with amp.autocast(
                                enabled=self.use_amp, dtype=self.amp_dtype
                            ):
                                outputs_real = learner(
                                    **batch_to_cuda(batch_gm_real["learner"]),
                                    output_hidden_states=True,
                                )
                                outputs_syn = learner(
                                    **batch_to_cuda(batch_gm_syn["learner"]),
                                    output_hidden_states=True,
                                )
                                h_real = outputs_real.hidden_states[
                                    self.config.dm_feature_layer
                                ][:, 0]
                                h_syn = outputs_syn.hidden_states[
                                    self.config.dm_feature_layer
                                ][:, 0]

                                if dm_projection_heads is None:
                                    dm_projection_heads = (
                                        self._build_dm_projection_heads(
                                            hidden_dim=h_real.shape[-1],
                                            device=h_real.device,
                                        )
                                    )

                                dist = 0.0
                                for projection_head in dm_projection_heads:
                                    z_real = projection_head(h_real)
                                    z_syn = projection_head(h_syn)
                                    mu_real = z_real.mean(dim=0, keepdim=True)
                                    dist_k = ((z_syn - mu_real) ** 2).mean(dim=1)
                                    dist = dist + dist_k
                                dist = dist / len(dm_projection_heads)
                                reward = -dist
                                if self.config.dm_normalize_reward:
                                    reward = (reward - reward.mean()) / (
                                        reward.std(unbiased=False)
                                        + self.config.dm_reward_eps
                                    )
                                reward = reward.detach()
                                loss_dm_label = (reward * gen_losses).mean()

                            loss_label = (
                                loss_label
                                + loss_dm_label * self.config.dm_lambda / num_labels
                            )
                            loss_dm += loss_dm_label.item()

                        scaler.scale(loss_label).backward()

                        grad_sim += grad_sim_label.item()
                        if hvp_sim_label is not None:
                            hvp_sim += hvp_sim_label.item()

                    grad_sim /= num_labels
                    hvp_sim /= num_labels
                    ogm_sim /= num_labels
                    # loss_dc reflects the matched quantities: gradient distance,
                    # plus Hessian distance (HM) or PC-variance distance (OGM).
                    loss_dc = (1 - grad_sim)
                    if is_hm:
                        loss_dc += 1 - hvp_sim
                    if is_ogm:
                        loss_dc += self.config.ogm_lambda * (1 - ogm_sim)
                    loss_dm /= num_labels
                else:
                    loss_dc = 0.0
                    loss_dm = 0.0

                if self.config.lm_lambda > 0:
                    batch_lm = next(lm_loader)
                    with amp.autocast(enabled=self.use_amp, dtype=self.amp_dtype):
                        loss_lm = generator.compute_loss(
                            **batch_to_cuda(batch_lm["generator"])
                        )
                        loss_lm = loss_lm.mean()

                    # backward for lm loss
                    scaler.scale(loss_lm * self.config.lm_lambda).backward()
                    loss_lm = loss_lm.item()
                else:
                    loss_lm = 0.0

                loss = (
                    loss_dc * (1 - self.config.lm_lambda)
                    + loss_dm * self.config.dm_lambda
                    + loss_lm * self.config.lm_lambda
                )

                # update generator
                self.train_step(generator, optimizer, scheduler, scaler)

                outer_loop_train_log = {
                    "train.loss": loss,
                    "train.loss_dc": loss_dc,
                    "train.loss_lm": loss_lm,
                    "train.loss_dm": loss_dm,
                    "train.loss_tm": loss_tm,
                    "train.grad_sim": grad_sim,
                    "train.hvp_sim": hvp_sim,
                    "train.ogm_sim": ogm_sim,
                }
                outer_loop_train_logs.append(outer_loop_train_log)

                # update learner (GM/HM only; TM keeps the body frozen)
                if not is_tm and (outer_step + 1) < self.config.inner_loop:
                    for _ in range(self.config.model_step_per_inner_step):
                        batch_learner = next(learner_train_loader)
                        batch_learner = batch_to_cuda(batch_learner["learner"])
                        with amp.autocast(enabled=self.use_amp, dtype=self.amp_dtype):
                            outputs = learner(**batch_to_cuda(batch_learner))
                            loss_learner = outputs.loss.mean()

                        learner_scaler.scale(loss_learner).backward()
                        self.train_step(
                            learner,
                            learner_optimizer,
                            learner_scheduler,
                            learner_scaler,
                        )

            average_outer_loop_train_logs = average(outer_loop_train_logs)
            train_logs.append(average_outer_loop_train_logs)

            if ((ol + 1) * self.config.inner_loop) % self.config.log_interval == 0:
                train_logs = average(train_logs)
                train_logs["train.lr"] = scheduler.get_last_lr()[0]

                wandb_utils.log_metrics(
                    train_logs, step=(ol + 1) * self.config.inner_loop
                )
                logger.info(
                    "TRAIN [{:>{}}/{}]: {}".format(
                        (ol + 1) * self.config.inner_loop,
                        len(str(self.config.total_train_step)),
                        self.config.total_train_step,
                        train_logs,
                    )
                )
                train_logs = []

        logger.info("Finish training!!")

        results = self.evaluate(
            generator,
            learner,
            evaluator,
            data_module,
            coreset_module,
            step=(ol + 1) * self.config.inner_loop,
        )

        if results[f"valid.{evaluator.metric_key}"] > best_val_score:
            best_val_score = results[f"valid.{evaluator.metric_key}"]
            generator.save_model(best_ckpt_path)
            logger.info(f"Save best checkpoint at `{best_ckpt_path}`")

        # save last checkpoint
        last_ckpt_path = os.path.join(self.config.save_model_dir, "last-ckpt")
        generator.save_model(last_ckpt_path)
        logger.info(f"Save last checkpoint at `{last_ckpt_path}`")

        # load best checkpoint
        generator.load_model(best_ckpt_path)

    def syn_data_to_batch(self, syn_data: Dataset, data_module: DataModule):
        syn_data = data_module.preprocess_dataset(syn_data)
        train_loader = DataLoader(
            syn_data,
            batch_size=len(syn_data),
            collate_fn=data_module.data_collator,
            shuffle=True,
            drop_last=True,
        )
        return next(iter(train_loader))

    def _make_loss_fn(self, learner, buffers, loss_weights, input_ids, kwargs):
        """Build a scalar loss as a function of params only (for func transforms)."""

        def loss_fn(params: dict[str, torch.Tensor]):
            outputs = torch.func.functional_call(
                learner, (params, buffers), args=input_ids, kwargs=kwargs
            )
            loss = outputs.loss
            if loss_weights is None:
                return loss.mean()
            assert loss.shape == loss_weights.shape
            return loss.dot(loss_weights)

        return loss_fn

    def _filter_and_flatten(
        self, learner: LearnerModel, grad_dict: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Optionally keep only classifier params, then flatten to a vector."""
        if self.config.classifier_grad_only:
            keep = set(learner.classifier_param_names())
            grad_dict = {n: g for n, g in grad_dict.items() if n in keep}
        return torch.concat([g.reshape(-1) for g in grad_dict.values()], dim=0)

    def compute_grad(
        self,
        learner: LearnerModel,
        params: dict[str, torch.Tensor],
        buffers: dict[str, torch.Tensor],
        loss_weights: torch.Tensor | None = None,
        input_ids=torch.LongTensor,
        **kwargs,
    ) -> torch.Tensor:
        """Return flatten gradient vector"""
        loss_fn = self._make_loss_fn(learner, buffers, loss_weights, input_ids, kwargs)

        # Full gradient matching double-backprops through attention; force the
        # math SDPA backend so the second-order derivative is available. The
        # classifier-only path never re-differentiates attention, so it keeps
        # the faster fused kernels.
        attention_ctx = (
            nullcontext() if self.config.classifier_grad_only else math_sdp_attention()
        )
        with attention_ctx:
            grad_dict = torch.func.grad(loss_fn)(params)

        return self._filter_and_flatten(learner, grad_dict)

    def sample_hvp_vectors(
        self, learner: LearnerModel, params: dict[str, torch.Tensor], num: int
    ) -> list[dict[str, torch.Tensor]]:
        """Random gaussian probe directions matching the gradient subspace.

        The same vector is reused for the real and synthetic Hessian-vector
        products so the cosine between them is meaningful.

        When ``classifier_grad_only`` is set, the probe is restricted to the
        classifier parameters (zero elsewhere) so the HVP equals the
        classifier-block Hessian times a classifier-space vector, ``H_cc v_c``
        -- i.e. the Hessian of only the last layer -- consistent with the
        gradient subspace being matched. The full dict structure is kept
        because ``jvp`` requires a tangent matching the full ``params`` tree.
        """
        keep = (
            set(learner.classifier_param_names())
            if self.config.classifier_grad_only
            else None
        )

        def make_vector() -> dict[str, torch.Tensor]:
            return {
                name: (
                    torch.randn_like(p)
                    if keep is None or name in keep
                    else torch.zeros_like(p)
                )
                for name, p in params.items()
            }

        return [make_vector() for _ in range(num)]

    def compute_grad_and_hvps(
        self,
        learner: LearnerModel,
        params: dict[str, torch.Tensor],
        buffers: dict[str, torch.Tensor],
        vectors: list[dict[str, torch.Tensor]],
        loss_weights: torch.Tensor | None = None,
        input_ids=torch.LongTensor,
        **kwargs,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Return (flat gradient, [flat Hessian-vector product for each vector]).

        HVPs use forward-over-reverse autodiff (``jvp`` of ``grad``), so the
        Hessian is never formed explicitly; the primal output of the jvp is the
        gradient itself, reused across vectors. The math SDPA backend is forced
        because this double-backprops through attention.
        """
        loss_fn = self._make_loss_fn(learner, buffers, loss_weights, input_ids, kwargs)

        def grad_fn(p):
            return torch.func.grad(loss_fn)(p)

        grad_flat: torch.Tensor | None = None
        hvp_flats: list[torch.Tensor] = []
        with math_sdp_attention():
            for vector in vectors:
                grad_dict, hvp_dict = torch.func.jvp(grad_fn, (params,), (vector,))
                if grad_flat is None:
                    grad_flat = self._filter_and_flatten(learner, grad_dict)
                hvp_flats.append(self._filter_and_flatten(learner, hvp_dict))

        return grad_flat, hvp_flats

    def compute_per_sample_grads(
        self,
        learner: LearnerModel,
        params: dict[str, torch.Tensor],
        buffers: dict[str, torch.Tensor],
        input_ids=torch.LongTensor,
        **kwargs,
    ) -> torch.Tensor:
        """Per-sample classifier-head gradients, shape (batch, d), detached.

        Uses vmap(grad) over the batch dimension, differentiating only the
        classifier head (body frozen). The learner is put in eval mode so
        dropout randomness does not break vmap. The result is detached: it does
        not depend on the generator, so OGM's only differentiable path is the
        per-sample weighting applied to these constants.
        """
        head_names = list(learner.classifier_param_names())
        head_set = set(head_names)
        body = {n: v.detach() for n, v in params.items() if n not in head_set}
        head = {n: params[n].detach() for n in head_names}

        kw_keys = list(kwargs.keys())

        def loss_single(head_dict, single_input_ids, *kw_vals):
            sample_kwargs = {k: v.unsqueeze(0) for k, v in zip(kw_keys, kw_vals)}
            outputs = torch.func.functional_call(
                learner,
                ({**body, **head_dict}, buffers),
                args=(single_input_ids.unsqueeze(0),),
                kwargs=sample_kwargs,
            )
            return outputs.loss.reshape(())

        grad_fn = torch.func.grad(loss_single)
        per_sample_grad_fn = torch.func.vmap(
            grad_fn, in_dims=(None, 0) + (0,) * len(kw_keys)
        )

        batch_size = input_ids.shape[0]
        was_training = learner.training
        learner.eval()
        try:
            per_sample = per_sample_grad_fn(
                head, input_ids, *[kwargs[k] for k in kw_keys]
            )
        finally:
            learner.train(was_training)

        return torch.cat(
            [per_sample[n].reshape(batch_size, -1) for n in head_names], dim=1
        ).detach()

    @staticmethod
    def _top_pcs(grads: torch.Tensor, num_pcs: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Top-k principal directions and variances of a (M, d) gradient matrix."""
        centered = grads - grads.mean(0, keepdim=True)
        _, singular_values, vh = torch.linalg.svd(centered, full_matrices=False)
        k = min(num_pcs, vh.shape[0])
        components = vh[:k].transpose(0, 1).contiguous()  # (d, k)
        denom = max(grads.shape[0] - 1, 1)
        variances = (singular_values[:k] ** 2) / denom  # (k,)
        return components, variances

    def _ogm_label(
        self,
        generator: GeneratorModel,
        learner: LearnerModel,
        params: dict[str, torch.Tensor],
        buffers: dict[str, torch.Tensor],
        gm_real_loaders: dict,
        gm_syn_loaders: dict,
        label: int,
        num_labels: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Orthogonal-GM loss for one class.

        Matches (a) the mean synthetic gradient to the mean real gradient
        (the GM term) and (b) the variance profile of the synthetic gradients
        along the real gradient's principal components to the real eigenvalues
        (the spread term). Only the per-sample weighting carries a gradient to
        the generator, so per-sample gradients are treated as constants.
        """
        # real per-sample gradients -> mean + principal components (detached target)
        grad_real_list = []
        for _ in range(self.config.gm_real_grad_accum_step):
            batch_gm_real = next(gm_real_loaders[label])
            grad_real_list.append(
                self.compute_per_sample_grads(
                    learner, params, buffers, **batch_to_cuda(batch_gm_real["learner"])
                )
            )
        grads_real = torch.cat(grad_real_list, dim=0)  # (M, d)
        grad_real_mean = grads_real.mean(0)
        components, real_var = self._top_pcs(grads_real, self.config.ogm_num_pcs)

        # synthetic side: per-sample grads (constant) + generation weights (differentiable)
        batch_gm_syn = next(gm_syn_loaders[label])
        gen_losses = generator.compute_loss(**batch_to_cuda(batch_gm_syn["generator"]))
        loss_weights = F.softmax(
            -gen_losses / self.config.normalize_temperature, dim=-1
        )
        grads_syn = self.compute_per_sample_grads(
            learner, params, buffers, **batch_to_cuda(batch_gm_syn["learner"])
        )  # (N, d), detached

        # differentiable weighted mean and weighted variance along each real PC
        grad_syn_mean = loss_weights @ grads_syn  # (d,)
        centered = grads_syn - grad_syn_mean  # (N, d)
        projections = centered @ components  # (N, k)
        syn_var = (loss_weights.unsqueeze(1) * projections.pow(2)).sum(0)  # (k,)

        grad_sim_label = F.cosine_similarity(grad_real_mean, grad_syn_mean, dim=0)
        ogm_sim_label = F.cosine_similarity(syn_var, real_var, dim=0)
        loss_dc_label = (
            (1 - grad_sim_label)
            + self.config.ogm_lambda * (1 - ogm_sim_label)
        ) / num_labels
        return loss_dc_label, grad_sim_label, ogm_sim_label

    def _load_tm_buffer(self, learner: LearnerModel) -> dict:
        """Load the expert trajectory buffer produced by src/buffer.py.

        Expected structure:
            {"head_param_names": [...],
             "snapshot_steps": [...],
             "experts": [[{name: tensor} per snapshot] per expert]}
        """
        path = self.config.tm_buffer_path
        assert os.path.exists(path), f"TM buffer not found: {path} (run src/buffer.py)"
        buffer = torch.load(path, map_location="cpu")

        head_names = list(learner.classifier_param_names())
        assert set(buffer["head_param_names"]) == set(head_names), (
            "TM buffer head params do not match the current learner: "
            f"{buffer['head_param_names']} vs {head_names}"
        )
        assert len(buffer["experts"]) >= 1
        n_snap = len(buffer["experts"][0])
        assert n_snap > self.config.tm_expert_snapshot_gap, (
            f"buffer has {n_snap} snapshots but tm_expert_snapshot_gap="
            f"{self.config.tm_expert_snapshot_gap}"
        )
        return buffer

    def _tm_inner_step(
        self,
        generator: GeneratorModel,
        learner: LearnerModel,
        params: dict[str, torch.Tensor],
        buffers: dict[str, torch.Tensor],
        gm_syn_loaders: dict,
        expert_buffer: dict,
        num_labels: int,
    ) -> torch.Tensor:
        """One trajectory-matching step (MTT) over the classifier head.

        Starts the student head at an expert snapshot, takes ``tm_syn_steps``
        gradient steps on the synthetic data (head only, body frozen), and
        returns the normalized parameter-space distance to a later expert
        snapshot. The synthetic per-sample loss is reweighted by ``loss_weights``
        (softmax over generator losses) so the distance is differentiable w.r.t.
        the generator. Body is frozen, so no gradient flows through attention.
        """
        head_names = list(learner.classifier_param_names())
        head_set = set(head_names)
        device = params[head_names[0]].device

        # frozen body (everything except the classifier head)
        body = {n: v.detach() for n, v in params.items() if n not in head_set}

        # sample an expert and a start snapshot
        expert = random.choice(expert_buffer["experts"])
        gap = self.config.tm_expert_snapshot_gap
        max_start = min(self.config.tm_max_start_snapshot, len(expert) - 1 - gap)
        start_idx = random.randint(0, max(max_start, 0))
        start_head = {n: expert[start_idx][n].to(device) for n in head_names}
        target_head = {n: expert[start_idx + gap][n].to(device) for n in head_names}

        def flatten(d: dict) -> torch.Tensor:
            return torch.cat([d[n].reshape(-1) for n in head_names])

        start_vec = flatten(start_head).detach()
        target_vec = flatten(target_head).detach()

        # draw the synthetic batches once; the generator is fixed across the N
        # student steps, so loss_weights are computed once and reused (their
        # graph to the generator is retained for the outer backward).
        syn_batches = []
        for label in range(num_labels):
            batch_syn = next(gm_syn_loaders[label])
            gen_losses = generator.compute_loss(
                **batch_to_cuda(batch_syn["generator"])
            )
            loss_weights = F.softmax(
                -gen_losses / self.config.normalize_temperature, dim=-1
            )
            syn_batches.append((batch_to_cuda(batch_syn["learner"]), loss_weights))

        # differentiable head rollout via unrolled SGD (create_graph=True)
        head = {n: start_head[n].clone().requires_grad_(True) for n in head_names}
        lr = self.config.tm_student_lr
        for _ in range(self.config.tm_syn_steps):
            total_loss = 0.0
            for learner_batch, loss_weights in syn_batches:
                full_params = {**body, **head}
                outputs = torch.func.functional_call(
                    learner,
                    (full_params, buffers),
                    args=(learner_batch["input_ids"],),
                    kwargs={
                        k: v for k, v in learner_batch.items() if k != "input_ids"
                    },
                )
                total_loss = total_loss + outputs.loss.dot(loss_weights)

            grads = torch.autograd.grad(
                total_loss, list(head.values()), create_graph=True
            )
            head = {
                name: head[name] - lr * g for name, g in zip(head_names, grads)
            }

        student_vec = flatten(head)
        numerator = ((student_vec - target_vec) ** 2).sum()
        denominator = ((start_vec - target_vec) ** 2).sum() + 1.0e-8
        return numerator / denominator

    def learner_optimizer(self, learner: LearnerModel, evaluate_config: EvaluateConfig):
        return configure_optimizer(
            learner,
            lr=evaluate_config.lr,
            optimizer_type=evaluate_config.optimizer_type,
            scheduler_type=evaluate_config.scheduler_type,
            weight_decay=evaluate_config.weight_decay,
            warmup_ratio=evaluate_config.warmup_ratio,
            num_train_steps=self.config.inner_loop
            * self.config.model_step_per_inner_step,
        )

    def get_gm_real_loaders(
        self,
        data_module: DataModule,
        learner: LearnerModel | None = None,
        repset_teachers: list[Dataset] | None = None,
    ) -> dict[int, Generator[dict[str, torch.Tensor], None, None]]:
        """Return dataloader of real data for gradient matching for each label"""
        num_labels = data_module.num_labels

        # setup data loader for gm loss
        if self.config.repset_teacher:
            # use repset teachers
            assert repset_teachers is not None
            assert self.config.gm_real_grad_accum_step <= self.config.n_repset
            concat_repset_teachers = concatenate_datasets(repset_teachers)
            repset_size = self.config.repset_dpc * num_labels
            assert len(concat_repset_teachers) == repset_size * self.config.n_repset

            return {
                label: endless_dataloader(
                    data_module.get_train_loader(
                        dataset=concat_repset_teachers,
                        batch_size=self.config.gm_real_dpc,
                        label=label,
                        shuffle=False,
                        drop_last=False,
                    ),
                    max_iteration=self.config.total_train_step
                    * self.config.gm_real_grad_accum_step,
                )
                for label in range(num_labels)
            }

        else:
            # use real data
            assert not self.config.repset_teacher
            if self.config.n_clusters_for_real_sampler > 1:
                return self.cluster_wise_dataloader(
                    data_module.preprocessed_datasets["train"],
                    data_module=data_module,
                    learner=learner,
                    dpc=self.config.gm_real_dpc,
                    n_clusters=self.config.n_clusters_for_real_sampler,
                    max_iteration=self.config.inner_loop
                    * self.config.generate_dataset_interval,
                )
            else:
                return {
                    label: endless_dataloader(
                        data_module.get_train_loader(
                            batch_size=self.config.gm_real_dpc,
                            label=label,
                        ),
                        max_iteration=self.config.total_train_step
                        * self.config.gm_real_grad_accum_step,
                    )
                    for label in range(num_labels)
                }

    def get_lm_loader(self, data_module: DataModule) -> Generator[dict, None, None]:
        """Return dataloader of real data for language modeling loss"""
        lm_loader = data_module.get_train_loader(batch_size=self.config.lm_batch_size)
        lm_loader = endless_dataloader(
            lm_loader, max_iteration=self.config.total_train_step
        )
        return lm_loader

    def get_learner_train_loader(
        self, data_module: DataModule
    ) -> Generator[dict, None, None]:
        """Return dataloader of real data for language modeling loss"""
        learner_train_step = (
            self.config.total_train_step * self.config.model_step_per_inner_step
        )
        learner_train_loader = data_module.get_train_loader()
        learner_train_loader = endless_dataloader(
            learner_train_loader, max_iteration=learner_train_step
        )
        return learner_train_loader

    def get_gm_syn_loaders(
        self, generator: GeneratorModel, learner: LearnerModel, data_module: DataModule
    ) -> dict[int, Generator[dict[str, torch.Tensor], None, None]]:
        """Return dataloader of synthetic data for gradient matching for each label"""

        # generate synthetic data
        syn_datasets = generator.generate_dataset(
            dpc=self.config.gm_syn_dpc,
            n=self.config.inner_loop * self.config.generate_dataset_interval,
        )

        # preprocess synthetic data
        preprocessed_syn_dataset = data_module.preprocess_dataset(
            concatenate_datasets(syn_datasets)
        )

        # setup dataloader
        if self.config.n_clusters_for_syn_sampler > 1:
            # return cluster-wise balanced dataloader
            return self.cluster_wise_dataloader(
                preprocessed_syn_dataset,
                data_module=data_module,
                learner=learner,
                dpc=self.config.gm_syn_dpc,
                n_clusters=self.config.n_clusters_for_syn_sampler,
                max_iteration=self.config.inner_loop
                * self.config.generate_dataset_interval,
            )

        # return random dataloader
        gm_syn_loaders = {
            label: iter(
                data_module.get_train_loader(
                    dataset=preprocessed_syn_dataset,
                    batch_size=self.config.gm_syn_dpc,
                    shuffle=False,
                    drop_last=False,
                    label=label,
                )
            )
            for label in range(data_module.num_labels)
        }

        return gm_syn_loaders
