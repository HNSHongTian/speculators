"""Frozen train/val loop from a66f57d for BV-disabled regression only."""
import time
import torch
import torch.distributed as dist
from tqdm.rich import tqdm
from speculators.train.trainer import _StepTimer, MIN_STEP_PCT, metric_logger
from speculators.train.utils import normalize_counted_metrics

class LegacyEpochs:
    def train_epoch(self, epoch: int):
        self.model.train()
        if hasattr(self.train_loader.batch_sampler, "set_epoch"):
            self.train_loader.batch_sampler.set_epoch(epoch)  # type: ignore[union-attr]

        # Capture full-epoch step count before any resume fast-skip mutation.
        num_steps = len(self.train_loader)

        # Determine how many batches to skip for mid-epoch resume.
        skip_steps = self._prepare_resume_skip(epoch)

        train_loader = self.train_loader
        if self.rank == 0:
            train_loader = tqdm(train_loader, desc=f"Epoch {epoch}")  # type: ignore[assignment]

        step_interval = (
            max(1, round(num_steps * self.config.checkpoint_freq))
            if self.config.checkpoint_freq < 1
            else None
        )
        t_before_fetch = time.perf_counter()
        timer = _StepTimer()
        for local_step_rel, batch in enumerate(train_loader, 1):
            # local_step is 1-based index into the *full* epoch (not the slice).
            local_step = local_step_rel + skip_steps
            timer.reset(self.global_step % self.config.log_freq == 0)

            timer.mark_value("start", t_before_fetch)
            gpu_batch = {
                k: v.to(self.local_rank, non_blocking=True)
                if isinstance(v, torch.Tensor)
                else v
                for k, v in batch.items()
            }

            with torch.autocast(
                self.device_type, dtype=self.config.hidden_states_dtype
            ):
                timer.mark("fetch")
                _draft_tokens, loss, metrics = self.model(
                    **gpu_batch, **(self.config.train_call_kwargs or {})
                )

            timer.mark("fwd")
            self._optimizers_zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            timer.mark("bwd")
            self._optimizers_step()

            current_lrs = {
                type(opt).__name__: opt.param_groups[0]["lr"] for opt in self.optimizers
            }
            self._schedulers_step()
            timer.mark("opt")
            t_before_fetch = timer.now() or time.perf_counter()

            profile = None
            if timer.enabled:
                num_tokens = int((gpu_batch["document_ids"] != -1).sum().item())
                profile = timer.profile(num_tokens)
                if self.is_distributed:
                    for v in metrics.values():
                        dist.reduce(v, dst=0, op=dist.ReduceOp.SUM)

                metrics = {k: v.item() for k, v in metrics.items()}
                world_size = dist.get_world_size() if self.is_distributed else 1
                metrics = normalize_counted_metrics(metrics, world_size)
                lr_info = (
                    current_lrs
                    if len(current_lrs) > 1
                    else next(iter(current_lrs.values()))
                )
                metric_logger.info(
                    {
                        "train": metrics,
                        "profile": profile,
                        "epoch": epoch,
                        "lr": lr_info,
                        "global_step": self.global_step,
                    },
                    extra={"step": self.global_step},
                )
            self.global_step += 1

            if (
                step_interval is not None
                and not self.config.save_best
                and local_step % step_interval == 0
                and num_steps - local_step >= step_interval * MIN_STEP_PCT
                # Avoid saving back to back ay the end of each epoch
            ):
                self.maybe_save_checkpoint(epoch, local_step=local_step)

    @torch.no_grad()
    def val_epoch(self, epoch: int) -> dict[str, float] | None:
        if self.val_loader is None:
            return None
        self.model.eval()
        if hasattr(self.val_loader.batch_sampler, "set_epoch"):
            self.val_loader.batch_sampler.set_epoch(epoch)  # type: ignore[union-attr]
        val_loader = self.val_loader
        if self.rank == 0:
            val_loader = tqdm(val_loader, desc=f"Epoch {epoch}")  # type: ignore[assignment]

        val_metrics: dict[str, float] = {}
        num_batches = len(val_loader)
        for batch in val_loader:
            gpu_batch = {
                k: v.to(self.local_rank, non_blocking=True)
                if isinstance(v, torch.Tensor)
                else v
                for k, v in batch.items()
            }

            with torch.autocast(
                self.device_type, dtype=self.config.hidden_states_dtype
            ):
                _draft_tokens, _loss, metrics = self.model(
                    **gpu_batch, **(self.config.val_call_kwargs or {})
                )

            if self.is_distributed:
                for m in metrics.values():
                    dist.all_reduce(m, op=dist.ReduceOp.SUM)

            for k, v in metrics.items():
                val_metrics[k] = val_metrics.get(k, 0.0) + v.item()

        world_size = dist.get_world_size() if self.is_distributed else 1
        val_metrics = {k: v / num_batches for k, v in val_metrics.items()}
        val_metrics = normalize_counted_metrics(val_metrics, world_size)
        val_metrics = {f"{k}_epoch": v for k, v in val_metrics.items()}

        metric_logger.info(
            {"val": val_metrics, "epoch": epoch}, extra={"step": self.global_step}
        )

        return val_metrics
