from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict
from torch.distributed.checkpoint.stateful import Stateful

logger = logging.getLogger(__name__)


class TrainingState(Stateful):
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
        step: int = 0,
        tokens_seen: int = 0,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.step = step
        self.tokens_seen = tokens_seen

    def state_dict(self) -> dict:
        model_sd, optim_sd = get_state_dict(self.model, self.optimizer)
        state = {
            "model": model_sd,
            "optimizer": optim_sd,
            "step": self.step,
            "tokens_seen": self.tokens_seen,
        }
        if self.scheduler is not None:
            state["scheduler"] = self.scheduler.state_dict()
        return state

    def load_state_dict(self, state_dict: dict) -> None:
        set_state_dict(
            self.model,
            self.optimizer,
            model_state_dict=state_dict["model"],
            optim_state_dict=state_dict["optimizer"],
        )
        self.step = state_dict["step"]
        self.tokens_seen = state_dict["tokens_seen"]
        if self.scheduler is not None and "scheduler" in state_dict:
            self.scheduler.load_state_dict(state_dict["scheduler"])


def save_checkpoint(state: TrainingState, checkpoint_dir: str) -> None:
    dcp.save({"training": state}, checkpoint_id=checkpoint_dir)
    logger.info("Saved checkpoint to %s (step %d)", checkpoint_dir, state.step)


def load_checkpoint(state: TrainingState, checkpoint_dir: str) -> None:
    dcp.load({"training": state}, checkpoint_id=checkpoint_dir)
    logger.info(
        "Loaded checkpoint from %s (step %d, tokens_seen %d)",
        checkpoint_dir,
        state.step,
        state.tokens_seen,
    )
