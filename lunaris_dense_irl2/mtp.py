"""Parameter-efficient multi-token prediction coupled to IRL-2."""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from .config import LunarisDenseConfig
from .irl2 import IRL2Cell
from .norms import RMSNorm


class IRLMultiTokenPredictor(nn.Module):
    """Predict several future horizons with one shared IRL transition.

    Horizon ``k`` combines the backbone state at position ``t`` with the
    teacher-forced embedding at ``t + k - 1``. All horizons share the fusion
    projections and recurrent cell; only a small learned horizon embedding is
    horizon-specific. The regular LM head is reused by the parent model.
    """

    def __init__(self, config: LunarisDenseConfig) -> None:
        super().__init__()
        if config.mtp_num_heads <= 0:
            raise ValueError("IRLMultiTokenPredictor requires mtp_num_heads > 0")

        self.num_heads = int(config.mtp_num_heads)
        self.irl_iters = int(config.mtp_irl_iters)
        self.use_gradient_checkpointing = bool(config.use_gradient_checkpointing)

        self.hidden_norm = RMSNorm(config.d_model, config.rms_norm_eps)
        self.token_norm = RMSNorm(config.d_model, config.rms_norm_eps)
        self.hidden_projection = nn.Linear(config.d_model, config.d_model, bias=False)
        self.token_projection = nn.Linear(config.d_model, config.d_model, bias=False)
        self.token_gate = nn.Parameter(torch.zeros(config.d_model))
        self.horizon_embeddings = nn.Parameter(
            torch.empty(self.num_heads, config.d_model)
        )
        self.fusion_norm = RMSNorm(config.d_model, config.rms_norm_eps)
        self.output_norm = RMSNorm(config.d_model, config.rms_norm_eps)

        mtp_hidden_dim = config.mtp_ffn_hidden_dim or config.d_model
        self.irl = IRL2Cell(config, hidden_dim=mtp_hidden_dim)
        nn.init.normal_(
            self.horizon_embeddings,
            mean=0.0,
            std=float(config.initializer_range),
        )

    def _predict_one(
        self,
        hidden_states: torch.Tensor,
        previous_token_embeddings: torch.Tensor,
        horizon_index: int,
        *,
        return_consistency: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fuse one horizon and run the shared recurrent transition."""

        hidden = self.hidden_projection(self.hidden_norm(hidden_states))
        token = self.token_projection(self.token_norm(previous_token_embeddings))
        token = token * torch.sigmoid(self.token_gate).to(token.dtype)
        horizon = self.horizon_embeddings[horizon_index].to(hidden.dtype)
        anchor = self.fusion_norm(hidden + token + horizon)
        correction, _, consistency = self.irl(
            anchor,
            n_iters=self.irl_iters,
            return_stats=False,
            return_consistency=return_consistency,
        )
        return self.output_norm(anchor + correction), consistency

    def forward(
        self,
        hidden_states: torch.Tensor,
        token_embeddings: torch.Tensor,
        *,
        active_heads: int,
        return_consistency: bool,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Return aligned prediction states for horizons 2 through H + 1."""

        active_heads = min(max(0, int(active_heads)), self.num_heads)
        prediction_states: List[torch.Tensor] = []
        consistency_losses: List[torch.Tensor] = []
        sequence_length = hidden_states.size(1)

        for horizon_index in range(active_heads):
            horizon = horizon_index + 2
            if sequence_length <= horizon:
                break
            base_hidden = hidden_states[:, :-horizon]
            previous_tokens = token_embeddings[:, horizon - 1 : -1]

            should_checkpoint = self.training and self.use_gradient_checkpointing
            if should_checkpoint:

                def checkpointed_prediction(
                    base: torch.Tensor,
                    teacher: torch.Tensor,
                    index: int = horizon_index,
                ) -> Tuple[torch.Tensor, torch.Tensor]:
                    return self._predict_one(
                        base,
                        teacher,
                        index,
                        return_consistency=return_consistency,
                    )

                prediction, consistency = checkpoint(
                    checkpointed_prediction,
                    base_hidden,
                    previous_tokens,
                    use_reentrant=False,
                )
            else:
                prediction, consistency = self._predict_one(
                    base_hidden,
                    previous_tokens,
                    horizon_index,
                    return_consistency=return_consistency,
                )

            prediction_states.append(prediction)
            consistency_losses.append(consistency)

        return prediction_states, consistency_losses
