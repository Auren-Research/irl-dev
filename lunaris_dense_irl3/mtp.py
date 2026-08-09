"""Multi-token prediction heads supervised from the solved IRL-3 state."""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn

from lunaris_dense_irl2.irl2 import SwiGLU
from lunaris_dense_irl2.norms import RMSNorm

from .config import LunarisIRL3Config


class IRL3MultiTokenPredictor(nn.Module):
    """Parameter-efficient shared fusion for horizons two and beyond.

    The predictor deliberately does not contain a second recurrent computer.
    It asks the final IRL-3 representation to support farther-horizon token
    predictions without making the auxiliary objective define the core design.
    """

    def __init__(self, config: LunarisIRL3Config) -> None:
        super().__init__()
        if config.mtp_num_heads <= 0:
            raise ValueError("IRL3MultiTokenPredictor requires mtp_num_heads > 0")
        self.num_heads = config.mtp_num_heads
        hidden_dim = config.mtp_ffn_hidden_dim or config.d_model

        self.hidden_norm = RMSNorm(config.d_model, config.rms_norm_eps)
        self.token_norm = RMSNorm(config.d_model, config.rms_norm_eps)
        self.hidden_projection = nn.Linear(config.d_model, config.d_model, bias=False)
        self.token_projection = nn.Linear(config.d_model, config.d_model, bias=False)
        self.horizon_embeddings = nn.Parameter(
            torch.empty(self.num_heads, config.d_model)
        )
        self.fusion_norm = RMSNorm(config.d_model, config.rms_norm_eps)
        self.ffn = SwiGLU(config.d_model, hidden_dim)
        self.update_gate = nn.Linear(2 * config.d_model, config.d_model)
        self.output_norm = RMSNorm(config.d_model, config.rms_norm_eps)
        nn.init.normal_(
            self.horizon_embeddings,
            mean=0.0,
            std=float(config.initializer_range),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        token_embeddings: torch.Tensor,
        *,
        active_heads: int,
    ) -> List[torch.Tensor]:
        active_heads = min(max(0, int(active_heads)), self.num_heads)
        predictions: List[torch.Tensor] = []
        sequence_length = hidden_states.size(1)
        for horizon_index in range(active_heads):
            horizon = horizon_index + 2
            if sequence_length <= horizon:
                break
            hidden = self.hidden_projection(
                self.hidden_norm(hidden_states[:, :-horizon])
            )
            teacher = self.token_projection(
                self.token_norm(token_embeddings[:, horizon - 1 : -1])
            )
            horizon_embedding = self.horizon_embeddings[horizon_index].to(hidden.dtype)
            fused = self.fusion_norm(hidden + teacher + horizon_embedding)
            delta = self.ffn(fused)
            gate = torch.sigmoid(
                self.update_gate(torch.cat((fused, self.output_norm(delta)), dim=-1))
            )
            predictions.append(self.output_norm(fused + gate * delta))
        return predictions
