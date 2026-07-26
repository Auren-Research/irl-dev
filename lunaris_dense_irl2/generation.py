"""Cache-aware autoregressive generation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

if TYPE_CHECKING:
    from .model import LunarisDenseForCausalLM


def _sample_next_token(
    logits: torch.Tensor,
    *,
    temperature: float,
    top_k: Optional[int],
    top_p: Optional[float],
    generator: Optional[torch.Generator],
) -> torch.Tensor:
    """Sample one token with optional top-k and nucleus filtering."""

    if temperature <= 0.0:
        return logits.argmax(dim=-1, keepdim=True)

    filtered = logits.float() / temperature
    if top_k is not None:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        top_k = min(int(top_k), filtered.size(-1))
        threshold = torch.topk(filtered, top_k, dim=-1).values[:, -1:]
        filtered = filtered.masked_fill(filtered < threshold, float("-inf"))

    if top_p is not None:
        if not (0.0 < top_p <= 1.0):
            raise ValueError("top_p must be in (0, 1]")
        sorted_logits, sorted_indices = torch.sort(filtered, descending=True, dim=-1)
        sorted_probabilities = torch.softmax(sorted_logits, dim=-1)
        cumulative = sorted_probabilities.cumsum(dim=-1)
        remove = cumulative > top_p
        remove[:, 1:] = remove[:, :-1].clone()
        remove[:, 0] = False
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        filtered = torch.full_like(filtered, float("-inf"))
        filtered.scatter_(1, sorted_indices, sorted_logits)

    probabilities = torch.softmax(filtered, dim=-1)
    return torch.multinomial(probabilities, num_samples=1, generator=generator)


@torch.no_grad()
def generate(
    model: "LunarisDenseForCausalLM",
    input_ids: torch.Tensor,
    *,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    eos_token_id: Optional[int] = None,
    n_iters: Optional[int] = None,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Generate from equally sized, unpadded prompts."""

    if input_ids.dim() != 2:
        raise ValueError("input_ids must have shape [batch, sequence]")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens cannot be negative")
    if input_ids.size(1) + max_new_tokens > model.config.max_seq_len:
        raise ValueError("Requested generation exceeds max_seq_len")
    if max_new_tokens == 0:
        return input_ids

    previous_training_state = model.training
    model.eval()

    generated = input_ids
    cache = model.init_cache(batch_size=input_ids.size(0))
    output = model(
        input_ids,
        cache=cache,
        n_iters=n_iters,
        logits_to_keep=1,
    )
    next_logits = output.logits[:, -1]

    stop_token = model.config.eos_token_id if eos_token_id is None else eos_token_id
    finished = torch.zeros(input_ids.size(0), device=input_ids.device, dtype=torch.bool)

    for step in range(max_new_tokens):
        next_token = _sample_next_token(
            next_logits,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            generator=generator,
        )
        if stop_token is not None:
            forced_stop = torch.full_like(next_token, int(stop_token))
            next_token = torch.where(finished.unsqueeze(-1), forced_stop, next_token)
            finished = finished | next_token.squeeze(-1).eq(int(stop_token))

        generated = torch.cat((generated, next_token), dim=-1)
        if bool(finished.all()) or step + 1 == max_new_tokens:
            break

        output = model(
            next_token,
            cache=cache,
            n_iters=n_iters,
            logits_to_keep=1,
        )
        next_logits = output.logits[:, -1]

    if previous_training_state:
        model.train()
    return generated
