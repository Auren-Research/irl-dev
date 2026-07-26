"""Run a small end-to-end training and generation smoke test."""

from __future__ import annotations

import torch

from lunaris_dense_irl2 import LunarisDenseConfig, LunarisDenseForCausalLM


def main() -> None:
    """Exercise loss, gradients, diagnostics, cache, and generation."""

    torch.manual_seed(7)
    config = LunarisDenseConfig.debug(
        vocab_size=257,
        d_model=64,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        ffn_hidden_dim=128,
        max_seq_len=64,
        local_window=8,
    )
    model = LunarisDenseForCausalLM(config)

    input_ids = torch.randint(0, config.vocab_size, (2, 12))
    model.train()
    output = model(input_ids, labels=input_ids, return_irl_stats=True)
    if output.loss is None:
        raise RuntimeError("Training loss was not produced")
    output.loss.backward()

    model.eval()
    generated = model.generate(
        input_ids[:, :4],
        max_new_tokens=4,
        temperature=0.0,
        n_iters=2,
    )

    print(f"parameters={model.get_num_parameters():,}")
    print(f"logits={tuple(output.logits.shape)}")
    print(f"loss={float(output.loss.detach()):.6f}")
    print(f"generated={tuple(generated.shape)}")
    print("smoke_test=ok")


if __name__ == "__main__":
    main()
