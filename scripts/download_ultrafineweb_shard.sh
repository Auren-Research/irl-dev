#!/usr/bin/env bash
set -euo pipefail

# Download exactly one public Ultra-FineWeb token shard and its tokenizer.
repository_url="https://huggingface.co/datasets/meryyllebr543/lunaris-ultrafineweb-20b-tokenized/resolve/main"
destination="${1:-data/source/ultrafineweb}"
shard_path="${destination}/shard_0000.npy"
tokenizer_path="${destination}/lunaris-tokenizer.json"
shard_partial="${shard_path}.partial"
tokenizer_partial="${tokenizer_path}.partial"

mkdir -p "${destination}"

if [[ ! -f "${shard_path}" ]]; then
  echo "[download] fetching one 1B-token shard"
  curl \
    --fail \
    --location \
    --continue-at - \
    --retry 5 \
    --retry-all-errors \
    --output "${shard_partial}" \
    "${repository_url}/shard_0000.npy?download=true"
  mv "${shard_partial}" "${shard_path}"
else
  echo "[download] reusing ${shard_path}"
fi

if [[ ! -f "${tokenizer_path}" ]]; then
  echo "[download] fetching tokenizer"
  curl \
    --fail \
    --location \
    --retry 5 \
    --retry-all-errors \
    --output "${tokenizer_partial}" \
    "${repository_url}/lunaris-tokenizer.json?download=true"
  mv "${tokenizer_partial}" "${tokenizer_path}"
else
  echo "[download] reusing ${tokenizer_path}"
fi

ls -lh "${shard_path}" "${tokenizer_path}"
