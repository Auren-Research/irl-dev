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

download_file() {
  local url="$1"
  local output="$2"

  if command -v curl >/dev/null 2>&1; then
    curl \
      --fail \
      --location \
      --continue-at - \
      --retry 5 \
      --retry-all-errors \
      --output "${output}" \
      "${url}"
    return
  fi

  if command -v wget >/dev/null 2>&1; then
    wget \
      --continue \
      --tries=5 \
      --retry-connrefused \
      --timeout=60 \
      --output-document="${output}" \
      "${url}"
    return
  fi

  echo "[download] neither curl nor wget is available" >&2
  return 127
}

if [[ ! -f "${shard_path}" ]]; then
  echo "[download] fetching one 1B-token shard"
  download_file \
    "${repository_url}/shard_0000.npy?download=true" \
    "${shard_partial}"
  mv "${shard_partial}" "${shard_path}"
else
  echo "[download] reusing ${shard_path}"
fi

if [[ ! -f "${tokenizer_path}" ]]; then
  echo "[download] fetching tokenizer"
  download_file \
    "${repository_url}/lunaris-tokenizer.json?download=true" \
    "${tokenizer_partial}"
  mv "${tokenizer_partial}" "${tokenizer_path}"
else
  echo "[download] reusing ${tokenizer_path}"
fi

ls -lh "${shard_path}" "${tokenizer_path}"
