#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export CLEARML_CONFIG_FILE="${CLEARML_CONFIG_FILE:-/storage/homes/dterbe/clearml.conf}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

exec uv run feldvision-train --config configs/context_separate_clearml_cuda0.yaml
