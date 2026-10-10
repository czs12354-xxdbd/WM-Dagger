#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "Usage: $0 SCRIPT_PATH CONFIG_PATH" >&2
    exit 2
fi

script_path=$1
config_path=$2

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    IFS=',' read -r -a visible_gpus <<< "$CUDA_VISIBLE_DEVICES"
    gpu_count=${#visible_gpus[@]}
else
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "nvidia-smi is required when CUDA_VISIBLE_DEVICES is unset" >&2
        exit 1
    fi
    gpu_count=$(nvidia-smi --list-gpus | wc -l | tr -d ' ')
fi

if (( gpu_count < 1 )); then
    echo "No GPUs available" >&2
    exit 1
fi

if [[ -z "${WORLD_SIZE:-}" ]]; then
    torchrun --nnodes=1 --nproc_per_node="$gpu_count" --node_rank=0 \
        "$script_path" --config_file "$config_path"
else
    : "${RANK:?RANK is required for multi-node execution}"
    : "${MASTER_ADDR:?MASTER_ADDR is required for multi-node execution}"
    : "${MASTER_PORT:?MASTER_PORT is required for multi-node execution}"
    torchrun --nnodes="$WORLD_SIZE" --nproc_per_node="$gpu_count" \
        --node_rank="$RANK" --master-addr="$MASTER_ADDR" \
        --master-port="$MASTER_PORT" "$script_path" --config_file "$config_path"
fi
