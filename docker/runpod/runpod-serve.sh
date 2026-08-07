#!/usr/bin/env bash
set -euo pipefail

export HF_HOME="${HF_HOME:-/workspace/cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-/workspace/cache/vllm}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/workspace/cache/torchinductor}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/workspace/cache/triton}"

model="${RUNPOD_VLLM_MODEL:-Qwen/Qwen3.6-35B-A3B-FP8}"
host="${RUNPOD_VLLM_HOST:-0.0.0.0}"
port="${RUNPOD_VLLM_PORT:-8000}"
max_model_len="${RUNPOD_VLLM_MAX_MODEL_LEN:-4096}"
gpu_memory_utilization="${RUNPOD_VLLM_GPU_MEMORY_UTILIZATION:-0.90}"
max_num_batched_tokens="${RUNPOD_VLLM_MAX_NUM_BATCHED_TOKENS:-2048}"
max_num_seqs="${RUNPOD_VLLM_MAX_NUM_SEQS:-8}"

args=(
    serve "${model}"
    --host "${host}"
    --port "${port}"
    --max-model-len "${max_model_len}"
    --gpu-memory-utilization "${gpu_memory_utilization}"
    --max-num-batched-tokens "${max_num_batched_tokens}"
    --max-num-seqs "${max_num_seqs}"
)

if [[ -n "${MOE_PROFILE:-}" ]]; then
    if [[ ! -f "${MOE_PROFILE}" ]]; then
        echo "Expert selection profile does not exist: ${MOE_PROFILE}" >&2
        exit 2
    fi
    args+=(--moe-expert-selection-profile "${MOE_PROFILE}")
fi

if [[ "${RUNPOD_CAPTURE_ROUTING:-0}" == "1" ]]; then
    args+=(
        --enable-return-routed-experts
        --enable-return-routed-expert-weights
    )
fi

mkdir -p /workspace/logs
{
    echo "source_ref=${RUNPOD_VLLM_SOURCE_REF:-unknown}"
    echo "model=${model}"
    echo "profile=${MOE_PROFILE:-baseline}"
    echo "capture_routing=${RUNPOD_CAPTURE_ROUTING:-0}"
    echo "command=vllm ${args[*]} $*"
} >/workspace/logs/vllm-server-config.txt

echo "Starting: vllm ${args[*]} $*"
exec /opt/vllm-venv/bin/vllm "${args[@]}" "$@"
