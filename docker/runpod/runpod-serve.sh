#!/usr/bin/env bash
set -euo pipefail

export HF_HOME="${HF_HOME:-/workspace/cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-/workspace/cache/vllm}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/workspace/cache/torchinductor}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/workspace/cache/triton}"

model="${RUNPOD_VLLM_MODEL:-Qwen/Qwen3.6-35B-A3B-FP8}"
revision="${RUNPOD_VLLM_REVISION:-}"
host="${RUNPOD_VLLM_HOST:-0.0.0.0}"
port="${RUNPOD_VLLM_PORT:-8000}"
max_model_len="${RUNPOD_VLLM_MAX_MODEL_LEN:-4096}"
gpu_memory_utilization="${RUNPOD_VLLM_GPU_MEMORY_UTILIZATION:-0.90}"
max_num_batched_tokens="${RUNPOD_VLLM_MAX_NUM_BATCHED_TOKENS:-2048}"
max_num_seqs="${RUNPOD_VLLM_MAX_NUM_SEQS:-8}"
enable_request_metrics="${RUNPOD_ENABLE_REQUEST_METRICS:-1}"
reasoning_parser="${RUNPOD_REASONING_PARSER:-qwen3}"
preflight_python="${RUNPOD_VLLM_PREFLIGHT_PYTHON:-/opt/vllm-venv/bin/python}"
preflight_timeout="${RUNPOD_VLLM_PREFLIGHT_TIMEOUT_SECONDS:-300}"

require_integer() {
    local name="$1"
    local value="$2"
    local minimum="$3"
    local maximum="$4"
    if [[ ! "${value}" =~ ^[0-9]{1,8}$ ]]; then
        echo "${name} must be an integer from ${minimum} to ${maximum}." >&2
        exit 2
    fi
    local normalized=$((10#${value}))
    if (( normalized < minimum || normalized > maximum )); then
        echo "${name} must be an integer from ${minimum} to ${maximum}." >&2
        exit 2
    fi
}

if [[ -z "${model}" || -z "${host}" ]]; then
    echo "RUNPOD_VLLM_MODEL and RUNPOD_VLLM_HOST cannot be empty." >&2
    exit 2
fi
if [[ -n "${revision}" && ! "${revision}" =~ ^[0-9a-f]{40}$ ]]; then
    echo "RUNPOD_VLLM_REVISION must be a 40-character lowercase hex commit." >&2
    exit 2
fi
require_integer RUNPOD_VLLM_PORT "${port}" 1 65535
require_integer RUNPOD_VLLM_MAX_MODEL_LEN "${max_model_len}" 1 10000000
require_integer \
    RUNPOD_VLLM_MAX_NUM_BATCHED_TOKENS \
    "${max_num_batched_tokens}" \
    1 \
    10000000
require_integer RUNPOD_VLLM_MAX_NUM_SEQS "${max_num_seqs}" 1 4096
require_integer \
    RUNPOD_VLLM_PREFLIGHT_TIMEOUT_SECONDS \
    "${preflight_timeout}" \
    30 \
    900
if [[ ! "${gpu_memory_utilization}" =~ ^(0\.[0-9]*[1-9][0-9]*|1(\.0+)?)$ ]]; then
    echo "RUNPOD_VLLM_GPU_MEMORY_UTILIZATION must be greater than 0 and at most 1." >&2
    exit 2
fi
if [[ "${enable_request_metrics}" != "0" && "${enable_request_metrics}" != "1" ]]; then
    echo "RUNPOD_ENABLE_REQUEST_METRICS must be 0 or 1." >&2
    exit 2
fi
if [[ "${RUNPOD_CAPTURE_ROUTING:-0}" != "0" \
    && "${RUNPOD_CAPTURE_ROUTING:-0}" != "1" ]]; then
    echo "RUNPOD_CAPTURE_ROUTING must be 0 or 1." >&2
    exit 2
fi
if [[ -n "${revision}" && ! -x "${preflight_python}" ]]; then
    echo "Tokenizer preflight Python is not executable: ${preflight_python}" >&2
    exit 2
fi

args=(
    serve "${model}"
    --host "${host}"
    --port "${port}"
    --max-model-len "${max_model_len}"
    --gpu-memory-utilization "${gpu_memory_utilization}"
    --max-num-batched-tokens "${max_num_batched_tokens}"
    --max-num-seqs "${max_num_seqs}"
)

if [[ -n "${revision}" ]]; then
    args+=(
        --revision "${revision}"
        --tokenizer-revision "${revision}"
    )
fi

if [[ "${enable_request_metrics}" == "1" ]]; then
    args+=(--enable-per-request-metrics)
fi

if [[ -n "${reasoning_parser}" ]]; then
    args+=(--reasoning-parser "${reasoning_parser}")
fi

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

if [[ -n "${revision}" ]]; then
    timeout \
        --signal=TERM \
        --kill-after=10s \
        "${preflight_timeout}s" \
        "${preflight_python}" - "${model}" "${revision}" <<'PY'
import sys

from transformers import AutoConfig, AutoTokenizer

model, revision = sys.argv[1:]
config = AutoConfig.from_pretrained(model, revision=revision)
if not getattr(config, "model_type", None):
    raise RuntimeError("the pinned model config has no model_type")
tokenizer = AutoTokenizer.from_pretrained(
    model,
    revision=revision,
    config=config,
)
if not tokenizer.is_fast:
    raise RuntimeError("the pinned model did not resolve a fast tokenizer")
tokenizer.encode("tokenizer preflight", add_special_tokens=False)
print(f"Validated pinned model and tokenizer metadata for {model}@{revision}.")
PY
fi

mkdir -p /workspace/logs
{
    echo "source_ref=${RUNPOD_VLLM_SOURCE_REF:-unknown}"
    echo "model=${model}"
    echo "revision=${revision:-unpinned}"
    echo "profile=${MOE_PROFILE:-baseline}"
    echo "capture_routing=${RUNPOD_CAPTURE_ROUTING:-0}"
    echo "request_metrics=${enable_request_metrics}"
    echo "reasoning_parser=${reasoning_parser:-disabled}"
    echo "extra_args_count=$#"
} >/workspace/logs/vllm-server-config.txt

echo "Starting managed vLLM for ${model} on ${host}:${port}."
exec /opt/vllm-venv/bin/vllm "${args[@]}" "$@"
