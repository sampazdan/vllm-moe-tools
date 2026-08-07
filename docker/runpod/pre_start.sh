#!/usr/bin/env bash
set -euo pipefail

mkdir -p \
    "${HF_HOME}" \
    "${HUGGINGFACE_HUB_CACHE}" \
    "${VLLM_CACHE_ROOT}" \
    "${TORCHINDUCTOR_CACHE_DIR}" \
    "${TRITON_CACHE_DIR}" \
    "${RUNPOD_RESULTS_DIR}" \
    /workspace/logs \
    /workspace/profiles

ln -sfn /opt/vllm-src /workspace/vllm-src

echo "vLLM MoE tools image: ${RUNPOD_VLLM_SOURCE_REF}"
/opt/vllm-venv/bin/python -c 'import torch, vllm; print(f"vLLM {vllm.__version__}; torch {torch.__version__}; CUDA {torch.version.cuda}")'
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
echo "Run 'runpod-serve' to start the API and 'runpod-bench' from another terminal."
