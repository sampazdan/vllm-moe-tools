#!/usr/bin/env bash

export PATH="/opt/vllm-venv/bin:${PATH}"
export VIRTUAL_ENV="/opt/vllm-venv"
export HF_HOME="${HF_HOME:-/workspace/cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-/workspace/cache/vllm}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/workspace/cache/torchinductor}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/workspace/cache/triton}"
export RUNPOD_RESULTS_DIR="${RUNPOD_RESULTS_DIR:-/workspace/results}"
