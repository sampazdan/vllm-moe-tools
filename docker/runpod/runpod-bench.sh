#!/usr/bin/env bash
set -euo pipefail

base_url="${RUNPOD_BENCH_BASE_URL:-http://127.0.0.1:8000}"
model="${RUNPOD_BENCH_MODEL:-${RUNPOD_VLLM_MODEL:-Qwen/Qwen3.6-35B-A3B-FP8}}"
input_len="${RUNPOD_BENCH_INPUT_LEN:-512}"
output_len="${RUNPOD_BENCH_OUTPUT_LEN:-128}"
num_prompts="${RUNPOD_BENCH_NUM_PROMPTS:-128}"
request_rate="${RUNPOD_BENCH_REQUEST_RATE:-inf}"
label="${RUNPOD_BENCH_LABEL:-manual}"
timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
safe_label="$(printf '%s' "${label}" | tr -c 'A-Za-z0-9_.-' '_')"
run_dir="${RUNPOD_RESULTS_DIR:-/workspace/results}/${timestamp}-${safe_label}"

mkdir -p "${run_dir}"

{
    echo "timestamp=${timestamp}"
    echo "source_ref=${RUNPOD_VLLM_SOURCE_REF:-unknown}"
    echo "model=${model}"
    echo "profile=${MOE_PROFILE:-baseline}"
    echo "base_url=${base_url}"
    /opt/vllm-venv/bin/python -VV
    /opt/vllm-venv/bin/vllm --version
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
} >"${run_dir}/environment.txt"

curl --fail --silent --show-error "${base_url}/health" >/dev/null
curl --fail --silent --show-error "${base_url}/v1/models" \
    >"${run_dir}/models.json"
nvidia-smi -q >"${run_dir}/nvidia-smi.txt"
if [[ -f /workspace/logs/vllm-server-config.txt ]]; then
    cp /workspace/logs/vllm-server-config.txt "${run_dir}/server-config.txt"
fi

echo "Writing benchmark artifacts to ${run_dir}"
/opt/vllm-venv/bin/vllm bench serve \
    --backend openai \
    --base-url "${base_url}" \
    --endpoint /v1/completions \
    --model "${model}" \
    --dataset-name random \
    --input-len "${input_len}" \
    --output-len "${output_len}" \
    --num-prompts "${num_prompts}" \
    --request-rate "${request_rate}" \
    --save-result \
    --result-dir "${run_dir}" \
    --result-filename result.json \
    "$@" 2>&1 | tee "${run_dir}/benchmark.log"
