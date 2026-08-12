# RunPod development image

This image packages the current checkout, the Stage 2 expert-selection feature,
vLLM's benchmark extras, and small wrappers for repeatable RunPod experiments.
It uses the same RunPod CUDA 13.0 base and precompiled vLLM extension commit as
the Stage 2 RTX PRO 6000 hardware-acceptance run. The base is pinned to its
`linux/amd64` manifest digest so macOS Buildx does not select the image's
attestation manifest by mistake.

## Build and push

Build from the repository root. RunPod hosts are x86-64, so the platform flag is
required when building on Apple Silicon.

```bash
docker login
docker buildx build \
  --platform linux/amd64 \
  --file docker/Dockerfile.runpod \
  --build-arg VLLM_SOURCE_REF="$(git rev-parse --short HEAD)-stage2-worktree" \
  --tag YOUR_DOCKER_USER/vllm-moe-tools:stage2-v1 \
  --push \
  .
```

Use a new immutable tag for each build. The Docker context excludes `.git`, so
the image contains the current tracked and uncommitted source files without
including repository history.

The default build reuses CUDA 13.0 precompiled extensions from upstream commit
`b1e12d142d8c9533f857f8da13d8fc368e95a8cd`, the upstream merge base used by
the acceptance build. This is appropriate for the Python-only Stage 1 and
Stage 2 changes. Rebuild vLLM normally instead if a future experiment changes
C++, CUDA, or extension interfaces.

## RunPod template

Recommended starting configuration:

- Image: `YOUR_DOCKER_USER/vllm-moe-tools:stage2-v1`
- GPU: one RTX PRO 6000 Blackwell Server Edition, or another GPU with at least
  enough memory for the selected model
- HTTP port: `8000`
- TCP port: `22`
- Network volume: at least 100 GB mounted at `/workspace`
- Environment: `HF_TOKEN` if the model requires Hugging Face authentication
- Container start command: leave blank so the inherited `/start.sh` enables
  SSH, the web terminal, and Jupyter

Model downloads, compilation caches, profiles, logs, and benchmark results are
written under `/workspace` and therefore survive pod restarts when a network
volume is attached. The source and virtual environment are baked under `/opt`
to avoid slow small-file access on the network volume.

For a gated model launched from an SSH session, export `HF_TOKEN` in that
session or prefix it on `runpod-serve`; RunPod template environment variables
are attached to the container's main process and are not always propagated to
SSH login shells.

## Start the validated model

From the pod terminal, first create a profile. Qwen3.6-35B-A3B-FP8 has 40 MoE
layers and 256 logical experts per layer:

```bash
make-moe-profile \
  --layers 40 \
  --experts 256 \
  --keep 128 \
  --output /workspace/profiles/qwen-keep-128.json
```

Start the OpenAI-compatible server with pruning:

```bash
MOE_PROFILE=/workspace/profiles/qwen-keep-128.json runpod-serve
```

For a baseline, omit `MOE_PROFILE`:

```bash
runpod-serve
```

The defaults reproduce the acceptance configuration: model
`Qwen/Qwen3.6-35B-A3B-FP8`, max model length 4096, 0.90 GPU-memory utilization,
2048 max batched tokens, 8 max sequences, compilation and CUDA graphs enabled,
and DeepGemm disabled on Blackwell. Override a setting with environment
variables or append ordinary `vllm serve` arguments:

```bash
RUNPOD_VLLM_MAX_NUM_SEQS=32 runpod-serve --disable-log-stats
```

Pin production loads to an immutable Hugging Face commit with
`RUNPOD_VLLM_REVISION`. It must be exactly 40 lowercase hexadecimal characters.
The accepted Qwen revision is:

```bash
RUNPOD_VLLM_REVISION=95a723d08a9490559dae23d0cff1d9466213d989 \
  runpod-serve
```

The wrapper binds both `--revision` and `--tokenizer-revision` to this value,
then preflights the pinned config and fast tokenizer into the Hugging Face cache
before vLLM starts. It records the revision in
`/workspace/logs/vllm-server-config.txt`.

Set `RUNPOD_CAPTURE_ROUTING=1` when routed-expert IDs and weights are needed.
Leave it disabled for clean performance measurements.

## Benchmark

Keep the server running and open a second terminal:

```bash
RUNPOD_BENCH_LABEL=keep-128 runpod-bench
```

The wrapper runs `vllm bench serve` with a synthetic random workload and stores
the result JSON, complete log, model response, vLLM/source metadata, and
`nvidia-smi` snapshot in a timestamped directory under `/workspace/results`.
Defaults are 128 prompts with 512 input and 128 output tokens at unlimited
request rate. They can be changed without editing the image:

```bash
RUNPOD_BENCH_INPUT_LEN=1024 \
RUNPOD_BENCH_OUTPUT_LEN=256 \
RUNPOD_BENCH_NUM_PROMPTS=512 \
RUNPOD_BENCH_REQUEST_RATE=8 \
RUNPOD_BENCH_LABEL=keep-128-rps8 \
runpod-bench --max-concurrency 32
```

To compare profiles, stop the server, restart it with the next `MOE_PROFILE`,
wait for `/health`, and run the same benchmark settings with a different label.
