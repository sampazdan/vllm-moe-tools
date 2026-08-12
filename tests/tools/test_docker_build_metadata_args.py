# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import shlex
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
HELPER = REPO_ROOT / ".buildkite" / "scripts" / "docker-build-metadata-args.sh"


def run_helper(
    *args: str,
    env: dict[str, str] | None = None,
    path: str | None = None,
) -> list[str]:
    helper_env = {"PATH": path or os.environ["PATH"]}
    if env:
        helper_env.update(env)
    result = subprocess.run(
        ["bash", str(HELPER), *args],
        check=True,
        env=helper_env,
        stdout=subprocess.PIPE,
        text=True,
    )
    return shlex.split(result.stdout)


def option_values(args: list[str], option: str) -> list[str]:
    return [args[i + 1] for i, arg in enumerate(args[:-1]) if arg == option]


def build_args(args: list[str]) -> dict[str, str]:
    values = {}
    for value in option_values(args, "--build-arg"):
        key, arg_value = value.split("=", 1)
        values[key] = arg_value
    return values


def test_release_metadata_args_prefer_pipeline_id() -> None:
    args = run_helper(
        "cu130-ubuntu2404",
        env={
            "BUILDKITE": "1",
            "BUILDKITE_COMMIT": "abc123",
            "BUILDKITE_PIPELINE_ID": "pipe-uuid",
            "BUILDKITE_PIPELINE_SLUG": "release",
            "BUILDKITE_BUILD_URL": "https://buildkite.example/vllm/builds/1",
            "RELEASE_VERSION": "v0.20.0",
        },
    )

    assert build_args(args) == {
        "VLLM_BUILD_COMMIT": "abc123",
        "VLLM_BUILD_PIPELINE": "pipe-uuid",
        "VLLM_BUILD_URL": "https://buildkite.example/vllm/builds/1",
        "VLLM_IMAGE_TAG": "vllm/vllm-openai:v0.20.0-cu130-ubuntu2404",
    }
    expected_tag = (
        "public.ecr.aws/q9t5s3a7/vllm-release-repo:"
        f"abc123-{os.uname().machine}-cu130-ubuntu2404"
    )
    assert option_values(args, "--tag") == [expected_tag]


def test_nightly_metadata_args_fall_back_to_pipeline_slug() -> None:
    args = run_helper(
        "ubuntu2404",
        env={
            "BUILDKITE": "1",
            "BUILDKITE_COMMIT": "def456",
            "BUILDKITE_PIPELINE_SLUG": "release",
            "BUILDKITE_BUILD_URL": "https://buildkite.example/vllm/builds/2",
            "NIGHTLY": "1",
        },
    )

    assert build_args(args) == {
        "VLLM_BUILD_COMMIT": "def456",
        "VLLM_BUILD_PIPELINE": "release",
        "VLLM_BUILD_URL": "https://buildkite.example/vllm/builds/2",
        "VLLM_IMAGE_TAG": "vllm/vllm-openai:nightly-def456-ubuntu2404",
    }
    expected_tag = (
        "public.ecr.aws/q9t5s3a7/vllm-release-repo:"
        f"def456-{os.uname().machine}-ubuntu2404"
    )
    assert option_values(args, "--tag") == [expected_tag]


def test_local_metadata_args_use_local_overrides() -> None:
    args = run_helper(
        env={
            "VLLM_IMAGE_TAG": "local/test:dev",
            "VLLM_BUILD_COMMIT": "localsha",
            "VLLM_BUILD_PIPELINE": "local-pipeline",
            "VLLM_BUILD_URL": "https://buildkite.example/local",
        },
    )

    assert build_args(args) == {
        "VLLM_BUILD_COMMIT": "localsha",
        "VLLM_BUILD_PIPELINE": "local-pipeline",
        "VLLM_BUILD_URL": "https://buildkite.example/local",
        "VLLM_IMAGE_TAG": "local/test:dev",
    }
    assert option_values(args, "--tag") == ["local/test:dev"]


def test_release_version_lookup_failure_falls_back_to_commit(
    tmp_path: Path,
) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    buildkite_agent = fake_bin / "buildkite-agent"
    buildkite_agent.write_text("#!/bin/sh\nexit 1\n")
    buildkite_agent.chmod(0o755)

    args = run_helper(
        "cu129",
        env={
            "BUILDKITE": "1",
            "BUILDKITE_COMMIT": "fallback123",
            "BUILDKITE_PIPELINE_SLUG": "release",
        },
        path=f"{fake_bin}:{os.environ['PATH']}",
    )

    assert build_args(args)["VLLM_IMAGE_TAG"] == ("vllm/vllm-openai:vfallback123-cu129")


def test_vllm_openai_image_embeds_metadata_contract() -> None:
    dockerfile = (REPO_ROOT / "docker" / "Dockerfile").read_text()

    for expected in (
        "ARG VLLM_BUILD_COMMIT",
        "ARG VLLM_BUILD_PIPELINE",
        "ARG VLLM_BUILD_URL",
        "ARG VLLM_IMAGE_TAG",
        "VLLM_BUILD_COMMIT=${VLLM_BUILD_COMMIT:-unknown}",
        "VLLM_BUILD_PIPELINE=${VLLM_BUILD_PIPELINE:-local}",
        "VLLM_BUILD_URL=${VLLM_BUILD_URL:-}",
        "VLLM_IMAGE_TAG=${VLLM_IMAGE_TAG:-local/vllm-openai:dev}",
        'ai.vllm.build.commit="${VLLM_BUILD_COMMIT}"',
        'ai.vllm.build.pipeline="${VLLM_BUILD_PIPELINE}"',
        'ai.vllm.build.url="${VLLM_BUILD_URL}"',
        'ai.vllm.image.tag="${VLLM_IMAGE_TAG}"',
    ):
        assert expected in dockerfile


def test_runpod_launcher_pins_model_and_tokenizer_to_same_revision() -> None:
    launcher = (REPO_ROOT / "docker" / "runpod" / "runpod-serve.sh").read_text()

    assert '--revision "${revision}"' in launcher
    assert '--tokenizer-revision "${revision}"' in launcher


def test_runpod_launcher_fails_before_vllm_when_tokenizer_preflight_fails(
    tmp_path: Path,
) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    timeout = fake_bin / "timeout"
    timeout.write_text(
        '#!/bin/sh\nprintf \'%s\\n\' "$1" "$2" "$3" '
        '>"$RUNPOD_TEST_TIMEOUT_ARGS"\nshift 3\nexec "$@"\n'
    )
    timeout.chmod(0o755)
    preflight = tmp_path / "preflight-python"
    preflight.write_text(
        '#!/bin/sh\nprintf \'%s\\n\' "$@" >"$RUNPOD_TEST_PREFLIGHT_ARGS"\nexit 29\n'
    )
    preflight.chmod(0o755)
    preflight_args = tmp_path / "preflight-args"
    timeout_args = tmp_path / "timeout-args"
    revision = "1" * 40

    result = subprocess.run(
        ["bash", str(REPO_ROOT / "docker" / "runpod" / "runpod-serve.sh")],
        env={
            "PATH": f"{fake_bin}:/usr/bin:/bin",
            "RUNPOD_TEST_PREFLIGHT_ARGS": str(preflight_args),
            "RUNPOD_TEST_TIMEOUT_ARGS": str(timeout_args),
            "RUNPOD_VLLM_MODEL": "example/model",
            "RUNPOD_VLLM_PREFLIGHT_PYTHON": str(preflight),
            "RUNPOD_VLLM_REVISION": revision,
        },
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 29
    assert preflight_args.read_text().splitlines() == [
        "-",
        "example/model",
        revision,
    ]
    assert timeout_args.read_text().splitlines() == [
        "--signal=TERM",
        "--kill-after=10s",
        "300s",
    ]
    assert "Starting managed vLLM" not in result.stdout


def test_rocm_ci_base_bake_embeds_content_hash_label() -> None:
    bake_file = (REPO_ROOT / "docker" / "docker-bake-rocm.hcl").read_text()

    for expected in (
        'variable "CI_BASE_CONTENT_HASH"',
        'target "ci-base-rocm"',
        'target   = "ci_base"',
        '"vllm.ci_base.content_hash" = CI_BASE_CONTENT_HASH',
    ):
        assert expected in bake_file


def test_rocm_ci_base_metadata_inputs_cover_ci_base_files() -> None:
    ci_bake = (REPO_ROOT / ".buildkite" / "scripts" / "ci-bake-rocm.sh").read_text()

    for expected in (
        "requirements/common.txt",
        "requirements/rocm.txt",
        "requirements/test/rocm.txt",
        "docker/Dockerfile.rocm_base",
        "docker/Dockerfile.rocm",
        "docker/ci-rocm.hcl",
        "docker/docker-bake-rocm.hcl",
        ".buildkite/scripts/ci-bake-rocm.sh",
    ):
        assert expected in ci_bake
