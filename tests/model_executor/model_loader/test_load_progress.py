# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from vllm import model_load_progress as load_progress
from vllm.config import CompilationMode
from vllm.model_executor.model_loader import weight_utils
from vllm.v1.worker import gpu_worker

pytestmark = pytest.mark.skip_global_cleanup


class _Response:
    status = 204

    def read(self, _size: int) -> bytes:
        return b""


class _Connection:
    calls: list[tuple[str, str, bytes, dict[str, str]]] = []

    def __init__(self, host: str, port: int, timeout: float) -> None:
        assert host == "127.0.0.1"
        assert port == 4321
        assert timeout == 1.0

    def request(
        self,
        method: str,
        path: str,
        *,
        body: bytes,
        headers: dict[str, str],
    ) -> None:
        self.calls.append((method, path, body, headers))

    def getresponse(self) -> _Response:
        return _Response()

    def close(self) -> None:
        return None


def test_progress_callback_is_loopback_only_and_separately_authenticated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    token = "t" * 43
    monkeypatch.setenv(
        "VLLM_MOE_MODEL_LOAD_PROGRESS_URL",
        "http://127.0.0.1:4321/model-load/nonce",
    )
    monkeypatch.setenv("VLLM_MOE_MODEL_LOAD_PROGRESS_TOKEN", token)
    monkeypatch.setenv("VLLM_MOE_MODEL_LOAD_PROGRESS_SESSION_ID", "session-a")
    monkeypatch.setattr(load_progress.http.client, "HTTPConnection", _Connection)
    _Connection.calls.clear()

    assert load_progress.emit_model_load_progress(
        "downloading",
        "completed",
        "Pinned snapshot materialized",
        rank=1,
        world_size=2,
        bytes_current=4096,
        bytes_total=4096,
        files_current=4,
        files_total=4,
    )

    method, path, body, headers = _Connection.calls.pop()
    payload = json.loads(body)
    assert (method, path) == ("POST", "/model-load/nonce")
    assert headers["X-vLLM-Model-Load-Token"] == token
    assert token.encode() not in body
    assert payload["session_id"] == "session-a"
    assert payload["phase"] == "downloading"
    assert payload["rank"] == 1
    assert payload["world_size"] == 2
    assert payload["bytes_current"] == 4096

    monkeypatch.setenv(
        "VLLM_MOE_MODEL_LOAD_PROGRESS_URL",
        "https://example.com/model-load/nonce",
    )
    assert not load_progress.emit_model_load_progress(
        "downloading", "started", "must stay local"
    )
    assert _Connection.calls == []


def test_phase_context_reports_completion_and_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[tuple[str, str, str, int | None, int]] = []

    def record(
        phase: str,
        status: str,
        detail: str,
        *,
        rank: int | None = None,
        world_size: int = 1,
        **_kwargs,
    ) -> bool:
        events.append((phase, status, detail, rank, world_size))
        return True

    monkeypatch.setattr(load_progress, "emit_model_load_progress", record)
    owner = SimpleNamespace(
        rank=0,
        parallel_config=SimpleNamespace(world_size=2),
    )

    with load_progress.model_load_progress(
        "loading_weights", "Loading checkpoint weights", owner=owner
    ):
        pass

    with (
        pytest.raises(RuntimeError, match="boom"),
        load_progress.model_load_progress(
            "capturing_graphs", "Capturing CUDA graphs", owner=owner
        ),
    ):
        raise RuntimeError("boom")

    assert [(phase, status) for phase, status, *_ in events] == [
        ("loading_weights", "started"),
        ("loading_weights", "completed"),
        ("capturing_graphs", "started"),
        ("capturing_graphs", "failed"),
    ]
    assert all(rank == 0 and world_size == 2 for *_, rank, world_size in events)
    assert "RuntimeError" in events[-1][2]


def test_phase_decorator_reports_only_when_real_boundary_is_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[tuple[str, str]] = []

    monkeypatch.setattr(
        load_progress,
        "emit_model_load_progress",
        lambda phase, status, _detail, **_metrics: events.append((phase, status)),
    )

    class Runner:
        enabled = False

        @load_progress.track_model_load_phase(
            "capturing_graphs",
            "Capturing CUDA graphs",
            enabled_when=lambda runner: runner.enabled,
        )
        def capture(self) -> int:
            return 7

    runner = Runner()
    assert runner.capture() == 7
    assert events == []

    runner.enabled = True
    assert runner.capture() == 7
    assert events == [
        ("capturing_graphs", "started"),
        ("capturing_graphs", "completed"),
    ]


def test_startup_compile_phase_completes_after_all_configured_shapes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sequence: list[tuple[str, int | str]] = []
    progress_events: list[tuple[str, str, str]] = []

    @contextmanager
    def record(phase: str, detail: str, *, owner: object):
        assert owner is worker
        sequence.append((phase, "started"))
        progress_events.append((phase, "started", detail))
        yield
        sequence.append((phase, "completed"))
        progress_events.append((phase, "completed", detail))

    class Runner:
        lora_config = None

        def _dummy_run(
            self,
            size: int,
            *,
            skip_eplb: bool,
            remove_lora: bool,
        ) -> None:
            assert skip_eplb and not remove_lora
            sequence.append(("shape", size))

    worker = SimpleNamespace(
        rank=0,
        parallel_config=SimpleNamespace(world_size=2),
        vllm_config=SimpleNamespace(
            compilation_config=SimpleNamespace(mode=CompilationMode.VLLM_COMPILE)
        ),
        model_runner=Runner(),
    )
    monkeypatch.setattr(gpu_worker, "model_load_progress", record)
    monkeypatch.setenv("VLLM_USE_AOT_COMPILE", "0")

    gpu_worker.Worker._run_startup_compile_warmups(worker, [4, 8])

    assert sequence == [
        ("compiling", "started"),
        ("shape", 8),
        ("shape", 4),
        ("compiling", "completed"),
    ]
    assert progress_events == [
        ("compiling", "started", "Compiling configured startup model shapes"),
        ("compiling", "completed", "Compiling configured startup model shapes"),
    ]

    sequence.clear()
    progress_events.clear()
    monkeypatch.setenv("VLLM_USE_AOT_COMPILE", "1")
    gpu_worker.Worker._run_startup_compile_warmups(worker, [4, 8])
    assert sequence == [
        ("compiling", "started"),
        ("shape", 8),
        ("shape", 4),
        ("compiling", "completed"),
    ]
    assert progress_events == [
        (
            "compiling",
            "started",
            "Preparing and validating configured AOT startup model artifacts",
        ),
        (
            "compiling",
            "completed",
            "Preparing and validating configured AOT startup model artifacts",
        ),
    ]

    sequence.clear()
    progress_events.clear()
    monkeypatch.setenv("VLLM_USE_AOT_COMPILE", "0")
    gpu_worker.Worker._run_startup_compile_warmups(worker, [])
    assert sequence == []
    assert progress_events == []

    worker.vllm_config.compilation_config.mode = CompilationMode.NONE
    gpu_worker.Worker._run_startup_compile_warmups(worker, [4, 8])
    assert sequence == [("shape", 8), ("shape", 4)]
    assert progress_events == []


def test_hugging_face_snapshot_reports_resolved_file_and_byte_counts(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "model-00001.safetensors").write_bytes(b"a" * 10)
    (tmp_path / "model-00002.safetensors").write_bytes(b"b" * 20)
    events: list[dict[str, object]] = []

    monkeypatch.setattr(weight_utils, "current_worker_coordinates", lambda: (0, 2))
    monkeypatch.setattr(weight_utils, "_cached_weight_stats", lambda *args: (2, 30))
    monkeypatch.setattr(
        weight_utils,
        "_download_weights_from_hf",
        lambda *args: (str(tmp_path), ["*.safetensors"]),
    )
    monkeypatch.setattr(
        weight_utils,
        "emit_model_load_progress",
        lambda phase, status, detail, **metrics: events.append(
            {
                "phase": phase,
                "status": status,
                "detail": detail,
                **metrics,
            }
        ),
    )

    resolved = weight_utils.download_weights_from_hf(
        "org/model",
        None,
        ["*.safetensors"],
        "1" * 40,
    )

    assert resolved == str(tmp_path)
    assert [(event["phase"], event["status"]) for event in events] == [
        ("checking_cache", "started"),
        ("checking_cache", "completed"),
        ("downloading", "started"),
        ("downloading", "completed"),
    ]
    assert events[1]["files_current"] == 2
    assert events[1]["bytes_current"] == 30
    assert events[-1]["files_current"] == 2
    assert events[-1]["bytes_current"] == 30
    assert "not measured network transfer" in str(events[-1]["detail"])
    assert all(event["rank"] == 0 for event in events)
    assert all(event["world_size"] == 1 for event in events)


def test_hugging_face_snapshot_progress_is_rank_zero_only(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[tuple[str, str]] = []
    downloads: list[str] = []

    def fake_download(model_name: str, *args: object) -> tuple[str, list[str]]:
        downloads.append(model_name)
        return str(tmp_path), ["*.bin"]

    monkeypatch.setattr(weight_utils, "current_worker_coordinates", lambda: (1, 2))
    monkeypatch.setattr(
        weight_utils,
        "_download_weights_from_hf",
        fake_download,
    )
    monkeypatch.setattr(
        weight_utils,
        "emit_model_load_progress",
        lambda phase, status, _detail, **_metrics: events.append((phase, status)),
    )

    assert weight_utils.download_weights_from_hf("org/model", None, ["*.bin"]) == str(
        tmp_path
    )
    assert downloads == ["org/model"]
    assert events == []
