# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
import os
from concurrent.futures import Future
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from vllm.entrypoints.serve.expert_context.api_router import attach_router
from vllm.model_executor.layers.fused_moe.expert_context import (
    set_frontend_expert_context,
    set_frontend_expert_context_transition,
)
from vllm.v1.engine.core import _finish_pause_after_idle

pytestmark = pytest.mark.skip_global_cleanup

_TOKEN = "test-expert-context-token-at-least-32-characters"
_HEADERS = {"X-vLLM-Expert-Context-Token": _TOKEN}
_BASELINE_FINGERPRINT = "a" * 64
_MASKED_FINGERPRINT = "b" * 64
_REGISTERED_FINGERPRINT = "c" * 64
_PROFILE_FINGERPRINT = "d" * 64
_TOPOLOGY_FINGERPRINT = "e" * 64


@pytest.fixture(autouse=True)
def reset_frontend_context():
    set_frontend_expert_context(None, None)
    set_frontend_expert_context_transition(False)
    yield
    set_frontend_expert_context(None, None)
    set_frontend_expert_context_transition(False)


class FakeEngineClient:
    def __init__(self):
        self.vllm_config = SimpleNamespace(
            parallel_config=SimpleNamespace(
                data_parallel_size=1,
                pipeline_parallel_size=1,
            )
        )
        self.calls = []
        self.contexts = {
            "baseline": _BASELINE_FINGERPRINT,
            "masked": _MASKED_FINGERPRINT,
        }
        self.active = "baseline"
        self.previous = None
        self.pending = None
        self.fail_current = False
        self.fail_commit = False
        self.fail_rollback = False
        self.fail_pause = False
        self.fail_resume = False
        self.pause_entered = asyncio.Event()
        self.pause_release: asyncio.Event | None = None

    def _current(self):
        return {
            "ok": True,
            "rank": 0,
            "active_context_id": self.active,
            "active_context_fingerprint": self.contexts[self.active],
            "profile_fingerprint": self.contexts[self.active],
            "topology_fingerprint": _TOPOLOGY_FINGERPRINT,
            "process_id": 42,
            "weights_reloaded": False,
        }

    async def collective_rpc(self, method, timeout=None, args=(), kwargs=None):
        self.calls.append((method, kwargs))
        kwargs = kwargs or {}
        if method == "get_expert_context":
            if self.fail_current:
                return [{"ok": False, "rank": 0, "error": "current failed"}]
            return [self._current()]
        if method == "get_expert_context_capabilities":
            return [
                {
                    **self._current(),
                    "supported": True,
                    "unsupported_reason": None,
                    "layers": [],
                }
            ]
        if method == "register_expert_context":
            self.contexts[kwargs["context_id"]] = _REGISTERED_FINGERPRINT
            return [
                {
                    "ok": True,
                    "rank": 0,
                    "context_id": kwargs["context_id"],
                    "context_fingerprint": _REGISTERED_FINGERPRINT,
                    "profile_fingerprint": _PROFILE_FINGERPRINT,
                    "topology_fingerprint": _TOPOLOGY_FINGERPRINT,
                    "layers": kwargs["layers"],
                }
            ]
        if method == "prepare_expert_context":
            context_id = kwargs["context_id"]
            if context_id not in self.contexts:
                return [{"ok": False, "rank": 0, "error": "unknown context"}]
            self.previous = self.active
            self.pending = context_id
            return [
                {
                    "ok": True,
                    "rank": 0,
                    "context_fingerprint": self.contexts[context_id],
                    "topology_fingerprint": _TOPOLOGY_FINGERPRINT,
                }
            ]
        if method == "commit_expert_context":
            if self.fail_commit:
                return [{"ok": False, "rank": 0, "error": "injected failure"}]
            self.active = self.pending
            self.pending = None
            return [self._current()]
        if method == "rollback_expert_context":
            if self.fail_rollback:
                return [{"ok": False, "rank": 0, "error": "rollback failed"}]
            if self.previous is not None:
                self.active = self.previous
            self.previous = None
            self.pending = None
            return [self._current()]
        raise AssertionError(method)

    async def pause_generation(self, **kwargs):
        self.calls.append(("pause_generation", kwargs))
        self.pause_entered.set()
        if self.pause_release is not None:
            await self.pause_release.wait()
        if self.fail_pause:
            raise RuntimeError("connector cache reset failed")

    async def resume_generation(self):
        self.calls.append(("resume_generation", None))
        if self.fail_resume:
            raise RuntimeError("resume failed")


def _app(monkeypatch, engine=None):
    monkeypatch.setenv("VLLM_MOE_EXPERT_CONTEXT_CONTROL_TOKEN", _TOKEN)
    app = FastAPI()
    app.state.engine_client = engine or FakeEngineClient()
    attach_router(app)
    return app


def test_control_api_requires_loopback_and_dedicated_token(monkeypatch):
    app = _app(monkeypatch)
    with TestClient(app, client=None) as client:
        assert (
            client.get("/v1/internal/moe-contexts/current", headers=_HEADERS)
        ).status_code == 404

    with TestClient(app, client=("203.0.113.5", 1234)) as client:
        assert (
            client.get("/v1/internal/moe-contexts/current", headers=_HEADERS)
        ).status_code == 404

    with TestClient(app, client=("127.0.0.1", 1234)) as client:
        assert (client.get("/v1/internal/moe-contexts/current")).status_code == 401
        response = client.get("/v1/internal/moe-contexts/current", headers=_HEADERS)
        assert response.status_code == 200
        current = response.json()
        assert current["active_context_id"] == "baseline"
        assert current["active_context_fingerprint"] == _BASELINE_FINGERPRINT
        assert current["process_id"] == os.getpid()
        assert current["worker_process_ids"] == [42]
        assert set(current) == {
            "active_context_id",
            "active_context_fingerprint",
            "profile_fingerprint",
            "topology_fingerprint",
            "process_id",
            "worker_process_ids",
            "weights_reloaded",
            "serving_safe",
            "recovery_required",
        }
        assert current["serving_safe"] is True
        assert current["recovery_required"] is False
        capabilities = client.get(
            "/v1/internal/moe-contexts/capabilities", headers=_HEADERS
        ).json()
        assert capabilities["process_id"] == os.getpid()
        assert capabilities["worker_process_ids"] == [42]


def test_current_rejects_worker_profile_fingerprint_disagreement(monkeypatch):
    class DisagreeingCurrentEngine(FakeEngineClient):
        async def collective_rpc(self, method, timeout=None, args=(), kwargs=None):
            results = await super().collective_rpc(method, timeout, args, kwargs)
            if method != "get_expert_context":
                return results
            disagreeing = {
                **results[0],
                "rank": 1,
                "profile_fingerprint": "f" * 64,
            }
            return [results[0], disagreeing]

    app = _app(monkeypatch, DisagreeingCurrentEngine())
    with TestClient(app, client=("127.0.0.1", 1234)) as client:
        response = client.get(
            "/v1/internal/moe-contexts/current",
            headers=_HEADERS,
        )

    assert response.status_code == 409
    assert "workers disagree on profile_fingerprint" in response.json()["detail"]


def test_control_api_registers_and_activates_transactionally(monkeypatch):
    engine = FakeEngineClient()
    app = _app(monkeypatch, engine)
    with TestClient(app, client=("127.0.0.1", 1234)) as client:
        registered = client.post(
            "/v1/internal/moe-contexts/register",
            headers=_HEADERS,
            json={
                "context_id": "registered",
                "layers": {"2": {"keep": [0, 1]}},
            },
        )
        assert registered.status_code == 200
        activated = client.post(
            "/v1/internal/moe-contexts/activate",
            headers=_HEADERS,
            json={"context_id": "registered"},
        )

    assert activated.status_code == 200
    assert activated.json()["old_context_id"] == "baseline"
    assert activated.json()["new_context_id"] == "registered"
    assert activated.json()["weights_reloaded"] is False
    methods = [method for method, _ in engine.calls]
    assert methods[-5:] == [
        "get_expert_context",
        "pause_generation",
        "prepare_expert_context",
        "commit_expert_context",
        "resume_generation",
    ]


@pytest.mark.parametrize(
    ("field", "mismatch"),
    [
        ("context_id", "different-context"),
        ("context_fingerprint", "f" * 64),
        ("profile_fingerprint", "f" * 64),
        ("topology_fingerprint", "f" * 64),
        ("layers", {"2": {"keep": [0, 2]}}),
    ],
)
def test_control_api_rejects_register_worker_disagreement(monkeypatch, field, mismatch):
    class DisagreeingRegisterEngine(FakeEngineClient):
        async def collective_rpc(self, method, timeout=None, args=(), kwargs=None):
            results = await super().collective_rpc(method, timeout, args, kwargs)
            if method != "register_expert_context":
                return results
            disagreeing = {**results[0], "rank": 1, field: mismatch}
            return [results[0], disagreeing]

    app = _app(monkeypatch, DisagreeingRegisterEngine())
    with TestClient(app, client=("127.0.0.1", 1234)) as client:
        response = client.post(
            "/v1/internal/moe-contexts/register",
            headers=_HEADERS,
            json={
                "context_id": "registered",
                "layers": {"2": {"keep": [0, 1]}},
            },
        )

    assert response.status_code == 422
    assert f"workers disagree on {field}" in response.json()["detail"]


def test_control_api_rolls_back_and_resumes_after_commit_failure(monkeypatch):
    engine = FakeEngineClient()
    engine.fail_commit = True
    app = _app(monkeypatch, engine)
    with TestClient(app, client=("127.0.0.1", 1234)) as client:
        response = client.post(
            "/v1/internal/moe-contexts/activate",
            headers=_HEADERS,
            json={"context_id": "masked"},
        )

    assert response.status_code == 409
    assert engine.active == "baseline"
    methods = [method for method, _ in engine.calls]
    assert "rollback_expert_context" in methods
    assert methods[-1] == "resume_generation"


def test_cache_reset_failure_prevents_prepare_and_commit(monkeypatch):
    engine = FakeEngineClient()
    engine.fail_pause = True
    app = _app(monkeypatch, engine)
    with TestClient(app, client=("127.0.0.1", 1234)) as client:
        response = client.post(
            "/v1/internal/moe-contexts/activate",
            headers=_HEADERS,
            json={"context_id": "masked"},
        )

    assert response.status_code == 409
    methods = [method for method, _ in engine.calls]
    assert "prepare_expert_context" not in methods
    assert "commit_expert_context" not in methods
    assert methods[-1] == "resume_generation"


def test_async_pause_cache_reset_failure_resolves_future_with_error():
    def fail_reset():
        raise RuntimeError("connector cache reset failed")

    engine = SimpleNamespace(_reset_caches=fail_reset)
    future = Future()

    _finish_pause_after_idle(engine, future, clear_cache=True)

    with pytest.raises(RuntimeError, match="connector cache reset failed"):
        future.result()


def test_activation_closes_admission_and_drains_requests_before_switch(monkeypatch):
    engine = FakeEngineClient()
    app = _app(monkeypatch, engine)
    entered = asyncio.Event()
    release = asyncio.Event()

    @app.get("/hold")
    async def hold():
        entered.set()
        await release.wait()
        return {"ok": True}

    async def scenario():
        transport = httpx.ASGITransport(app=app, client=("127.0.0.1", 1234))
        async with httpx.AsyncClient(
            transport=transport, base_url="http://testserver"
        ) as client:
            admitted = asyncio.create_task(client.get("/hold"))
            await entered.wait()
            activation = asyncio.create_task(
                client.post(
                    "/v1/internal/moe-contexts/activate",
                    headers=_HEADERS,
                    json={"context_id": "masked"},
                )
            )
            while not app.state.expert_context_transition:
                await asyncio.sleep(0)

            assert not any(call[0] == "pause_generation" for call in engine.calls)
            rejected = await client.get("/hold")
            assert rejected.status_code == 503

            release.set()
            assert (await admitted).status_code == 200
            activated = await activation

        assert activated.status_code == 200
        pause_calls = [call for call in engine.calls if call[0] == "pause_generation"]
        assert pause_calls == [
            ("pause_generation", {"mode": "abort", "clear_cache": True})
        ]

    asyncio.run(scenario())


def test_activation_rollback_failure_keeps_inference_fail_closed(monkeypatch):
    engine = FakeEngineClient()
    engine.fail_commit = True
    engine.fail_rollback = True
    app = _app(monkeypatch, engine)
    with TestClient(
        app, client=("127.0.0.1", 1234), raise_server_exceptions=False
    ) as client:
        response = client.post(
            "/v1/internal/moe-contexts/activate",
            headers=_HEADERS,
            json={"context_id": "masked"},
        )
        ordinary = client.get("/ordinary")

    assert response.status_code == 500
    assert ordinary.status_code == 503
    assert app.state.expert_context_transition is True
    assert not any(call[0] == "resume_generation" for call in engine.calls)


def test_activation_resume_failure_keeps_inference_fail_closed(monkeypatch):
    engine = FakeEngineClient()
    engine.fail_resume = True
    app = _app(monkeypatch, engine)
    with TestClient(
        app, client=("127.0.0.1", 1234), raise_server_exceptions=False
    ) as client:
        response = client.post(
            "/v1/internal/moe-contexts/activate",
            headers=_HEADERS,
            json={"context_id": "masked"},
        )
        ordinary = client.get("/ordinary")

    assert response.status_code == 500
    assert ordinary.status_code == 503
    assert engine.active == "masked"
    assert app.state.expert_context_transition is True
    assert app.state.expert_context_recovery_required is True


def test_retry_after_resume_failure_recovers_before_reopening(monkeypatch):
    engine = FakeEngineClient()
    engine.fail_resume = True
    app = _app(monkeypatch, engine)
    with TestClient(
        app, client=("127.0.0.1", 1234), raise_server_exceptions=False
    ) as client:
        failed = client.post(
            "/v1/internal/moe-contexts/activate",
            headers=_HEADERS,
            json={"context_id": "masked"},
        )
        assert failed.status_code == 500
        assert client.get("/ordinary").status_code == 503

        engine.fail_resume = False
        recovered = client.post(
            "/v1/internal/moe-contexts/activate",
            headers=_HEADERS,
            json={"context_id": "masked"},
        )
        ordinary = client.get("/ordinary")

    assert recovered.status_code == 200
    assert ordinary.status_code == 404
    assert app.state.expert_context_transition is False
    assert app.state.expert_context_recovery_required is False
    assert [call[0] for call in engine.calls].count("resume_generation") == 2


def test_activation_requires_unanimous_current_state_before_reopening(monkeypatch):
    engine = FakeEngineClient()
    engine.fail_current = True
    app = _app(monkeypatch, engine)
    with TestClient(
        app, client=("127.0.0.1", 1234), raise_server_exceptions=False
    ) as client:
        response = client.post(
            "/v1/internal/moe-contexts/activate",
            headers=_HEADERS,
            json={"context_id": "masked"},
        )
        ordinary = client.get("/ordinary")

    assert response.status_code == 409
    assert ordinary.status_code == 503
    assert app.state.expert_context_transition is True


def test_cancelled_pause_is_resumed_and_reopens_admission(monkeypatch):
    engine = FakeEngineClient()
    engine.pause_release = asyncio.Event()
    app = _app(monkeypatch, engine)

    async def scenario():
        transport = httpx.ASGITransport(
            app=app, client=("127.0.0.1", 1234), raise_app_exceptions=False
        )
        async with httpx.AsyncClient(
            transport=transport, base_url="http://testserver"
        ) as client:
            activation = asyncio.create_task(
                client.post(
                    "/v1/internal/moe-contexts/activate",
                    headers=_HEADERS,
                    json={"context_id": "masked"},
                )
            )
            await engine.pause_entered.wait()
            activation.cancel()
            with pytest.raises(asyncio.CancelledError):
                await activation
            assert app.state.expert_context_transition is False
            assert (await client.get("/ordinary")).status_code == 404

        assert engine.calls[-1] == ("resume_generation", None)

    asyncio.run(scenario())


def test_cancelled_drain_reopens_admission_without_switching(monkeypatch):
    engine = FakeEngineClient()
    app = _app(monkeypatch, engine)
    entered = asyncio.Event()
    release = asyncio.Event()

    @app.get("/hold-cancel")
    async def hold_cancel():
        entered.set()
        await release.wait()
        return {"ok": True}

    async def scenario():
        transport = httpx.ASGITransport(
            app=app, client=("127.0.0.1", 1234), raise_app_exceptions=False
        )
        async with httpx.AsyncClient(
            transport=transport, base_url="http://testserver"
        ) as client:
            admitted = asyncio.create_task(client.get("/hold-cancel"))
            await entered.wait()
            activation = asyncio.create_task(
                client.post(
                    "/v1/internal/moe-contexts/activate",
                    headers=_HEADERS,
                    json={"context_id": "masked"},
                )
            )
            while not app.state.expert_context_transition:
                await asyncio.sleep(0)
            activation.cancel()
            with pytest.raises(asyncio.CancelledError):
                await activation

            assert app.state.expert_context_transition is False
            assert not any(call[0] == "pause_generation" for call in engine.calls)
            assert (await client.get("/ordinary")).status_code == 404
            release.set()
            assert (await admitted).status_code == 200

    asyncio.run(scenario())
