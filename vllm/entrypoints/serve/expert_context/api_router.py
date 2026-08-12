# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Loopback-only, exact-method API for transactional expert contexts."""

from __future__ import annotations

import asyncio
import ipaddress
import os
import secrets
import time
from http import HTTPStatus
from typing import Any

from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, StrictInt

from vllm.engine.protocol import EngineClient
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.expert_context import (
    set_frontend_expert_context,
    set_frontend_expert_context_transition,
)

logger = init_logger(__name__)

_BASE_PATH = "/v1/internal/moe-contexts"
_TOKEN_ENV = "VLLM_MOE_EXPERT_CONTEXT_CONTROL_TOKEN"
_TOKEN_HEADER = "X-vLLM-Expert-Context-Token"
_MAX_BODY_BYTES = 256 * 1024

router = APIRouter(prefix=_BASE_PATH)


class LayerSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    keep: list[StrictInt]


class RegisterRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    context_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
    layers: dict[str, LayerSpec]
    creation_source: str = Field(default="api", min_length=1, max_length=128)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ActivateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    context_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")


def _engine_client(request: Request) -> EngineClient:
    return request.app.state.engine_client


def _require_private_client(request: Request) -> None:
    configured_token = os.environ.get(_TOKEN_ENV)
    if configured_token is None:
        raise HTTPException(status_code=HTTPStatus.NOT_FOUND)
    client = request.client
    if client is None:
        raise HTTPException(status_code=HTTPStatus.NOT_FOUND)
    try:
        is_loopback = ipaddress.ip_address(client.host).is_loopback
    except ValueError:
        is_loopback = False
    if not is_loopback:
        raise HTTPException(status_code=HTTPStatus.NOT_FOUND)
    supplied_token = request.headers.get(_TOKEN_HEADER, "")
    if not secrets.compare_digest(supplied_token, configured_token):
        raise HTTPException(status_code=HTTPStatus.UNAUTHORIZED)
    content_length = request.headers.get("content-length")
    if content_length is not None:
        try:
            too_large = int(content_length) > _MAX_BODY_BYTES
        except ValueError:
            too_large = True
        if too_large:
            raise HTTPException(status_code=HTTPStatus.REQUEST_ENTITY_TOO_LARGE)


def _normalize_results(results: object) -> list[dict[str, Any]]:
    if isinstance(results, dict):
        candidates = [results]
    elif isinstance(results, list):
        candidates = results
    else:
        raise RuntimeError("expert context worker response has an invalid type")
    if not candidates or any(not isinstance(result, dict) for result in candidates):
        raise RuntimeError("expert context worker response is empty or invalid")
    return candidates


async def _collective(
    request: Request,
    method: str,
    *,
    kwargs: dict[str, object] | None = None,
) -> list[dict[str, Any]]:
    results = await _engine_client(request).collective_rpc(method=method, kwargs=kwargs)
    return _normalize_results(results)


def _require_success(
    results: list[dict[str, Any]],
    *,
    fields: tuple[str, ...] = (),
) -> dict[str, Any]:
    failures = [result for result in results if result.get("ok") is not True]
    if failures:
        details = [
            {
                "rank": result.get("rank"),
                "error": result.get("error", "unknown worker error"),
                "error_type": result.get("error_type"),
            }
            for result in failures
        ]
        raise RuntimeError(f"expert context worker operation failed: {details}")
    first = results[0]
    for field in fields:
        values = {result.get(field) for result in results}
        if len(values) != 1:
            raise RuntimeError(
                f"expert context workers disagree on {field}: "
                f"{sorted(repr(value) for value in values)}"
            )
    return first


def _single_dp_unsupported_reason(request: Request) -> str | None:
    parallel_config = _engine_client(request).vllm_config.parallel_config
    if parallel_config.data_parallel_size > 1:
        return (
            "transactional expert contexts currently require data_parallel_size=1; "
            "cross-DP result aggregation is unavailable"
        )
    if parallel_config.pipeline_parallel_size > 1:
        return (
            "transactional expert contexts currently require "
            "pipeline_parallel_size=1; cross-stage topology aggregation is "
            "unavailable"
        )
    args = getattr(request.app.state, "args", None)
    if getattr(args, "api_server_count", 1) > 1:
        return (
            "transactional expert contexts currently require one API server "
            "process so request admission has a single committed context"
        )
    return None


def _parallel_unsupported_reason(engine_client: EngineClient) -> str | None:
    parallel_config = engine_client.vllm_config.parallel_config
    if parallel_config.data_parallel_size > 1:
        return "data parallel expert-context aggregation is unavailable"
    if parallel_config.pipeline_parallel_size > 1:
        return "pipeline parallel expert-context aggregation is unavailable"
    return None


async def _current_workers(request: Request) -> tuple[dict[str, Any], list[int]]:
    results = await _collective(request, "get_expert_context")
    first = _require_success(
        results,
        fields=(
            "active_context_id",
            "active_context_fingerprint",
            "topology_fingerprint",
        ),
    )
    return first, [int(result["process_id"]) for result in results]


async def _rollback_workers(request: Request) -> list[dict[str, Any]]:
    results = await _collective(request, "rollback_expert_context")
    _require_success(
        results,
        fields=(
            "active_context_id",
            "active_context_fingerprint",
            "topology_fingerprint",
        ),
    )
    return results


async def _complete_cleanup(awaitable) -> tuple[Any, bool]:
    task = asyncio.ensure_future(awaitable)
    cancelled = False
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            cancelled = True
    return task.result(), cancelled


async def _begin_transition(request: Request) -> None:
    """Close admission atomically, then drain handlers already past the gate."""
    admission_lock: asyncio.Lock = request.app.state.expert_context_admission_lock
    async with admission_lock:
        request.app.state.expert_context_transition = True
        set_frontend_expert_context_transition(True)
    await request.app.state.expert_context_idle.wait()


async def _end_transition(request: Request) -> None:
    admission_lock: asyncio.Lock = request.app.state.expert_context_admission_lock
    async with admission_lock:
        request.app.state.expert_context_transition = False
        set_frontend_expert_context_transition(False)


async def _release_admitted_request(app: FastAPI) -> None:
    admission_lock: asyncio.Lock = app.state.expert_context_admission_lock
    async with admission_lock:
        active = app.state.expert_context_active_requests - 1
        if active < 0:
            raise RuntimeError("expert context admission counter underflow")
        app.state.expert_context_active_requests = active
        if active == 0:
            app.state.expert_context_idle.set()


async def initialize_frontend_context(engine_client: EngineClient) -> None:
    """Initialize request provenance after the engine is ready."""
    if _parallel_unsupported_reason(engine_client) is not None:
        return
    results = _normalize_results(
        await engine_client.collective_rpc(method="get_expert_context")
    )
    first = _require_success(
        results,
        fields=(
            "active_context_id",
            "active_context_fingerprint",
            "topology_fingerprint",
        ),
    )
    set_frontend_expert_context(
        first["active_context_id"], first["active_context_fingerprint"]
    )


async def _activate(request: Request, context_id: str) -> dict[str, object]:
    unsupported_reason = _single_dp_unsupported_reason(request)
    if unsupported_reason is not None:
        raise HTTPException(status_code=HTTPStatus.CONFLICT, detail=unsupported_reason)
    lock: asyncio.Lock = request.app.state.expert_context_lock
    async with lock:
        started = time.monotonic()
        transition_started = True
        pause_requested = False
        transaction_started = False
        context_safe = True
        old: dict[str, Any] | None = None
        try:
            await _begin_transition(request)
            if request.app.state.expert_context_recovery_required:
                context_safe = False
                before_recovery, worker_process_ids = await _current_workers(request)
                _, recovery_cancelled = await _complete_cleanup(
                    _engine_client(request).resume_generation()
                )
                old, worker_process_ids = await _current_workers(request)
                for field in (
                    "active_context_id",
                    "active_context_fingerprint",
                    "topology_fingerprint",
                ):
                    if old[field] != before_recovery[field]:
                        raise RuntimeError(
                            "expert context changed during scheduler recovery"
                        )
                request.app.state.expert_context_recovery_required = False
                context_safe = True
                set_frontend_expert_context(
                    old["active_context_id"], old["active_context_fingerprint"]
                )
                if recovery_cancelled:
                    raise asyncio.CancelledError
            else:
                context_safe = False
                old, worker_process_ids = await _current_workers(request)
                context_safe = True
            if old["active_context_id"] == context_id:
                return {
                    "old_context_id": old["active_context_id"],
                    "old_context_fingerprint": old["active_context_fingerprint"],
                    "new_context_id": old["active_context_id"],
                    "new_context_fingerprint": old["active_context_fingerprint"],
                    "topology_fingerprint": old["topology_fingerprint"],
                    "duration_ms": 0.0,
                    "process_id": os.getpid(),
                    "worker_process_ids": worker_process_ids,
                    "weights_reloaded": False,
                }
            pause_requested = True
            await _engine_client(request).pause_generation(
                mode="abort", clear_cache=True
            )
            transaction_started = True
            context_safe = False
            prepared = await _collective(
                request,
                "prepare_expert_context",
                kwargs={"context_id": context_id},
            )
            _require_success(
                prepared,
                fields=("context_fingerprint", "topology_fingerprint"),
            )
            committed = await _collective(request, "commit_expert_context")
            new = _require_success(
                committed,
                fields=(
                    "active_context_id",
                    "active_context_fingerprint",
                    "topology_fingerprint",
                ),
            )
            if new["active_context_id"] != context_id:
                raise RuntimeError(
                    "expert context workers committed an unexpected context"
                )
            context_safe = True
            set_frontend_expert_context(
                new["active_context_id"],
                new["active_context_fingerprint"],
            )
            duration_ms = (time.monotonic() - started) * 1000
            return {
                "old_context_id": old["active_context_id"],
                "old_context_fingerprint": old["active_context_fingerprint"],
                "new_context_id": new["active_context_id"],
                "new_context_fingerprint": new["active_context_fingerprint"],
                "topology_fingerprint": new["topology_fingerprint"],
                "duration_ms": duration_ms,
                "process_id": os.getpid(),
                "worker_process_ids": worker_process_ids,
                "weights_reloaded": False,
            }
        except BaseException as error:
            if transaction_started:
                try:
                    rollback, cleanup_cancelled = await _complete_cleanup(
                        _rollback_workers(request)
                    )
                    restored = rollback[0]
                    if old is None or any(
                        restored[field] != old[field]
                        for field in (
                            "active_context_id",
                            "active_context_fingerprint",
                            "topology_fingerprint",
                        )
                    ):
                        raise RuntimeError(
                            "rollback did not restore the prior expert context"
                        )
                    context_safe = True
                    if cleanup_cancelled:
                        raise asyncio.CancelledError
                except BaseException as rollback_error:
                    if isinstance(rollback_error, asyncio.CancelledError):
                        raise
                    raise HTTPException(
                        status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                        detail=(
                            f"expert context activation failed ({error}); rollback "
                            f"also failed ({rollback_error})"
                        ),
                    ) from rollback_error
            if isinstance(error, asyncio.CancelledError | HTTPException):
                raise
            raise HTTPException(
                status_code=HTTPStatus.CONFLICT, detail=str(error)
            ) from error
        finally:
            if transition_started and not context_safe:
                request.app.state.expert_context_recovery_required = True
            elif transition_started:
                cleanup_cancelled = False
                if pause_requested:
                    try:
                        _, resume_cancelled = await _complete_cleanup(
                            _engine_client(request).resume_generation()
                        )
                    except BaseException:
                        request.app.state.expert_context_recovery_required = True
                        raise
                    cleanup_cancelled |= resume_cancelled
                try:
                    _, transition_cancelled = await _complete_cleanup(
                        _end_transition(request)
                    )
                except BaseException:
                    request.app.state.expert_context_recovery_required = True
                    raise
                cleanup_cancelled |= transition_cancelled
                if cleanup_cancelled:
                    raise asyncio.CancelledError


@router.get("/capabilities")
async def capabilities(request: Request):
    _require_private_client(request)
    try:
        if reason := _single_dp_unsupported_reason(request):
            return {
                "supported": False,
                "unsupported_reason": reason,
                "topology_fingerprint": None,
                "active_context_id": None,
                "active_context_fingerprint": None,
                "layers": [],
            }
        async with request.app.state.expert_context_lock:
            results = await _collective(request, "get_expert_context_capabilities")
            first = _require_success(
                results,
                fields=(
                    "supported",
                    "active_context_fingerprint",
                    "topology_fingerprint",
                ),
            )
        response = {
            key: value for key, value in first.items() if key not in {"ok", "rank"}
        }
        response["worker_count"] = len(results)
        response["worker_ranks"] = [result["rank"] for result in results]
        response["process_id"] = os.getpid()
        response["worker_process_ids"] = [
            int(result["process_id"]) for result in results
        ]
        set_frontend_expert_context(
            first["active_context_id"], first["active_context_fingerprint"]
        )
        return response
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(
            status_code=HTTPStatus.CONFLICT, detail=str(error)
        ) from error


@router.get("/current")
async def current(request: Request):
    _require_private_client(request)
    try:
        async with request.app.state.expert_context_lock:
            first, process_ids = await _current_workers(request)
            set_frontend_expert_context(
                first["active_context_id"],
                first["active_context_fingerprint"],
            )
        return {
            key: value for key, value in first.items() if key not in {"ok", "rank"}
        } | {
            "process_id": os.getpid(),
            "worker_process_ids": process_ids,
            "serving_safe": not request.app.state.expert_context_transition,
            "recovery_required": request.app.state.expert_context_recovery_required,
        }
    except Exception as error:
        raise HTTPException(
            status_code=HTTPStatus.CONFLICT, detail=str(error)
        ) from error


@router.post("/register")
async def register(body: RegisterRequest, request: Request):
    _require_private_client(request)
    if reason := _single_dp_unsupported_reason(request):
        raise HTTPException(status_code=HTTPStatus.CONFLICT, detail=reason)
    try:
        async with request.app.state.expert_context_lock:
            results = await _collective(
                request,
                "register_expert_context",
                kwargs={
                    "context_id": body.context_id,
                    "layers": {
                        layer_id: layer.model_dump()
                        for layer_id, layer in body.layers.items()
                    },
                    "creation_source": body.creation_source,
                    "metadata": body.metadata,
                },
            )
            first = _require_success(
                results,
                fields=("context_fingerprint", "topology_fingerprint"),
            )
        return {key: value for key, value in first.items() if key not in {"ok", "rank"}}
    except Exception as error:
        raise HTTPException(
            status_code=HTTPStatus.UNPROCESSABLE_ENTITY, detail=str(error)
        ) from error


@router.post("/activate")
async def activate(body: ActivateRequest, request: Request):
    _require_private_client(request)
    return await _activate(request, body.context_id)


@router.post("/reset")
async def reset(request: Request):
    _require_private_client(request)
    return await _activate(request, "baseline")


def attach_router(app: FastAPI) -> None:
    token = os.environ.get(_TOKEN_ENV)
    if token is None:
        return
    if len(token) < 32:
        raise ValueError(f"{_TOKEN_ENV} must contain at least 32 characters")
    app.state.expert_context_lock = asyncio.Lock()
    app.state.expert_context_admission_lock = asyncio.Lock()
    app.state.expert_context_active_requests = 0
    app.state.expert_context_idle = asyncio.Event()
    app.state.expert_context_idle.set()
    app.state.expert_context_transition = False
    app.state.expert_context_recovery_required = False

    @app.middleware("http")
    async def reject_during_transition(request: Request, call_next):
        if request.url.path.startswith(_BASE_PATH):
            try:
                _require_private_client(request)
            except HTTPException as error:
                return JSONResponse(
                    status_code=error.status_code,
                    content={"detail": error.detail},
                )
            return await call_next(request)
        if request.url.path in {"/health", "/ping"}:
            return await call_next(request)

        admission_lock: asyncio.Lock = request.app.state.expert_context_admission_lock
        async with admission_lock:
            if request.app.state.expert_context_transition:
                return JSONResponse(
                    status_code=HTTPStatus.SERVICE_UNAVAILABLE,
                    content={"detail": "expert context transition in progress"},
                )
            request.app.state.expert_context_active_requests += 1
            request.app.state.expert_context_idle.clear()

        released = False

        async def release() -> None:
            nonlocal released
            if released:
                return
            released = True
            _, cleanup_cancelled = await _complete_cleanup(
                _release_admitted_request(request.app)
            )
            if cleanup_cancelled:
                raise asyncio.CancelledError

        try:
            response = await call_next(request)
        except BaseException:
            await release()
            raise

        body_iterator = getattr(response, "body_iterator", None)
        if body_iterator is None:
            await release()
            return response

        async def body_with_admission_lease():
            try:
                async for chunk in body_iterator:
                    yield chunk
            finally:
                await release()

        response.body_iterator = body_with_admission_lease()
        return response

    app.include_router(router)
    logger.info("Enabled loopback-only expert-context control API")
