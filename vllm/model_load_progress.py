# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

import http.client
import json
import os
from collections.abc import Callable, Iterator
from contextlib import contextmanager, suppress
from functools import wraps
from typing import Any, ParamSpec, TypeVar
from urllib.parse import urlsplit
from uuid import uuid4

_ENDPOINT_ENV = "VLLM_MOE_MODEL_LOAD_PROGRESS_URL"
_TOKEN_ENV = "VLLM_MOE_MODEL_LOAD_PROGRESS_TOKEN"
_SESSION_ENV = "VLLM_MOE_MODEL_LOAD_PROGRESS_SESSION_ID"
_PHASES = {
    "checking_cache",
    "downloading",
    "loading_weights",
    "initializing_distributed_workers",
    "compiling",
    "capturing_graphs",
    "warming",
}
_STATUSES = {"started", "completed", "failed"}
_CALLBACK_TIMEOUT_SECONDS = 1.0

P = ParamSpec("P")
R = TypeVar("R")


def current_worker_coordinates(owner: object | None = None) -> tuple[int | None, int]:
    rank = getattr(owner, "rank", None)
    parallel_config = getattr(owner, "parallel_config", None)
    world_size = getattr(parallel_config, "world_size", None)
    if isinstance(rank, int) and isinstance(world_size, int) and world_size > 0:
        return rank, world_size

    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            return dist.get_rank(), dist.get_world_size()
    except Exception:
        pass
    return None, 1


def emit_model_load_progress(
    phase: str,
    status: str,
    detail: str,
    *,
    rank: int | None = None,
    world_size: int = 1,
    bytes_current: int | None = None,
    bytes_total: int | None = None,
    files_current: int | None = None,
    files_total: int | None = None,
) -> bool:
    """Send one bounded startup event to an authenticated loopback listener."""
    endpoint = os.environ.get(_ENDPOINT_ENV)
    token = os.environ.get(_TOKEN_ENV)
    session_id = os.environ.get(_SESSION_ENV)
    if not endpoint or not token or not session_id or len(token) < 32:
        return False
    if phase not in _PHASES or status not in _STATUSES:
        return False

    try:
        parsed = urlsplit(endpoint)
        port = parsed.port
    except ValueError:
        return False
    hostname = parsed.hostname
    if (
        parsed.scheme != "http"
        or hostname != "127.0.0.1"
        or port is None
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        return False

    payload = {
        "version": 1,
        "event_id": f"{os.getpid()}-{uuid4().hex}",
        "session_id": session_id,
        "phase": phase,
        "status": status,
        "detail": detail[:500],
        "process_id": os.getpid(),
        "rank": rank,
        "world_size": world_size,
        "bytes_current": bytes_current,
        "bytes_total": bytes_total,
        "files_current": files_current,
        "files_total": files_total,
    }
    body = json.dumps(payload, separators=(",", ":")).encode()
    path = parsed.path or "/"
    connection: http.client.HTTPConnection | None = None
    try:
        connection = http.client.HTTPConnection(
            hostname,
            port,
            timeout=_CALLBACK_TIMEOUT_SECONDS,
        )
        connection.request(
            "POST",
            path,
            body=body,
            headers={
                "Content-Type": "application/json",
                "Content-Length": str(len(body)),
                "X-vLLM-Model-Load-Token": token,
            },
        )
        response = connection.getresponse()
        response.read(1024)
        return response.status == 204
    except (OSError, TimeoutError, http.client.HTTPException):
        return False
    finally:
        if connection is not None:
            with suppress(OSError, http.client.HTTPException):
                connection.close()


@contextmanager
def model_load_progress(
    phase: str,
    detail: str,
    *,
    owner: object | None = None,
) -> Iterator[None]:
    rank, world_size = current_worker_coordinates(owner)
    emit_model_load_progress(
        phase,
        "started",
        detail,
        rank=rank,
        world_size=world_size,
    )
    try:
        yield
    except BaseException as error:
        emit_model_load_progress(
            phase,
            "failed",
            f"{detail} failed ({type(error).__name__})",
            rank=rank,
            world_size=world_size,
        )
        raise
    else:
        emit_model_load_progress(
            phase,
            "completed",
            detail,
            rank=rank,
            world_size=world_size,
        )


def track_model_load_phase(
    phase: str,
    detail: str,
    *,
    enabled_when: Callable[[Any], bool] | None = None,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    def decorate(function: Callable[P, R]) -> Callable[P, R]:
        @wraps(function)
        def wrapped(*args: P.args, **kwargs: P.kwargs) -> R:
            owner: Any = args[0] if args else None
            if enabled_when is not None:
                try:
                    enabled = enabled_when(owner)
                except Exception:
                    enabled = False
                if not enabled:
                    return function(*args, **kwargs)
            with model_load_progress(phase, detail, owner=owner):
                return function(*args, **kwargs)

        return wrapped

    return decorate
