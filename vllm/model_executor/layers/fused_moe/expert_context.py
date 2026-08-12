# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Resident, transactional expert-selection contexts for modular MoE routers."""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from typing import Any

import torch

_CONTEXT_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z")
_MAX_CONTEXTS = 64
_MAX_METADATA_BYTES = 8192
_frontend_context_id: str | None = None
_frontend_context_fingerprint: str | None = None
_frontend_context_transition = False


def set_frontend_expert_context(
    context_id: str | None,
    fingerprint: str | None,
) -> None:
    """Publish the committed context to frontend request admission."""
    global _frontend_context_fingerprint, _frontend_context_id
    _frontend_context_id = context_id
    _frontend_context_fingerprint = fingerprint


def set_frontend_expert_context_transition(active: bool) -> None:
    """Reject frontend admission while a global switch is in progress."""
    global _frontend_context_transition
    _frontend_context_transition = active


def get_frontend_expert_context() -> tuple[str | None, str | None, bool]:
    """Return context ID, fingerprint, and transition state for admission."""
    return (
        _frontend_context_id,
        _frontend_context_fingerprint,
        _frontend_context_transition,
    )


def _canonical_json(value: object) -> bytes:
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as error:
        raise ValueError("expert context data must be JSON serializable") from error
    return encoded.encode("ascii")


def _fingerprint(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _normalize_context_id(context_id: object) -> str:
    if not isinstance(context_id, str) or not _CONTEXT_ID_RE.fullmatch(context_id):
        raise ValueError(
            "context_id must contain 1-128 ASCII letters, digits, '.', '_', "
            "':' or '-', and start with a letter or digit"
        )
    return context_id


def _normalize_metadata(metadata: object) -> dict[str, Any]:
    if metadata is None:
        return {}
    if not isinstance(metadata, dict) or any(
        not isinstance(key, str) for key in metadata
    ):
        raise ValueError("expert context metadata must be a JSON object")
    if len(_canonical_json(metadata)) > _MAX_METADATA_BYTES:
        raise ValueError("expert context metadata exceeds 8192 bytes")
    return dict(metadata)


def _normalize_profile_layers(
    layers: object,
) -> dict[int, frozenset[int]]:
    if not isinstance(layers, Mapping):
        raise ValueError("expert context layers must be an object")
    normalized: dict[int, frozenset[int]] = {}
    for raw_layer_id, raw_layer in layers.items():
        if type(raw_layer_id) is int:
            layer_id = raw_layer_id
        elif (
            isinstance(raw_layer_id, str)
            and raw_layer_id.isascii()
            and raw_layer_id.isdigit()
        ):
            layer_id = int(raw_layer_id)
        else:
            raise ValueError(f"invalid layer ID {raw_layer_id!r}")
        if layer_id in normalized:
            raise ValueError(f"duplicate layer ID {layer_id}")
        if isinstance(raw_layer, Mapping):
            if set(raw_layer) != {"keep"}:
                raise ValueError(f"layer {layer_id} must contain only keep")
            keep = raw_layer["keep"]
        else:
            keep = raw_layer
        if (
            not isinstance(keep, Sequence)
            or isinstance(keep, str | bytes)
            or any(type(expert_id) is not int for expert_id in keep)
        ):
            raise ValueError(f"layer {layer_id} keep must be a list of integers")
        if len(keep) != len(set(keep)):
            raise ValueError(f"layer {layer_id} contains duplicate expert IDs")
        normalized[layer_id] = frozenset(keep)
    return normalized


@dataclass(frozen=True)
class ExpertContextSpec:
    """Unbound expert context received from a trusted control client."""

    context_id: str
    layers: dict[int, frozenset[int]]
    creation_source: str = "api"
    metadata: dict[str, Any] | None = None

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> ExpertContextSpec:
        allowed = {"context_id", "layers", "creation_source", "metadata"}
        if set(payload) - allowed:
            raise ValueError(
                "expert context contains unknown fields: "
                f"{sorted(set(payload) - allowed)}"
            )
        if "context_id" not in payload or "layers" not in payload:
            raise ValueError("expert context requires context_id and layers")
        creation_source = payload.get("creation_source", "api")
        if not isinstance(creation_source, str) or not 0 < len(creation_source) <= 128:
            raise ValueError("creation_source must contain 1-128 characters")
        return cls(
            context_id=_normalize_context_id(payload["context_id"]),
            layers=_normalize_profile_layers(payload["layers"]),
            creation_source=creation_source,
            metadata=_normalize_metadata(payload.get("metadata")),
        )


@dataclass(frozen=True)
class ExpertLayerTopology:
    """Runtime routing properties for one logical MoE layer."""

    layer_id: int
    num_experts: int
    top_k: int
    router_type: str
    routing_method: str
    supported: bool
    unsupported_reason: str | None

    def to_dict(self) -> dict[str, object]:
        return {
            "layer_id": self.layer_id,
            "num_experts": self.num_experts,
            "top_k": self.top_k,
            "router_type": self.router_type,
            "routing_method": self.routing_method,
            "supported": self.supported,
            "unsupported_reason": self.unsupported_reason,
        }


@dataclass(frozen=True)
class ExpertContextTopology:
    """Canonical topology discovered from the loaded model runners."""

    layers: tuple[ExpertLayerTopology, ...]
    model_identity: dict[str, object]
    fingerprint: str

    @property
    def supported(self) -> bool:
        return bool(self.layers) and all(layer.supported for layer in self.layers)

    @property
    def unsupported_reason(self) -> str | None:
        if not self.layers:
            return "model has no MoE runners"
        reasons = [
            f"layer {layer.layer_id}: {layer.unsupported_reason}"
            for layer in self.layers
            if not layer.supported
        ]
        return "; ".join(reasons) if reasons else None

    def to_dict(self) -> dict[str, object]:
        return {
            "fingerprint": self.fingerprint,
            "model_identity": dict(self.model_identity),
            "layers": [layer.to_dict() for layer in self.layers],
        }


@dataclass(frozen=True)
class ExpertContext:
    """A canonical full expert context bound to one model topology."""

    context_id: str
    layers: tuple[tuple[int, tuple[int, ...]], ...]
    profile_fingerprint: str
    topology_fingerprint: str
    fingerprint: str
    creation_source: str
    metadata: dict[str, Any]

    def keep_for_layer(self, layer_id: int) -> tuple[int, ...]:
        for candidate, keep in self.layers:
            if candidate == layer_id:
                return keep
        raise KeyError(layer_id)

    def to_dict(self) -> dict[str, object]:
        return {
            "context_id": self.context_id,
            "profile_fingerprint": self.profile_fingerprint,
            "topology_fingerprint": self.topology_fingerprint,
            "context_fingerprint": self.fingerprint,
            "creation_source": self.creation_source,
            "metadata": dict(self.metadata),
            "layers": {
                str(layer_id): {"keep": list(keep)} for layer_id, keep in self.layers
            },
        }

    @classmethod
    def bind(
        cls,
        spec: ExpertContextSpec,
        topology: ExpertContextTopology,
    ) -> ExpertContext:
        topology_by_id = {layer.layer_id: layer for layer in topology.layers}
        unknown = set(spec.layers) - set(topology_by_id)
        if unknown:
            raise ValueError(f"expert context has unknown layers: {sorted(unknown)}")

        full_layers: list[tuple[int, tuple[int, ...]]] = []
        for layer in topology.layers:
            keep = spec.layers.get(layer.layer_id, frozenset(range(layer.num_experts)))
            invalid = sorted(
                expert_id
                for expert_id in keep
                if not 0 <= expert_id < layer.num_experts
            )
            if invalid:
                raise ValueError(
                    f"layer {layer.layer_id} has out-of-range expert IDs: {invalid}"
                )
            if len(keep) < layer.top_k:
                raise ValueError(
                    f"layer {layer.layer_id} enables fewer experts than top_k"
                )
            full_layers.append((layer.layer_id, tuple(sorted(keep))))

        canonical_layers = [
            {"layer_id": layer_id, "keep": list(keep)} for layer_id, keep in full_layers
        ]
        profile_fingerprint = _fingerprint({"version": 1, "layers": canonical_layers})
        fingerprint = _fingerprint(
            {
                "version": 1,
                "profile_fingerprint": profile_fingerprint,
                "topology_fingerprint": topology.fingerprint,
            }
        )
        return cls(
            context_id=_normalize_context_id(spec.context_id),
            layers=tuple(full_layers),
            profile_fingerprint=profile_fingerprint,
            topology_fingerprint=topology.fingerprint,
            fingerprint=fingerprint,
            creation_source=spec.creation_source,
            metadata=_normalize_metadata(spec.metadata),
        )


def _discover_runners(model: torch.nn.Module) -> dict[int, Any]:
    from vllm.model_executor.layers.fused_moe.layer import MoERunner

    runners: dict[int, Any] = {}
    for module in model.modules():
        if not isinstance(module, MoERunner):
            continue
        if module.layer_id in runners:
            raise ValueError(f"model has duplicate MoE layer ID {module.layer_id}")
        runners[module.layer_id] = module
    return runners


def discover_expert_context_topology(
    model: torch.nn.Module,
    model_identity: Mapping[str, object] | None = None,
) -> ExpertContextTopology:
    """Discover actual loaded routing paths and their hot-switch support."""
    from vllm.model_executor.layers.fused_moe.router.base_router import BaseRouter

    identity = dict(model_identity or {})
    _canonical_json(identity)
    layers: list[ExpertLayerTopology] = []
    for layer_id, runner in sorted(_discover_runners(model).items()):
        router = runner.router
        num_experts = runner.moe_config.num_logical_experts
        top_k = getattr(router, "top_k", None)
        if top_k is None:
            top_k = runner.moe_config.top_k
        router_type = type(router).__name__
        routing_method = "unknown"
        with suppress(AttributeError, NotImplementedError, ValueError):
            routing_method = router.routing_method_type.name

        reason: str | None = None
        if runner._quant_method.is_monolithic:
            reason = "monolithic MoE routing path"
        elif not isinstance(router, BaseRouter):
            reason = f"router {router_type} is not a BaseRouter"
        elif not router.supports_expert_eligibility:
            reason = f"expert eligibility is unsupported for {router_type}"
        else:
            try:
                baseline = torch.ones(
                    num_experts,
                    dtype=torch.bool,
                    device=runner.moe_config.device,
                )
                router.validate_expert_eligibility_mask(
                    baseline, logical_num_experts=num_experts
                )
            except (RuntimeError, ValueError) as error:
                reason = str(error)
        layers.append(
            ExpertLayerTopology(
                layer_id=layer_id,
                num_experts=num_experts,
                top_k=top_k,
                router_type=router_type,
                routing_method=routing_method,
                supported=reason is None,
                unsupported_reason=reason,
            )
        )

    topology_data = {
        "version": 1,
        "model_identity": identity,
        "layers": [layer.to_dict() for layer in layers],
    }
    return ExpertContextTopology(
        layers=tuple(layers),
        model_identity=identity,
        fingerprint=_fingerprint(topology_data),
    )


class ExpertContextController:
    """Own resident contexts and apply them as a two-phase transaction."""

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        model_identity: Mapping[str, object] | None = None,
        synchronize: Callable[[], None] | None = None,
        initialize: bool = True,
    ) -> None:
        self.model = model
        self.runners = _discover_runners(model)
        self.topology = discover_expert_context_topology(model, model_identity)
        self._synchronize_fn = synchronize
        self._contexts: dict[str, ExpertContext] = {}
        self._prepared_context: ExpertContext | None = None
        self._prepared_masks: dict[int, torch.Tensor] | None = None
        self._transaction_previous: ExpertContext | None = None
        self._stable_initialized = False

        baseline = self.register_context(
            ExpertContextSpec(
                context_id="baseline",
                layers={},
                creation_source="builtin",
                metadata={"description": "All routed experts are eligible."},
            )
        )
        self._active_context = baseline
        if initialize and self.topology.supported:
            self.initialize_stable_buffers()

    @property
    def supported(self) -> bool:
        return self.topology.supported

    @property
    def unsupported_reason(self) -> str | None:
        return self.topology.unsupported_reason

    @property
    def active_context(self) -> ExpertContext:
        return self._active_context

    def register_context(self, spec: ExpertContextSpec) -> ExpertContext:
        context = ExpertContext.bind(spec, self.topology)
        existing = self._contexts.get(context.context_id)
        if existing is not None:
            if existing.fingerprint != context.fingerprint:
                raise ValueError(
                    f"context ID {context.context_id!r} is already registered "
                    "with a different fingerprint"
                )
            return existing
        if len(self._contexts) >= _MAX_CONTEXTS:
            raise ValueError(f"at most {_MAX_CONTEXTS} expert contexts may be resident")
        self._contexts[context.context_id] = context
        return context

    def register_payload(self, payload: Mapping[str, object]) -> ExpertContext:
        return self.register_context(ExpertContextSpec.from_dict(payload))

    def register_profile(
        self,
        context_id: str,
        layers: Mapping[int, frozenset[int]],
        *,
        creation_source: str,
        metadata: Mapping[str, object] | None = None,
    ) -> ExpertContext:
        return self.register_context(
            ExpertContextSpec(
                context_id=_normalize_context_id(context_id),
                layers=dict(layers),
                creation_source=creation_source,
                metadata=_normalize_metadata(dict(metadata or {})),
            )
        )

    def _build_masks(self, context: ExpertContext) -> dict[int, torch.Tensor]:
        if not self.supported:
            raise ValueError(self.unsupported_reason or "hot switching is unsupported")
        masks: dict[int, torch.Tensor] = {}
        for layer in self.topology.layers:
            runner = self.runners[layer.layer_id]
            mask = torch.zeros(
                layer.num_experts,
                dtype=torch.bool,
                device=runner.moe_config.device,
            )
            mask[list(context.keep_for_layer(layer.layer_id))] = True
            masks[layer.layer_id] = mask
        for layer in self.topology.layers:
            runner = self.runners[layer.layer_id]
            runner.router.validate_expert_eligibility_mask(
                masks[layer.layer_id], logical_num_experts=layer.num_experts
            )
        return masks

    def _apply_masks(self, masks: Mapping[int, torch.Tensor]) -> None:
        for layer in self.topology.layers:
            self.runners[layer.layer_id].router.set_expert_eligibility_mask(
                masks[layer.layer_id], logical_num_experts=layer.num_experts
            )

    def _synchronize(self) -> None:
        if self._synchronize_fn is not None:
            self._synchronize_fn()
            return
        if any(mask.device.type != "cpu" for mask in self._current_masks()):
            torch.accelerator.synchronize()

    def _current_masks(self) -> list[torch.Tensor]:
        return [
            runner.router.expert_eligibility_mask
            for runner in self.runners.values()
            if runner.router.expert_eligibility_mask is not None
        ]

    def initialize_stable_buffers(self) -> None:
        if self._stable_initialized:
            return
        masks = self._build_masks(self._contexts["baseline"])
        self._apply_masks(masks)
        self._synchronize()
        self._stable_initialized = True

    def prepare_context(self, context_id: str) -> ExpertContext:
        self._prepared_context = None
        self._prepared_masks = None
        self._transaction_previous = None
        if not self._stable_initialized:
            raise RuntimeError("expert context buffers are not initialized")
        context = self._contexts.get(context_id)
        if context is None:
            raise ValueError(f"unknown expert context {context_id!r}")
        masks = self._build_masks(context)
        self._prepared_context = context
        self._prepared_masks = masks
        self._transaction_previous = self._active_context
        return context

    def commit_context(self) -> ExpertContext:
        if self._prepared_context is None or self._prepared_masks is None:
            raise RuntimeError("no expert context has been prepared")
        target = self._prepared_context
        previous = self._transaction_previous
        assert previous is not None
        previous_masks = self._build_masks(previous)
        try:
            self._apply_masks(self._prepared_masks)
            self._synchronize()
        except Exception:
            self._apply_masks(previous_masks)
            self._synchronize()
            self._active_context = previous
            self._prepared_context = None
            self._prepared_masks = None
            raise
        self._active_context = target
        self._prepared_context = None
        self._prepared_masks = None
        return target

    def rollback_context(self) -> ExpertContext:
        previous = self._transaction_previous
        if previous is None:
            self._prepared_context = None
            self._prepared_masks = None
            return self._active_context
        if self._active_context.fingerprint != previous.fingerprint:
            self._apply_masks(self._build_masks(previous))
            self._synchronize()
            self._active_context = previous
        self._prepared_context = None
        self._prepared_masks = None
        self._transaction_previous = None
        return self._active_context

    def adopt_startup_context(self, context_id: str) -> ExpertContext:
        """Record a cold-start profile when hot switching is unsupported."""
        context = self._contexts.get(context_id)
        if context is None:
            raise ValueError(f"unknown expert context {context_id!r}")
        self._active_context = context
        return context

    def current(self) -> dict[str, object]:
        context = self._active_context
        return {
            "active_context_id": context.context_id,
            "active_context_fingerprint": context.fingerprint,
            "profile_fingerprint": context.profile_fingerprint,
            "topology_fingerprint": self.topology.fingerprint,
            "process_id": os.getpid(),
            "weights_reloaded": False,
        }

    def capabilities(self) -> dict[str, object]:
        current = self.current()
        return {
            "supported": self.supported,
            "unsupported_reason": self.unsupported_reason,
            **current,
            "layers": [layer.to_dict() for layer in self.topology.layers],
        }
