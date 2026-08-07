# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


class _DuplicateKeyError(ValueError):
    pass


def _object_from_unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateKeyError(
                f"duplicate key {key!r} in expert selection profile"
            )
        result[key] = value
    return result


@dataclass(frozen=True)
class ExpertSelectionProfile:
    """Canonical per-layer sets of eligible checkpoint expert IDs."""

    layers: dict[int, frozenset[int]]

    @classmethod
    def from_file(cls, path: str | Path) -> ExpertSelectionProfile:
        try:
            with open(path, encoding="utf-8") as file:
                data = json.load(file, object_pairs_hook=_object_from_unique_pairs)
        except OSError as error:
            raise ValueError(
                f"cannot read expert selection profile {path!s}: {error}"
            ) from error
        except json.JSONDecodeError as error:
            raise ValueError(f"malformed expert selection profile: {error}") from error
        if not isinstance(data, dict):
            raise ValueError("expert selection profile must be an object")
        if set(data) != {"version", "layers"}:
            raise ValueError(
                "expert selection profile must contain only version and layers"
            )
        if type(data["version"]) is not int or data["version"] != 1:
            raise ValueError("expert selection profile must have version 1")
        layers = data["layers"]
        if not isinstance(layers, dict) or not layers:
            raise ValueError(
                "expert selection profile must contain a non-empty layers object"
            )
        normalized: dict[int, frozenset[int]] = {}
        for raw_layer, layer_config in layers.items():
            if (
                not isinstance(raw_layer, str)
                or not raw_layer.isascii()
                or not raw_layer.isdigit()
            ):
                raise ValueError(f"invalid layer ID {raw_layer!r}")
            layer_id = int(raw_layer)
            if layer_id in normalized:
                raise ValueError(f"duplicate layer ID {layer_id}")
            if not isinstance(layer_config, dict) or set(layer_config) != {"keep"}:
                raise ValueError(f"layer {raw_layer} must contain only keep")
            keep = layer_config["keep"]
            if not isinstance(keep, list) or any(
                type(item) is not int for item in keep
            ):
                raise ValueError(f"layer {raw_layer} keep must be a list of integers")
            if len(keep) != len(set(keep)):
                raise ValueError(f"layer {raw_layer} contains duplicate expert IDs")
            normalized[layer_id] = frozenset(keep)
        return cls(normalized)


def bind_expert_selection_profile(
    model: torch.nn.Module, profile: ExpertSelectionProfile
) -> None:
    """Validate and attach a profile to modular MoE routers in ``model``."""
    from vllm.model_executor.layers.fused_moe.layer import MoERunner
    from vllm.model_executor.layers.fused_moe.router.base_router import BaseRouter

    runners: dict[int, Any] = {}
    for module in model.modules():
        if not isinstance(module, MoERunner):
            continue
        if module.layer_id in runners:
            raise ValueError(f"model has duplicate MoE layer ID {module.layer_id}")
        runners[module.layer_id] = module
    unknown = set(profile.layers) - set(runners)
    if unknown:
        raise ValueError(
            f"expert selection profile has unknown layers: {sorted(unknown)}"
        )
    for layer_id, keep in profile.layers.items():
        runner = runners[layer_id]
        if runner._quant_method.is_monolithic or not isinstance(
            runner.router, BaseRouter
        ):
            raise ValueError(
                f"expert selection is unsupported for layer {layer_id} routing path"
            )
        router = runner.router
        logical_num_experts = runner.moe_config.num_logical_experts
        invalid = [
            expert_id for expert_id in keep if not 0 <= expert_id < logical_num_experts
        ]
        if invalid:
            raise ValueError(
                f"layer {layer_id} has out-of-range expert IDs: {sorted(invalid)}"
            )
        if len(keep) < router.top_k:
            raise ValueError(f"layer {layer_id} enables fewer experts than top_k")
        mask = torch.zeros(
            logical_num_experts,
            dtype=torch.bool,
            device=runner.moe_config.device,
        )
        mask[list(keep)] = True
        router.set_expert_eligibility_mask(
            mask, logical_num_experts=logical_num_experts
        )
