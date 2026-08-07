# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import types
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch

from vllm.config import ModelConfig, VllmConfig
from vllm.config.compilation import CompilationMode
from vllm.distributed.eplb.eplb_state import EplbLayerState
from vllm.model_executor.layers.fused_moe.config import RoutingMethodType
from vllm.model_executor.layers.fused_moe.expert_selection import (
    ExpertSelectionProfile,
    bind_expert_selection_profile,
)
from vllm.model_executor.layers.fused_moe.routed_experts_capturer import (
    RoutedExpertsCapturer,
    RoutedExpertsManager,
    bind_routed_experts_capturer,
    get_routed_experts_attn_gid,
)
from vllm.model_executor.layers.fused_moe.router.base_router import BaseRouter
from vllm.model_executor.layers.fused_moe.router.custom_routing_router import (
    CustomRoutingRouter,
)
from vllm.model_executor.layers.fused_moe.router.fused_topk_bias_router import (
    FusedTopKBiasRouter,
)
from vllm.model_executor.layers.fused_moe.router.grouped_topk_router import (
    GroupedTopKRouter,
)
from vllm.model_executor.layers.fused_moe.router.routing_simulator_router import (
    RoutingSimulatorRouter,
)
from vllm.model_executor.layers.fused_moe.router.zero_expert_router import (
    ZeroExpertRouter,
)
from vllm.transformers_utils.model_arch_config_convertor import (
    ModelArchConfigConvertorBase,
)
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
)

pytestmark = pytest.mark.cpu_test

_REC_MODULE = "vllm.model_executor.layers.fused_moe.routed_experts_capturer"


def _capturer_with_buffer(
    *,
    max_tokens: int = 8,
    num_layers: int = 4,
    num_experts_per_tok: int = 2,
    dp_rank: int = 0,
    tp_size: int = 1,
    capture_weights: bool = False,
) -> RoutedExpertsCapturer:
    # Bypass __init__ so the test can use a CPU buffer and skip the
    # VllmConfig dependency. The CUDA device-tensor allocation in the
    # real constructor is not what we are exercising here.
    c = RoutedExpertsCapturer.__new__(RoutedExpertsCapturer)
    c.dp_rank = dp_rank
    c.tp_size = tp_size
    c.device_buffer = torch.full(
        (max_tokens, num_layers, num_experts_per_tok),
        -1,
        dtype=torch.int32,
    )
    c.capture_weights = capture_weights
    c.weight_device_buffer = (
        torch.full(
            (max_tokens, num_layers, num_experts_per_tok),
            torch.nan,
            dtype=torch.float32,
        )
        if capture_weights
        else None
    )
    return c


class DummyRouter(BaseRouter):
    supports_expert_eligibility = True

    @property
    def routing_method_type(self) -> RoutingMethodType:
        return RoutingMethodType.FUSED_TOPK

    def _compute_routing(
        self, hidden_states, router_logits, indices_type, *, input_ids=None
    ):
        topk_ids = torch.tensor([[1, 2], [3, 4]], dtype=torch.int64)
        topk_weights = torch.ones_like(topk_ids, dtype=torch.float32)
        return topk_weights, topk_ids

    def _apply_eplb_mapping(self, topk_ids: torch.Tensor) -> torch.Tensor:
        # Make mapping observable without requiring CUDA EPLB path.
        return topk_ids + 10


def _make_router(eplb_state: EplbLayerState | None = None) -> DummyRouter:
    return DummyRouter(
        top_k=2,
        global_num_experts=16,
        eplb_state=eplb_state,
    )


def _make_modular_routed_experts():
    return types.SimpleNamespace(
        quant_method=types.SimpleNamespace(is_monolithic=False),
    )


def _make_model_config(hf_config):
    num_experts_per_token = ModelArchConfigConvertorBase(
        hf_config, hf_config
    ).get_num_experts_per_token()
    model_config = SimpleNamespace(
        hf_text_config=hf_config,
        model_arch_config=SimpleNamespace(
            num_experts_per_token=num_experts_per_token,
        ),
    )
    model_config.get_num_experts = lambda: hf_config.num_experts
    model_config.get_num_experts_per_tok = lambda: (
        ModelConfig.get_num_experts_per_tok(model_config)
    )
    model_config.get_total_num_hidden_layers = lambda: hf_config.num_hidden_layers
    return model_config


def test_routed_experts_manager_uses_gemma4_top_k_experts():
    hf_config = SimpleNamespace(
        num_experts=8,
        top_k_experts=2,
        num_hidden_layers=3,
    )
    vllm_config = SimpleNamespace(model_config=_make_model_config(hf_config))
    kv_cache_spec = FullAttentionSpec(
        block_size=4,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.float32,
    )
    kv_cache_config = KVCacheConfig(
        num_blocks=2,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec(["layer"], kv_cache_spec)],
    )

    manager = RoutedExpertsManager(vllm_config, kv_cache_config)

    assert manager.routed_experts_by_slot.shape == (8, 3, 2)
    assert manager.routed_expert_weights_by_slot is None


def test_routed_experts_manager_uses_kimi_k3_experts_per_token():
    hf_config = SimpleNamespace(
        num_experts=8,
        num_experts_per_token=2,
        num_hidden_layers=3,
    )
    vllm_config = SimpleNamespace(model_config=_make_model_config(hf_config))
    kv_cache_spec = FullAttentionSpec(
        block_size=4,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.float32,
    )
    kv_cache_config = KVCacheConfig(
        num_blocks=2,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec(["layer"], kv_cache_spec)],
    )

    manager = RoutedExpertsManager(vllm_config, kv_cache_config)

    assert manager.routed_experts_by_slot.shape == (8, 3, 2)


def test_routed_experts_manager_stores_paired_weights_by_slot():
    hf_config = SimpleNamespace(
        num_experts=8,
        num_experts_per_token=2,
        num_hidden_layers=3,
    )
    model_config = _make_model_config(hf_config)
    model_config.enable_return_routed_expert_weights = True
    vllm_config = SimpleNamespace(model_config=model_config)
    kv_cache_spec = FullAttentionSpec(
        block_size=4,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.float32,
    )
    kv_cache_config = KVCacheConfig(
        num_blocks=2,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec(["layer"], kv_cache_spec)],
    )
    manager = RoutedExpertsManager(vllm_config, kv_cache_config)
    ids = np.arange(12, dtype=np.int32).reshape(2, 3, 2)
    weights = np.linspace(0.1, 0.9, 12, dtype=np.float32).reshape(2, 3, 2)

    manager.store_batch(ids, np.array([1, 5]), weights)

    assert manager.routed_expert_weights_by_slot is not None
    np.testing.assert_array_equal(manager.routed_experts_by_slot[[1, 5]], ids)
    np.testing.assert_array_equal(
        manager.routed_expert_weights_by_slot[[1, 5]], weights
    )
    np.testing.assert_array_equal(manager.get([0], 2, token_start=1), ids[:1])
    np.testing.assert_array_equal(
        manager.get_weights([0], 2, token_start=1), weights[:1]
    )


def test_base_router_capture_pre_eplb_mapping():
    router = _make_router()
    captured = []

    def capture_fn(ids):
        captured.append(ids.clone())

    router.set_capture_fn(capture_fn)
    topk_weights, topk_ids = router.select_experts(
        hidden_states=torch.empty(1),
        router_logits=torch.empty(1),
    )

    assert topk_weights.shape == topk_ids.shape
    assert len(captured) == 1
    assert torch.equal(captured[0], torch.tensor([[1, 2], [3, 4]]))
    assert torch.equal(topk_ids, torch.tensor([[11, 12], [13, 14]]))


def test_base_router_captures_paired_weights_pre_eplb_mapping():
    router = _make_router()
    captured = []

    def capture_fn(ids, weights):
        captured.append((ids.clone(), weights.clone()))

    router.set_capture_weights_fn(capture_fn)
    topk_weights, topk_ids = router.select_experts(
        hidden_states=torch.empty(1),
        router_logits=torch.empty(1),
    )

    assert len(captured) == 1
    captured_ids, captured_weights = captured[0]
    assert torch.equal(captured_ids, torch.tensor([[1, 2], [3, 4]]))
    assert torch.equal(captured_weights, torch.ones((2, 2)))
    assert torch.equal(captured_weights, topk_weights)
    assert torch.equal(topk_ids, torch.tensor([[11, 12], [13, 14]]))


def test_base_router_capture_with_eplb_enabled():
    eplb_state = EplbLayerState()
    eplb_state.expert_load_view = torch.zeros(32, dtype=torch.int64)
    eplb_state.logical_to_physical_map = torch.arange(32).view(32, 1)
    eplb_state.logical_replica_count = torch.ones(32, dtype=torch.int64)
    eplb_state.should_record_tensor = torch.ones((), dtype=torch.bool)
    eplb_state.num_unpadded_tokens_tensors = [torch.tensor(0, dtype=torch.int32)]
    router = _make_router(eplb_state=eplb_state)

    captured = []

    def capture_fn(ids):
        captured.append(ids.clone())

    router.set_capture_fn(capture_fn)
    _, topk_ids = router.select_experts(
        hidden_states=torch.empty(1),
        router_logits=torch.empty(1),
    )

    assert len(captured) == 1
    # Capture should see logical ids pre-EPLB mapping.
    assert torch.equal(captured[0], torch.tensor([[1, 2], [3, 4]]))
    # Our DummyRouter mapping adds +10.
    assert torch.equal(topk_ids, torch.tensor([[11, 12], [13, 14]]))


def test_base_router_eligibility_is_applied_before_topk():
    class TopKRouter(DummyRouter):
        def _compute_routing(
            self, hidden_states, router_logits, indices_type, *, input_ids=None
        ):
            return torch.topk(router_logits, self.top_k)

        def _apply_eplb_mapping(self, topk_ids):
            return topk_ids

    router = TopKRouter(top_k=2, global_num_experts=4)
    router.set_expert_eligibility_mask(torch.tensor([True, True, False, False]))
    weights, ids = router.select_experts(
        hidden_states=torch.empty(1),
        router_logits=torch.tensor([[1.0, 2.0, 100.0, 200.0]]),
    )

    assert ids.tolist() == [[1, 0]]
    assert weights.tolist() == [[2.0, 1.0]]


def test_base_router_rejects_too_few_eligible_experts():
    router = _make_router()

    with pytest.raises(ValueError, match="fewer than top_k"):
        router.set_expert_eligibility_mask(
            torch.tensor([True] + [False] * 15, dtype=torch.bool)
        )


def test_expert_selection_profile_normalizes(tmp_path):
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(
        '{"version": 1, "layers": {"2": {"keep": [0, 3]}}}',
        encoding="utf-8",
    )
    profile = ExpertSelectionProfile.from_file(profile_path)
    assert profile.layers == {2: frozenset({0, 3})}


@pytest.mark.parametrize(
    ("contents", "error"),
    [
        ("not json", "malformed"),
        ("[]", "must be an object"),
        ('{"version": true, "layers": {"0": {"keep": [0]}}}', "version 1"),
        ('{"version": 2, "layers": {"0": {"keep": [0]}}}', "version 1"),
        ('{"version": 1}', "only version and layers"),
        ('{"version": 1, "layers": {}, "extra": 1}', "only version and layers"),
        ('{"version": 1, "layers": {}}', "non-empty layers object"),
        (
            '{"version": 1, "layers": {"bad": {"keep": [0]}}}',
            "invalid layer ID",
        ),
        (
            '{"version": 1, "layers": {"1": {"keep": [0]}, "01": {"keep": [1]}}}',
            "duplicate layer ID",
        ),
        (
            '{"version": 1, "layers": {"1": {"keep": [0]}, "1": {"keep": [1]}}}',
            "duplicate key",
        ),
        (
            '{"version": 1, "layers": {"2": {"keep": [0], "drop": [1]}}}',
            "contain only keep",
        ),
        (
            '{"version": 1, "layers": {"2": {"keep": true}}}',
            "list of integers",
        ),
        (
            '{"version": 1, "layers": {"2": {"keep": [true]}}}',
            "list of integers",
        ),
        (
            '{"version": 1, "layers": {"2": {"keep": [3, 3]}}}',
            "duplicate expert IDs",
        ),
    ],
)
def test_expert_selection_profile_rejects_invalid_schema(tmp_path, contents, error):
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(contents, encoding="utf-8")

    with pytest.raises(ValueError, match=error):
        ExpertSelectionProfile.from_file(profile_path)


def test_expert_selection_profile_rejects_missing_file(tmp_path):
    with pytest.raises(ValueError, match="cannot read expert selection profile"):
        ExpertSelectionProfile.from_file(tmp_path / "missing.json")


def _bind_profile_to_router(
    monkeypatch,
    router,
    *,
    keep=frozenset({0, 1}),
    layer_id=2,
    logical_num_experts=4,
    is_monolithic=False,
):
    class DummyMoERunner:
        def __init__(self):
            self.layer_id = layer_id
            self.router = router
            self._quant_method = SimpleNamespace(is_monolithic=is_monolithic)
            self.moe_config = SimpleNamespace(
                num_logical_experts=logical_num_experts,
                device=torch.device("cpu"),
            )

    import vllm.model_executor.layers.fused_moe.layer as fused_moe_layer

    monkeypatch.setattr(fused_moe_layer, "MoERunner", DummyMoERunner)
    runner = DummyMoERunner()
    model = SimpleNamespace(modules=lambda: [runner])
    profile = ExpertSelectionProfile({layer_id: keep})
    bind_expert_selection_profile(model, profile)
    return runner


def test_profile_binding_retains_device_resident_logical_mask(monkeypatch):
    router = DummyRouter(top_k=2, global_num_experts=6)

    _bind_profile_to_router(monkeypatch, router, keep=frozenset({0, 3}))

    mask = router.expert_eligibility_mask
    assert mask is not None
    assert mask.device == torch.device("cpu")
    assert mask.tolist() == [True, False, False, True]
    assert router._expert_ineligibility_mask is not None
    mask_identity = id(mask)
    router.select_experts(torch.empty(1), torch.zeros(1, 4))
    router.select_experts(torch.empty(1), torch.zeros(1, 4))
    assert id(router.expert_eligibility_mask) == mask_identity


@pytest.mark.parametrize(
    ("keep", "layer_id", "logical_num_experts", "error"),
    [
        (frozenset({0, 1}), 5, 4, "unknown layers"),
        (frozenset({0, 4}), 2, 4, "out-of-range expert IDs"),
        (frozenset({0}), 2, 4, "fewer experts than top_k"),
    ],
)
def test_profile_binding_rejects_invalid_model_references(
    monkeypatch, keep, layer_id, logical_num_experts, error
):
    router = DummyRouter(top_k=2, global_num_experts=logical_num_experts)

    if layer_id == 5:

        class DummyMoERunner:
            def __init__(self):
                self.layer_id = 2
                self.router = router
                self._quant_method = SimpleNamespace(is_monolithic=False)
                self.moe_config = SimpleNamespace(
                    num_logical_experts=logical_num_experts,
                    device=torch.device("cpu"),
                )

        import vllm.model_executor.layers.fused_moe.layer as fused_moe_layer

        monkeypatch.setattr(fused_moe_layer, "MoERunner", DummyMoERunner)
        model = SimpleNamespace(modules=lambda: [DummyMoERunner()])
        profile = ExpertSelectionProfile({layer_id: keep})
        with pytest.raises(ValueError, match=error):
            bind_expert_selection_profile(model, profile)
    else:
        with pytest.raises(ValueError, match=error):
            _bind_profile_to_router(
                monkeypatch,
                router,
                keep=keep,
                layer_id=layer_id,
                logical_num_experts=logical_num_experts,
            )


def test_profile_binding_rejects_monolithic_and_unsupported_routers(monkeypatch):
    router = DummyRouter(top_k=2, global_num_experts=4)
    with pytest.raises(ValueError, match="unsupported.*routing path"):
        _bind_profile_to_router(monkeypatch, router, is_monolithic=True)

    simulator = RoutingSimulatorRouter(top_k=2, global_num_experts=4)
    with pytest.raises(ValueError, match="unsupported for RoutingSimulatorRouter"):
        _bind_profile_to_router(monkeypatch, simulator)


def test_unsupported_modular_routing_paths_fail_closed():
    custom = CustomRoutingRouter(
        top_k=2,
        global_num_experts=4,
        custom_routing_function=lambda **kwargs: (None, None),
    )
    with pytest.raises(ValueError, match="unsupported for CustomRoutingRouter"):
        custom.set_expert_eligibility_mask(torch.ones(4, dtype=torch.bool))

    zero_expert = ZeroExpertRouter(
        top_k=2,
        global_num_experts=4,
        e_score_correction_bias=torch.zeros(4),
        num_logical_experts=4,
        zero_expert_type="identity",
    )
    with pytest.raises(ValueError, match="unsupported for ZeroExpertRouter"):
        zero_expert.set_expert_eligibility_mask(torch.ones(4, dtype=torch.bool))

    hash_router = FusedTopKBiasRouter(
        top_k=2,
        global_num_experts=4,
        hash_indices_table=torch.zeros((2, 2), dtype=torch.int32),
    )
    with pytest.raises(ValueError, match="unsupported for hash-based routing"):
        hash_router.set_expert_eligibility_mask(torch.ones(4, dtype=torch.bool))


def test_correction_bias_cannot_readmit_disabled_expert(monkeypatch):
    correction_bias = torch.tensor([0.0, 0.0, 100.0, 100.0])
    router = FusedTopKBiasRouter(
        top_k=2,
        global_num_experts=4,
        e_score_correction_bias=correction_bias,
        scoring_func="sigmoid",
    )

    def fake_fused_topk_bias(**kwargs):
        scores = kwargs["gating_output"].sigmoid()
        selection_scores = scores + kwargs["e_score_correction_bias"]
        topk_ids = selection_scores.topk(kwargs["topk"], dim=-1).indices
        return scores.gather(1, topk_ids), topk_ids

    monkeypatch.setattr(
        "vllm.model_executor.layers.fused_moe.router."
        "fused_topk_bias_router.fused_topk_bias",
        fake_fused_topk_bias,
    )
    router.set_expert_eligibility_mask(torch.tensor([True, True, False, False]))
    weights, ids = router.select_experts(
        torch.empty(1, 1), torch.tensor([[1.0, 2.0, 100.0, 200.0]])
    )

    assert set(ids[0].tolist()) == {0, 1}
    assert torch.all(weights >= 0)
    assert correction_bias.tolist() == [0.0, 0.0, 100.0, 100.0]


def test_grouped_router_masks_correction_bias_and_rejects_unsafe_grouping():
    correction_bias = torch.arange(8, dtype=torch.float32)
    router = GroupedTopKRouter(
        top_k=2,
        global_num_experts=8,
        num_expert_group=2,
        topk_group=1,
        e_score_correction_bias=correction_bias,
    )
    router.set_expert_eligibility_mask(
        torch.tensor([True, True, False, False, True, True, False, False])
    )
    assert router._eligibility_correction_bias is not None
    assert router._eligibility_correction_bias.tolist() == [
        0.0,
        1.0,
        float("-inf"),
        float("-inf"),
        4.0,
        5.0,
        float("-inf"),
        float("-inf"),
    ]

    unsafe_router = GroupedTopKRouter(
        top_k=4,
        global_num_experts=8,
        num_expert_group=4,
        topk_group=2,
    )
    with pytest.raises(ValueError, match="cannot guarantee top_k"):
        unsafe_router.set_expert_eligibility_mask(torch.tensor([True, False] * 4))


def test_no_profile_and_all_experts_profile_match_baseline():
    class TopKRouter(DummyRouter):
        def _compute_routing(
            self, hidden_states, router_logits, indices_type, *, input_ids=None
        ):
            return torch.topk(router_logits, self.top_k)

        def _apply_eplb_mapping(self, topk_ids):
            return topk_ids

    logits = torch.tensor([[1.0, 4.0, 3.0, 2.0]])
    baseline_router = TopKRouter(top_k=2, global_num_experts=4)
    baseline = baseline_router.select_experts(torch.empty(1), logits.clone())
    assert baseline_router.expert_eligibility_mask is None

    profile_router = TopKRouter(top_k=2, global_num_experts=4)
    profile_router.set_expert_eligibility_mask(torch.ones(4, dtype=torch.bool))
    profiled = profile_router.select_experts(torch.empty(1), logits.clone())

    assert torch.equal(profiled[0], baseline[0])
    assert torch.equal(profiled[1], baseline[1])


def test_public_binding_only_visits_target_model(monkeypatch):
    class DummyFusedMoE:
        def __init__(self, layer_id):
            self.layer_id = layer_id
            self.router = _make_router()
            self._quant_method = _make_modular_routed_experts().quant_method

    target_module = DummyFusedMoE(layer_id=7)
    draft_module = DummyFusedMoE(layer_id=0)

    import vllm.model_executor.layers.fused_moe.layer as fused_moe_layer

    monkeypatch.setattr(fused_moe_layer, "MoERunner", DummyFusedMoE)
    calls = []
    capturer = types.SimpleNamespace(capture=lambda *args: calls.append(args))

    bind_routed_experts_capturer(
        types.SimpleNamespace(modules=lambda: [target_module]), capturer
    )

    assert target_module.router.capture_fn is not None
    assert draft_module.router.capture_fn is None
    topk_ids = torch.tensor([[5, 6]])
    target_module.router.capture_fn(topk_ids)
    assert calls == [(7, topk_ids)]


def test_public_binding_uses_paired_callback_when_weights_enabled(monkeypatch):
    class DummyFusedMoE:
        def __init__(self, layer_id):
            self.layer_id = layer_id
            self.router = _make_router()
            self._quant_method = _make_modular_routed_experts().quant_method

    module = DummyFusedMoE(layer_id=7)
    import vllm.model_executor.layers.fused_moe.layer as fused_moe_layer

    monkeypatch.setattr(fused_moe_layer, "MoERunner", DummyFusedMoE)
    calls = []
    capturer = types.SimpleNamespace(
        capture_weights=True,
        capture=lambda *args: calls.append(args),
    )

    bind_routed_experts_capturer(
        types.SimpleNamespace(modules=lambda: [module]), capturer
    )

    assert module.router.capture_fn is None
    assert module.router.capture_weights_fn is not None
    topk_ids = torch.tensor([[5, 6]])
    topk_weights = torch.tensor([[0.75, 0.25]])
    module.router.capture_weights_fn(topk_ids, topk_weights)
    assert calls == [(7, topk_ids, topk_weights)]


def test_public_binding_rejects_monolithic_without_replay_support(monkeypatch):
    class DummyFusedMoE:
        def __init__(self):
            self.layer_id = 3
            self.router = _make_router()
            # Use a concrete monolithic expert and override its capability
            # instead of instantiating the abstract base class directly.
            from vllm.model_executor.layers.fused_moe.experts.cpu_moe import (
                CPUExpertsFp8,
            )

            fused_experts = CPUExpertsFp8.__new__(CPUExpertsFp8)
            self.routed_experts = types.SimpleNamespace(
                quant_method=types.SimpleNamespace(
                    is_monolithic=True,
                    moe_kernel=types.SimpleNamespace(
                        impl=types.SimpleNamespace(fused_experts=fused_experts)
                    ),
                )
            )
            self._quant_method = self.routed_experts.quant_method
            self._quant_method.moe_kernel.impl.fused_experts = fused_experts
            fused_experts.supports_routing_replay_capture = lambda: False

    class DummyCapturer:
        def capture(self, layer_id, topk_ids):
            pass

    dummy_module = DummyFusedMoE()
    import vllm.model_executor.layers.fused_moe.layer as fused_moe_layer

    monkeypatch.setattr(fused_moe_layer, "MoERunner", DummyFusedMoE)

    with pytest.raises(ValueError, match="monolithic MoE kernel"):
        bind_routed_experts_capturer(
            types.SimpleNamespace(modules=lambda: [dummy_module]), DummyCapturer()
        )


def test_routed_experts_capturer_single_dp_no_metadata():
    """dp_metadata is None: capture writes the full topk_ids rows."""
    capturer = _capturer_with_buffer(dp_rank=0)
    topk = torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.int32)
    ctx = SimpleNamespace(dp_metadata=None)
    with patch(f"{_REC_MODULE}.get_forward_context", return_value=ctx):
        capturer.capture(layer_id=0, topk_ids=topk)
    assert torch.equal(capturer.device_buffer[:3, 0, :], topk)
    assert capturer.device_buffer[3, 0, 0].item() == -1


def test_routed_experts_capturer_preserves_id_weight_pairing():
    capturer = _capturer_with_buffer(dp_rank=0, capture_weights=True)
    topk_ids = torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.int32)
    topk_weights = torch.tensor(
        [[0.6, 0.4], [0.75, 0.25], [0.9, 0.1]], dtype=torch.float16
    )
    ctx = SimpleNamespace(dp_metadata=None)

    with patch(f"{_REC_MODULE}.get_forward_context", return_value=ctx):
        capturer.capture(
            layer_id=0,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
        )

    assert torch.equal(capturer.device_buffer[:3, 0, :], topk_ids)
    assert capturer.weight_device_buffer is not None
    torch.testing.assert_close(
        capturer.weight_device_buffer[:3, 0, :],
        topk_weights.float(),
    )


def test_routed_experts_capturer_requires_weights_when_enabled():
    capturer = _capturer_with_buffer(dp_rank=0, capture_weights=True)
    ctx = SimpleNamespace(dp_metadata=None)

    with (
        patch(f"{_REC_MODULE}.get_forward_context", return_value=ctx),
        pytest.raises(ValueError, match="enabled but not provided"),
    ):
        capturer.capture(
            layer_id=0,
            topk_ids=torch.tensor([[1, 2]], dtype=torch.int32),
        )


def test_routed_experts_capturer_dp_naive_concatenated_all_ranks():
    """n == sum(num_tokens_dp): slice this rank's segment from concatenated topk."""
    capturer = _capturer_with_buffer(dp_rank=1)
    num_tokens_dp = torch.tensor([2, 3], dtype=torch.int32)
    ctx = SimpleNamespace(
        dp_metadata=SimpleNamespace(num_tokens_across_dp_cpu=num_tokens_dp)
    )
    # Concatenated order: rank0 rows then rank1 rows.
    topk = torch.tensor(
        [[0, 1], [2, 3], [10, 11], [12, 13], [14, 15]], dtype=torch.int32
    )
    with patch(f"{_REC_MODULE}.get_forward_context", return_value=ctx):
        capturer.capture(layer_id=0, topk_ids=topk)
    want = topk[2:5]
    assert torch.equal(capturer.device_buffer[:3, 0, :], want)


def test_routed_experts_capturer_dp_modular_local_tokens():
    """n == token_num_per_dp: topk is already local to this DP rank."""
    capturer = _capturer_with_buffer(dp_rank=1)
    num_tokens_dp = torch.tensor([2, 3], dtype=torch.int32)
    ctx = SimpleNamespace(
        dp_metadata=SimpleNamespace(num_tokens_across_dp_cpu=num_tokens_dp)
    )
    topk = torch.tensor([[10, 11], [12, 13], [14, 15]], dtype=torch.int32)
    with patch(f"{_REC_MODULE}.get_forward_context", return_value=ctx):
        capturer.capture(layer_id=0, topk_ids=topk)
    assert torch.equal(capturer.device_buffer[:3, 0, :], topk)


def test_routed_experts_capturer_dp_unexpected_batch_raises():
    """Mismatch between topk batch dim and DP layout: fail fast."""
    capturer = _capturer_with_buffer(dp_rank=0)
    num_tokens_dp = torch.tensor([2, 3], dtype=torch.int32)
    ctx = SimpleNamespace(
        dp_metadata=SimpleNamespace(num_tokens_across_dp_cpu=num_tokens_dp)
    )
    # total=5, local=2: n=1 matches neither naive (5) nor modular (2).
    topk = torch.tensor([[1, 2]], dtype=torch.int32)
    with (
        patch(f"{_REC_MODULE}.get_forward_context", return_value=ctx),
        pytest.raises(AssertionError, match="unexpected topk_ids batch dim"),
    ):
        capturer.capture(layer_id=0, topk_ids=topk)
    assert capturer.device_buffer[0, 0, 0].item() == -1


def test_routed_experts_attention_group_is_shared_and_fail_closed(monkeypatch):
    class FullAttentionSpec:
        pass

    monkeypatch.setattr(f"{_REC_MODULE}.FullAttentionSpec", FullAttentionSpec)
    config = SimpleNamespace(
        kv_cache_groups=[
            SimpleNamespace(kv_cache_spec=object()),
            SimpleNamespace(kv_cache_spec=FullAttentionSpec()),
        ]
    )
    assert get_routed_experts_attn_gid(config) == 1

    with pytest.raises(ValueError, match="requires a full-attention KV cache group"):
        get_routed_experts_attn_gid(SimpleNamespace(kv_cache_groups=[]))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_mrv2_async_output_returns_existing_routed_experts_field():
    from vllm.v1.outputs import ModelRunnerOutput, RoutedExpertsTensors
    from vllm.v1.worker.gpu.async_utils import AsyncOutput
    from vllm.v1.worker.gpu.sample.output import SamplerOutput

    routed_experts = RoutedExpertsTensors(
        routing_data=torch.arange(6, dtype=torch.int32, device="cuda").reshape(3, 1, 2),
        slot_mapping=torch.tensor([11, 12, 13], device="cuda"),
        routing_weights=torch.linspace(
            0.1, 0.6, 6, dtype=torch.float32, device="cuda"
        ).reshape(3, 1, 2),
    )
    num_sampled = torch.tensor([1], dtype=torch.int32, device="cuda")
    sampler_output = SamplerOutput(
        sampled_token_ids=torch.tensor([[1]], device="cuda"),
        logprobs_tensors=None,
        num_nans=None,
        num_sampled=num_sampled,
        num_rejected=torch.tensor([0], dtype=torch.int32, device="cuda"),
    )
    output = AsyncOutput(
        model_runner_output=ModelRunnerOutput(req_ids=["req"], req_id_to_index={}),
        sampler_output=sampler_output,
        num_sampled_tokens=num_sampled,
        main_stream=torch.cuda.current_stream(),
        copy_stream=torch.cuda.Stream(),
        routed_experts=routed_experts,
    ).get_output()

    assert output.routed_experts is not None
    assert output.routed_experts.routing_data[:, 0, 0].tolist() == [0, 2, 4]
    assert output.routed_experts.slot_mapping.tolist() == [11, 12, 13]
    assert output.routed_experts.routing_weights is not None
    np.testing.assert_allclose(
        output.routed_experts.routing_weights[:, 0, 0],
        np.array([0.1, 0.3, 0.5], dtype=np.float32),
    )


@pytest.mark.parametrize("rank", [0, 1])
def test_all_tp_ranks_initialize_capture(monkeypatch, rank):
    pytest.importorskip("vllm.vllm_flash_attn", exc_type=ImportError)
    import vllm.v1.worker.gpu.model_runner as model_runner

    capturer = Mock()
    constructor = Mock(return_value=capturer)
    bind = Mock()
    monkeypatch.setattr(model_runner, "RoutedExpertsCapturer", constructor)
    monkeypatch.setattr(model_runner, "bind_routed_experts_capturer", bind)

    runner = model_runner.GPUModelRunner.__new__(model_runner.GPUModelRunner)
    runner.max_num_tokens = 32
    runner.vllm_config = SimpleNamespace(parallel_config=SimpleNamespace(rank=rank))
    runner.kv_cache_config = SimpleNamespace()
    runner.model = Mock()

    runner.init_routed_experts_capturer()

    constructor.assert_called_once_with(
        max_num_batched_tokens=32,
        vllm_config=runner.vllm_config,
        kv_cache_config=runner.kv_cache_config,
    )
    bind.assert_called_once_with(runner.model, capturer)
    assert runner.routed_experts_capturer is capturer


def test_v2_model_runner_accepts_routed_experts(monkeypatch):
    monkeypatch.setattr("importlib.metadata.entry_points", lambda **_: ())
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            enable_return_routed_experts=True,
            enable_return_routed_expert_weights=True,
            use_mla=False,
            logits_processors=None,
            enable_prompt_embeds=False,
        ),
        speculative_config=None,
        parallel_config=SimpleNamespace(
            prefill_context_parallel_size=1,
            tensor_parallel_size=1,
            distributed_executor_backend=None,
            pipeline_parallel_size=1,
            enable_dbo=False,
            enable_elastic_ep=False,
        ),
        compilation_config=SimpleNamespace(
            mode=CompilationMode.NONE,
            pass_config=SimpleNamespace(enable_sp=False),
        ),
        cache_config=SimpleNamespace(kv_sharing_fast_prefill=False),
        ec_transfer_config=None,
    )

    unsupported = VllmConfig._get_v2_model_runner_unsupported_features(config)

    assert "routed experts capture" not in unsupported
