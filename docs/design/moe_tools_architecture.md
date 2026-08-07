# MoE tools architecture reconnaissance

The current optimized modular path is `MoERunner._apply_quant_method` →
`BaseRouter.select_experts` → `(topk_weights, topk_ids)` →
`RoutedExperts.forward_modular` → `quant_method.apply`. `BaseRouter` is the
common pre/post-selection boundary for fused top-k, grouped top-k, biased
top-k, custom routing, zero-expert, and simulator routers. Logical IDs exist
together with weights immediately after `_compute_routing`, before EPLB maps
logical IDs to physical replicas.

`ExpertMapManager` and EPLB remain execution-placement mechanisms. Expert
eligibility is deliberately separate: a stable per-layer boolean mask is bound
on the router device before compilation and CUDA graph capture, then applied to
selection scores before the existing top-k. Correction-bias routers retain a
separate masked selection bias so disabled experts cannot re-enter after the
score activation. Weights stay loaded, checkpoint expert identity is unchanged,
and fused expert execution is not modified.

`RoutedExpertsCapturer` binds callbacks to modular `BaseRouter` instances, or
to monolithic kernels that explicitly implement routing replay. It writes
logical IDs to a preallocated device buffer, asynchronously transfers a step
snapshot, and uses attention slot mappings to associate prompt and generated
tokens with requests in the scheduler.

Qwen3 MoE constructs `FusedMoE` through the generic runner. Backend selection
is made by the quant method: modular backends expose routing before expert
execution, while monolithic FlashInfer/TRT-LLM-style implementations own
routing internally. Therefore eligibility currently fails closed for
monolithic paths. Random simulation, zero-expert routing, hash routing, and
custom routing also fail closed because they cannot guarantee
pre-top-k exclusion. The exact Qwen3.6 FP8 choice on an RTX PRO 6000 depends on
the installed CUDA/backend versions and runtime flags and must be recorded
from startup logs on the target host; it cannot be established from source or
validated on this CPU-only checkout.

Weight telemetry should extend the existing capturer with a parallel float32
buffer and carry IDs and weights through the same snapshot, D2H, slot replay,
and request-output structures. Monolithic implementations must explicitly
replay both tensors; they must not silently return IDs without weights.
