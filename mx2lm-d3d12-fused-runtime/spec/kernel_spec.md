# Kernel Specification — MX2LM D3D12 Fused Runtime

## Overview

The fused kernel is the **single execution contract** of this runtime.
Every entity is processed exactly once per dispatch. No multi-pass. No staging.

---

## Execution Contract

```
1 thread = 1 entity

Thread index   : SV_DispatchThreadID.x
Entity count   : root constant (entityCount)
```

### Inputs (read-only)

| Buffer         | Type      | Register | Description                        |
|----------------|-----------|----------|------------------------------------|
| `entities`     | `uint[]`  | `t0`     | Entity IDs / type tags             |
| `position`     | `float4[]`| `t1`     | World position (w = type mask)     |
| `velocity`     | `float4[]`| `t2`     | Velocity (w = speed scalar)        |
| `signal`       | `float[]` | `t3`     | Routing signal (pre-dispatch)      |
| `axes`         | `float3x3[]`| `t4`   | Local orientation frame            |
| `grid_offsets` | `uint[]`  | `t6`     | Grid cell start offsets            |
| `grid_counts`  | `uint[]`  | `t7`     | Entity count per cell              |
| `grid_indices` | `uint[]`  | `t8`     | Sorted entity indices per cell     |

### Outputs (read-write)

| Buffer         | Type      | Register | Description                        |
|----------------|-----------|----------|------------------------------------|
| `force`        | `float4[]`| `u5`     | Accumulated force output           |
| `signal`       | `float[]` | `u3`     | Updated routing signal             |
| `events`       | `uint[]`  | `u9`     | Emitted event type (0 = none)      |
| `event_params` | `float4[]`| `u10`    | Event parameter payload            |

---

## Pipeline — Single Pass

```
Input buffers
    │
    ▼
[QKV Projection]
    │  Q = position × Wq
    │  K = neighbor positions × Wk
    │  V = neighbor signals × Wv
    ▼
[Attention]
    │  scores[j] = dot(Q, K[j]) / sqrt(dim)
    │  weights   = softmax(scores)
    │  context   = Σ weights[j] × V[j]
    ▼
[Force Accumulation]
    │  force[i] = W_force × context + bias
    ▼
[MoE Routing]
    │  gate_scores   = softmax(W_gate × context)
    │  expert_index  = argmax(gate_scores)
    │  signal[i]     = expert_index  (routed output)
    ▼
[Event Emission]
    │  if (force magnitude > threshold) → emit event
    ▼
Output buffers
```

---

## Thread Group Layout

```hlsl
[numthreads(128, 1, 1)]
void CSMain(uint3 tid : SV_DispatchThreadID)
```

- 128 threads per group
- 1D dispatch: `ceil(entityCount / 128)` groups
- Out-of-bounds threads early-exit: `if (tid.x >= entityCount) return;`

---

## Neighbor Sampling

Each entity samples from its grid cell and 26 adjacent cells (3³ - 1).

```
Max neighbors sampled : NEIGHBOR_CAP (default 64)
Sampling order        : Morton-sorted within cell
Overflow behavior     : early-break (deterministic)
```

---

## Numerical Contracts

- All forces clamped to `[-FORCE_MAX, FORCE_MAX]` (default 1e4)
- Softmax: `exp(x - max(x))` for numerical stability
- Safe normalize: returns `float3(0,0,1)` when `length(v) < 1e-5`
- No NaN propagation: all outputs validated before write

---

## Versioning

| Version | Date       | Change                        |
|---------|------------|-------------------------------|
| 0.1     | 2026-03-30 | Initial frozen layout         |
