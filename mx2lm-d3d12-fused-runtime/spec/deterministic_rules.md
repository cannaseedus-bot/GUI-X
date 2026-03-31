# Deterministic Rules — MX2LM D3D12 Fused Runtime

## Core Law

> **Same input → same output. Always. On any compliant GPU.**

---

## Hard Rules (Non-Negotiable)

### 1. Fixed Dispatch Bounds

```
entityCount     : const uint  — fixed at step start
gridCellCount   : const uint  — fixed at step start
NEIGHBOR_CAP    : const uint  — compile-time constant (64)
```

No dynamic branching on iteration count. No "until convergence" loops.

---

### 2. No Atomics in Fused Kernel

The fused attention/force/MoE kernel **must not use atomics**.

- Grid build may use atomics (controlled, bounded)
- Fused kernel: zero atomics, zero inter-thread communication

Exception: explicitly flagged experimental mode (`ATOMIC_REDUCE_MODE`), disabled by default.

---

### 3. No Unordered Writes

Each output index `[i]` is written by exactly one thread: thread `i`.
No two threads write to the same output index.

```hlsl
// CORRECT
force[tid] = computedForce;

// FORBIDDEN
force[neighborId] += contribution;  // race condition
```

---

### 4. Deterministic Neighbor Order

Neighbor iteration order is deterministic:

1. Sort grid indices by Morton code within each cell (done in grid_build)
2. Iterate neighbors in sorted order
3. Break at `NEIGHBOR_CAP` — do not skip or reorder

---

### 5. No Wave-Level Intrinsics in Critical Path

`WaveActiveSum`, `WaveReadLaneFirst`, etc. are **forbidden** in the fused kernel.
Wave composition varies by hardware → non-deterministic results.

Permitted only in: `prefix_sum.hlsl` (explicitly documented deviation).

---

### 6. Stable Grid Indexing

The same entity always lands in the same grid cell given the same position.

```hlsl
int3 cellCoord = floor((position.xyz - gridOrigin) / cellSize);
uint cellId    = cellCoord.x
               + cellCoord.y * gridDim.x
               + cellCoord.z * gridDim.x * gridDim.y;
```

Grid origin, cell size, and grid dims are root constants — never derived.

---

## Numerical Rules

### Epsilon

```hlsl
#define EPSILON 1e-5f
```

Used in all safe-normalize and near-zero checks.

### Safe Normalize

```hlsl
float3 SafeNormalize(float3 v) {
    float len = length(v);
    return (len > EPSILON) ? (v / len) : float3(0.0f, 0.0f, 1.0f);
}
```

### Softmax Stability

```hlsl
// Always subtract max before exp()
float maxScore = max(scores[0], ...);
float sumExp   = 0.0f;
for (int j = 0; j < n; j++) {
    scores[j] = exp(scores[j] - maxScore);
    sumExp    += scores[j];
}
for (int j = 0; j < n; j++) {
    scores[j] /= (sumExp + EPSILON);
}
```

### Force Clamp

```hlsl
#define FORCE_MAX 1e4f
force = clamp(force, -FORCE_MAX, FORCE_MAX);
```

### NaN Guard

```hlsl
// Applied to all outputs before write
float GuardNaN(float v) {
    return isnan(v) || isinf(v) ? 0.0f : v;
}
float4 GuardNaN4(float4 v) {
    return float4(GuardNaN(v.x), GuardNaN(v.y), GuardNaN(v.z), GuardNaN(v.w));
}
```

---

## Replay Guarantee

Given:
- Identical `entities[]`, `position[]`, `velocity[]`, `signal[]`, `axes[]`
- Identical root constants
- Identical grid state

The outputs `force[]`, `signal[]`, `events[]`, `event_params[]` **must be bit-identical**.

This is validated by `tools/validate_buffers.py` in CI.

---

## Versioning

| Version | Date       | Change                              |
|---------|------------|-------------------------------------|
| 0.1     | 2026-03-30 | Initial frozen rules                |
