# Memory Layout — MX2LM D3D12 Fused Runtime

## Layout Contract: SoA (Structure of Arrays)

**FROZEN. Do not change without updating all shaders, loaders, and bindings.**

---

## Primary Entity Buffers

All arrays are indexed by entity ID `[0, entityCount)`.

```c
// ── Identity ─────────────────────────────────────────────
entities        : uint[]        // entity type tag / ID
                                // 16-bit type | 16-bit flags

// ── Spatial State ─────────────────────────────────────────
position        : float4[]      // xyz = world pos, w = type_mask
velocity        : float4[]      // xyz = velocity, w = speed
axes            : float3x3[]    // 3 float4 rows (padded to float4x3)

// ── Signal State ──────────────────────────────────────────
signal          : float[]       // scalar routing signal [0.0, N_experts)

// ── Output Buffers ────────────────────────────────────────
force           : float4[]      // xyz = force vector, w = magnitude
events          : uint[]        // event type (0 = none)
event_params    : float4[]      // event-specific payload
```

---

## Grid Buffers

Used for spatial neighbor lookup. Rebuilt every dispatch via `grid_build.hlsl`.

```c
grid_offsets    : uint[]        // [cell_id] → start index in grid_indices
grid_counts     : uint[]        // [cell_id] → entity count in cell
grid_indices    : uint[]        // flat sorted list of entity IDs per cell
```

### Grid Dimensions

```c
GRID_DIM        : uint3         // cells per axis (e.g. 64×64×64)
GRID_CELL_SIZE  : float         // world units per cell (e.g. 1.0)
GRID_ORIGIN     : float3        // world-space minimum corner
```

---

## Alignment Rules

| Rule                        | Requirement            |
|-----------------------------|------------------------|
| Base alignment              | 16 bytes (float4)      |
| Array element stride        | sizeof(float4) = 16    |
| `axes` stride               | 3 × float4 = 48 bytes  |
| No AoS                      | Hard requirement       |
| No interleaved attributes   | Hard requirement       |
| Contiguous allocation only  | Hard requirement       |
| No dynamic indexing via ptr | Hard requirement       |

---

## Buffer Size Formulas

```
entities        = entityCount × 4   bytes
position        = entityCount × 16  bytes
velocity        = entityCount × 16  bytes
signal          = entityCount × 4   bytes  (+ 12 pad per row if needed)
axes            = entityCount × 48  bytes  (3 × float4)
force           = entityCount × 16  bytes
events          = entityCount × 4   bytes
event_params    = entityCount × 16  bytes

grid_offsets    = gridCellCount × 4 bytes
grid_counts     = gridCellCount × 4 bytes
grid_indices    = entityCount   × 4 bytes  (worst case all in one cell)
```

---

## D3D12 Heap Placement

```
DEFAULT heap    : position, velocity, axes, signal, force,
                  grid_offsets, grid_counts, grid_indices,
                  events, event_params

UPLOAD heap     : staging buffers for CPU→GPU transfer
READBACK heap   : debug readback buffers only
```

---

## Register / Root Signature Binding

See `root_signature.h` for the authoritative binding table.

```
t0  entities        (SRV)
t1  position        (SRV, read-only pass)
t2  velocity        (SRV)
t3  signal          (SRV, pre-dispatch value)
t4  axes            (SRV)
u5  force           (UAV)
t6  grid_offsets    (SRV)
t7  grid_counts     (SRV)
t8  grid_indices    (SRV)
u3  signal          (UAV, post-dispatch write — same semantic, separate slot)
u9  events          (UAV)
u10 event_params    (UAV)
```

> Note: `signal` is bound twice — once as SRV (read) and once as UAV (write).
> The shader reads the pre-dispatch value and writes the updated value.
> Two distinct `D3D12_DESCRIPTOR_RANGE` entries required.

---

## Versioning

| Version | Date       | Change                        |
|---------|------------|-------------------------------|
| 0.1     | 2026-03-30 | Initial frozen layout         |
