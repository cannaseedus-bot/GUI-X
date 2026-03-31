# Execution Model — MX2LM D3D12 Fused Runtime

## Dispatch Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                        Frame / Step N                           │
│                                                                 │
│  [Load SCXQ2]  ──►  [Upload Buffers]  ──►  [Grid Build Pass]   │
│                                                 │               │
│                                                 ▼               │
│                                     [Fused Kernel Dispatch]     │
│                                                 │               │
│                                                 ▼               │
│                                     [Optional Readback / Viz]  │
│                                                 │               │
│                                                 ▼               │
│                                           [Frame N+1]           │
└─────────────────────────────────────────────────────────────────┘
```

---

## Stage 1: Grid Build

**Shader:** `grid_build.hlsl`
**Input:** `position[]`
**Output:** `grid_offsets[]`, `grid_counts[]`, `grid_indices[]`

### Steps

1. **Count pass** — each entity writes +1 to `grid_counts[cellId]`
2. **Prefix sum** — exclusive scan over `grid_counts` → `grid_offsets`
3. **Scatter pass** — each entity writes its index to `grid_indices[offset++]`

### Dispatch

```cpp
// Count pass
Dispatch(ceil(entityCount / 128), 1, 1);

// Prefix sum (recursive or single-pass for small grids)
Dispatch(ceil(gridCellCount / 256), 1, 1);

// Scatter pass
Dispatch(ceil(entityCount / 128), 1, 1);
```

### Resource Barriers

```
UAV barrier after count pass before prefix sum
UAV barrier after prefix sum before scatter pass
UAV barrier after scatter pass before fused kernel
```

---

## Stage 2: Fused Kernel Dispatch

**Shader:** `fused_attention_force_moe.hlsl`
**Input:** all entity buffers + grid buffers
**Output:** `force[]`, `signal[]` (updated), `events[]`, `event_params[]`

### Dispatch

```cpp
UINT groupCount = (entityCount + 127) / 128;
cmdList->Dispatch(groupCount, 1, 1);
```

### Resource Barriers

```
Transition force, signal (UAV→UAV), events, event_params
UAV barrier before optional readback
```

---

## Stage 3: Optional Readback

Used for debugging, validation, or host-side logic.

```cpp
// Copy GPU buffer → readback heap
cmdList->CopyResource(readbackBuffer, forceBuffer);

// Fence + wait
fence->Signal(++fenceValue);
// ... wait ...

// Map and inspect
float4* ptr = nullptr;
readbackBuffer->Map(0, nullptr, (void**)&ptr);
```

Only enabled in `DEBUG` builds or when `Config::enableReadback = true`.

---

## Command List Strategy

- **Single direct command list** per frame
- No bundles (reserved for future batch optimization)
- No async compute (reserved for multi-queue extension)
- Command list reset at start of each frame

---

## Synchronization

```
GPU timeline:
  grid_build_done → fused_kernel_done → readback_done

CPU-GPU sync:
  Signal fence after ExecuteCommandLists()
  WaitForSingleObject on fence event
```

---

## Fixed-Loop Guarantee

All dispatch sizes are computed from `entityCount` — a constant per step.
No dynamic branching on dispatch granularity. No resize mid-step.

```
entityCount     FROZEN per simulation step
gridCellCount   FROZEN per simulation step
bufferSizes     FROZEN per simulation step
```

---

## Error Handling

- Debug layer enabled in `DEBUG` builds
- `HRESULT` checked via `ThrowIfFailed()` after every D3D12 call
- Shader validation via PIX integration (optional, `#ifdef PIX_ENABLED`)
