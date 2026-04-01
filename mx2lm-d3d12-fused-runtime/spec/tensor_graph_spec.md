# 8D Tensor Graph Spec  —  MX2LM Compute Substrate

## What This Is

An XML/SVG-structured **compute graph format** for 8D entity/expert topology.

NOT a visualization. NOT for humans to read.

This is a **machine-readable substrate** executed by three runtimes:

| Runtime         | File                    | Where it runs       |
|-----------------|-------------------------|---------------------|
| WASM sandbox    | `tensor_kernel.wasm`    | Browser / Node.js   |
| JSON REST API   | `server.ps1 /graph/*`   | PowerShell server   |
| SHA256 runtime  | `sha256_runtime.js`     | Browser cache layer |

---

## 8 Dimensions

Each entity (node) carries an **8D probability vector** over the 8 MoE experts:

```
d0..d7  =  softmax(gate_scores[0..7])
```

This is the **routing simplex** — the same 8D vector computed by the
fused HLSL kernel's MoE gate. The XML graph exposes it for compute-level
inter-operation between the DX12 runtime and JS/WASM/REST layers.

---

## XML Format (wire format)

```xml
<tensor-graph xmlns="urn:mx2lm:tensor-graph:v0.1"
              version="0.1" dim="8"
              entity-count="4096" step="42"
              sha256="a3f1...">

  <meta>
    <entry k="scene_id" v="hellscape_01"/>
    <entry k="timestamp" v="2026-04-01T00:00:00Z"/>
  </meta>

  <!-- One cluster per expert (argmax of d0..d7) -->
  <cluster expert="5" gate-mean="0.412381" sha256="b2c4...">

    <node id="0"
          d0="0.02" d1="0.05" d2="0.01" d3="0.03"
          d4="0.08" d5="0.71" d6="0.07" d7="0.03"
          px="1.2" py="-3.4" pz="0.8"
          fx="0.12" fy="-0.05" fz="0.31"
          signal="5" event="0"
          sha256="a1b2..."/>

    <edge from="0" to="17" weight="0.421" type="attention"/>
    <edge from="17" to="0" weight="0.218" type="force"/>

  </cluster>

</tensor-graph>
```

---

## JSON Wire Format (REST API)

```json
{
  "version": "0.1",
  "dim": 8,
  "entityCount": 4096,
  "step": 42,
  "sha256": "a3f1...",
  "meta": { "scene_id": "hellscape_01" },
  "clusters": {
    "5": [
      { "id": 0, "d": [0.02,0.05,0.01,0.03,0.08,0.71,0.07,0.03],
        "px": 1.2, "py": -3.4, "pz": 0.8,
        "fx": 0.12, "fy": -0.05, "fz": 0.31,
        "signal": 5, "event": 0, "sha256": "a1b2..." }
    ]
  },
  "edges": [
    { "from": 0, "to": 17, "weight": 0.421, "type": "attention" }
  ]
}
```

---

## REST Endpoints (PowerShell server, port 3000)

| Method | Path                | Description                                   |
|--------|---------------------|-----------------------------------------------|
| POST   | `/graph/infer`      | Run one 8D inference step, return updated graph |
| POST   | `/graph/hash`       | Return SHA256 of posted graph state           |
| GET    | `/graph/:sha256`    | Retrieve cached output by hash                |

### `/graph/infer` contract

**Request:** JSON graph (as above)
**Response:** JSON graph (step+1, updated forces/signals, sha256 set)

Server-side inference (`Step-Graph8D`) mirrors `tensor_graph.js stepJS()`:
- For each node: accumulate weighted position of incoming attention edges
- Force = context_centroid − self_position (clamped ±1e4)
- Signal = argmax(d0..d7)
- Deterministic — same input hash → same output

### Cache behavior

- Input graph JSON → SHA256 → lookup cache (up to 2048 entries, LRU evict)
- Cache hit: returns immediately, `_fromCache: true` in response
- Cache miss: runs `Step-Graph8D`, stores result keyed by input hash

---

## WASM Kernel (`tensor_kernel.wat`)

Linear memory layout (16 pages = 1 MB):

| Page | Offset   | Content                           |
|------|----------|-----------------------------------|
| 0    | 0x00000  | entity_count (i32)                |
| 0    | 0x00010  | gate_scores[N × 8] (f32)         |
| 1    | 0x10000  | position_xyz[N × 3] (f32)        |
| 2    | 0x20000  | force_xyz[N × 3] (f32)           |
| 3    | 0x30000  | signal[N] (f32)                   |
| 4-6  | 0x40000  | edge_from[], edge_to[], weight[]  |
| 7    | 0x70000  | edge_count (i32)                  |
| 8    | 0x80000  | output_force[N × 3] (f32)        |
| 9    | 0x90000  | output_signal[N] (f32)            |

**Exported functions:**
- `infer_all()` — runs softmax + argmax + 8D force propagation
- `set_entity_count(n)`, `set_edge_count(n)`
- `gate_base()`, `pos_base()`, `efrom_base()`, `eto_base()`, `ew_base()`, `out_f_base()`, `out_s_base()`

Compile:
```bash
wat2wasm src/graph/tensor_kernel.wat -o src/graph/tensor_kernel.wasm
```

---

## SHA256 Content-Addressed Execution

Every graph state is deterministically hashed:

```
SHA256(canonical JSON) → cache key
```

Guarantees:
- Same 8D entity state → same SHA256 → same output (from cache or recompute)
- Replay: fetch any historical state by hash from `/graph/:sha256`
- Cross-backend consistency: WASM / REST / JS must produce identical SHA256 for same input

---

## Builder (from DX12 readback)

```js
import { graphFromBuffers } from './tensor_graph.js';

const graph = graphFromBuffers({
  entityCount:  4096,
  step:         simStep,
  positionArr:  gpuReadbackFloat32,   // stride 4
  forceArr:     gpuReadbackForce,     // stride 4
  signalArr:    gpuReadbackSignal,    // stride 1
  eventArr:     gpuReadbackEvent,     // stride 1 (uint32)
  gateScoresArr: gpuReadbackGates,   // stride 8  ← 8D vector
});
```

---

## Determinism Contract

1. `sha256(XML_graph) === sha256(JSON_graph)` for the same logical state
2. `REST_step(graph).sha256 === WASM_step(graph).sha256 === JS_step(graph).sha256`
3. `step(step(graph, seed=42)) === step(step(graph, seed=42))` always
