/**
 * wasm_sandbox.js  —  MX2LM WASM Compute Sandbox
 *
 * Loads tensor_kernel.wasm (compiled from tensor_kernel.wat),
 * writes graph state into linear memory, runs infer_all(),
 * reads results back as typed arrays.
 *
 * Compile .wat → .wasm:
 *   wat2wasm src/graph/tensor_kernel.wat -o src/graph/tensor_kernel.wasm
 *   (requires WABT: https://github.com/WebAssembly/wabt)
 *
 * Falls back to JavaScript inference if WASM is unavailable.
 */

'use strict';

import { TensorGraph } from './tensor_graph.js';

// ── WASM Loader ───────────────────────────────────────────────

let _wasmInstance = null;
let _wasmMem      = null;

/**
 * Load and instantiate the WASM module.
 * @param {string} wasmUrl  URL or path to tensor_kernel.wasm
 */
export async function initWasm(wasmUrl = './tensor_kernel.wasm') {
  try {
    const res    = await fetch(wasmUrl);
    const bytes  = await res.arrayBuffer();
    const result = await WebAssembly.instantiate(bytes, {});
    _wasmInstance = result.instance;
    _wasmMem      = _wasmInstance.exports.mem;
    console.log('[wasm_sandbox] WASM loaded, mem pages:', _wasmMem.buffer.byteLength / 65536);
    return true;
  } catch (e) {
    console.warn('[wasm_sandbox] WASM load failed, will use JS fallback:', e.message);
    return false;
  }
}

// ── Memory write helpers ──────────────────────────────────────

function f32View(offset, count) {
  return new Float32Array(_wasmMem.buffer, offset, count);
}

function i32View(offset, count) {
  return new Int32Array(_wasmMem.buffer, offset, count);
}

/**
 * Write TensorGraph into WASM linear memory.
 * Returns { entityCount, edgeCount }.
 */
function writeGraph(graph) {
  const exports = _wasmInstance.exports;
  const N = graph.nodes.size;
  const E = graph.edges.length;

  exports.set_entity_count(N);
  exports.set_edge_count(E);

  const gateBase  = exports.gate_base();
  const posBase   = exports.pos_base();
  const efromBase = exports.efrom_base();
  const etoBase   = exports.eto_base();
  const ewBase    = exports.ew_base();

  const gateView  = f32View(gateBase,  N * 8);
  const posView   = f32View(posBase,   N * 3);
  const efromView = i32View(efromBase, E);
  const etoView   = i32View(etoBase,   E);
  const ewView    = f32View(ewBase,    E);

  // Write nodes
  let ni = 0;
  for (const node of [...graph.nodes.values()].sort((a, b) => a.id - b.id)) {
    for (let k = 0; k < 8; k++) gateView[ni * 8 + k] = node.d[k];
    posView[ni * 3]     = node.px;
    posView[ni * 3 + 1] = node.py;
    posView[ni * 3 + 2] = node.pz;
    ni++;
  }

  // Write edges
  for (let i = 0; i < E; i++) {
    efromView[i] = graph.edges[i].from;
    etoView[i]   = graph.edges[i].to;
    ewView[i]    = graph.edges[i].weight;
  }

  return { entityCount: N, edgeCount: E };
}

/**
 * Read WASM output back into the graph (mutates node force/signal).
 */
function readResults(graph) {
  const exports   = _wasmInstance.exports;
  const outFBase  = exports.out_f_base();
  const outSBase  = exports.out_s_base();
  const N         = graph.nodes.size;

  const outForce  = f32View(outFBase, N * 3);
  const outSignal = f32View(outSBase, N);

  let ni = 0;
  for (const node of [...graph.nodes.values()].sort((a, b) => a.id - b.id)) {
    node.fx     = outForce[ni * 3];
    node.fy     = outForce[ni * 3 + 1];
    node.fz     = outForce[ni * 3 + 2];
    node.signal = outSignal[ni];
    ni++;
  }
}

// ── Public API ────────────────────────────────────────────────

/**
 * Run one inference step on a TensorGraph.
 * Uses WASM if loaded, JS fallback otherwise.
 *
 * @param {TensorGraph} graph
 * @returns {TensorGraph}  (mutated in-place, sha256 invalidated)
 */
export function wasmStep(graph) {
  if (!_wasmInstance) {
    console.debug('[wasm_sandbox] WASM not loaded, using JS fallback');
    return graph.stepJS();
  }

  writeGraph(graph);
  _wasmInstance.exports.infer_all();
  readResults(graph);
  graph.step++;
  graph.sha256 = null;
  return graph;
}

/**
 * Run N inference steps.
 */
export function wasmRunN(graph, n = 1) {
  for (let i = 0; i < n; i++) wasmStep(graph);
  return graph;
}

/**
 * Check WASM availability
 */
export function isWasmReady() { return _wasmInstance !== null; }
