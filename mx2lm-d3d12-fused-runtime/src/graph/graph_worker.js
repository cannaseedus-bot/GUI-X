/**
 * graph_worker.js  —  MX2LM Web Worker: background graph inference
 *
 * Runs inside a Web Worker so the main thread stays unblocked.
 * Supports WASM, REST, and JS backends.
 *
 * Protocol (postMessage):
 *   IN:
 *     { type: 'init',   wasmUrl: string, restBase: string, backend: string }
 *     { type: 'step',   id: string, graph: object }      → run 1 step
 *     { type: 'runN',   id: string, graph: object, n: number }
 *     { type: 'hash',   id: string, graph: object }
 *     { type: 'stats' }
 *
 *   OUT:
 *     { type: 'ready' }
 *     { type: 'result', id, graph, hash, fromCache, backend, latencyMs }
 *     { type: 'error',  id, message }
 *     { type: 'stats',  report }
 */

'use strict';

importScripts('./tensor_graph.js', './wasm_sandbox.js', './sha256_runtime.js');

// Worker-local runtime instance
let runtime = null;

self.onmessage = async function({ data }) {
  const { type, id } = data;

  try {
    switch (type) {

      case 'init': {
        const { wasmUrl = './tensor_kernel.wasm',
                restBase = 'http://localhost:3000',
                backend  = 'wasm',
                maxCache = 4096 } = data;

        if (wasmUrl) {
          const ok = await initWasm(wasmUrl);
          if (!ok) console.warn('[graph_worker] WASM unavailable, using JS fallback');
        }

        runtime = new Sha256Runtime({ restBase, preferBackend: backend, maxCache });
        self.postMessage({ type: 'ready', wasmLoaded: isWasmReady() });
        break;
      }

      case 'step': {
        if (!runtime) throw new Error('Worker not initialized. Send "init" first.');
        const graph  = TensorGraph.fromJSON(data.graph);
        const result = await runtime.step(graph);
        self.postMessage({
          type: 'result', id,
          graph:     result.graph.toJSON(),
          hash:      result.hash,
          fromCache: result.fromCache,
          backend:   result.backend,
          latencyMs: result.latencyMs
        });
        break;
      }

      case 'runN': {
        if (!runtime) throw new Error('Worker not initialized.');
        const graph  = TensorGraph.fromJSON(data.graph);
        const { graph: outGraph, trace } = await runtime.runN(graph, data.n || 1);
        self.postMessage({
          type: 'result', id,
          graph: outGraph.toJSON(),
          hash:  outGraph.sha256,
          trace
        });
        break;
      }

      case 'hash': {
        const graph = TensorGraph.fromJSON(data.graph);
        const hash  = await graph.computeHash();
        self.postMessage({ type: 'result', id, hash });
        break;
      }

      case 'stats': {
        const report = runtime ? runtime.report() : null;
        self.postMessage({ type: 'stats', report });
        break;
      }

      default:
        throw new Error(`Unknown message type: ${type}`);
    }
  } catch (e) {
    self.postMessage({ type: 'error', id, message: e.message });
  }
};
