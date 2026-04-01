/**
 * sha256_runtime.js  —  MX2LM SHA256 Content-Addressed Execution
 *
 * DESIGN:
 *   Every graph state has a SHA256 hash.
 *   Same hash → same cached result (deterministic replay).
 *   Execution flow:
 *     1. hash(graph)  → check cache
 *     2. cache hit    → return cached output instantly
 *     3. cache miss   → run inference (WASM / REST), store result by hash
 *
 * THREE BACKENDS (tried in order):
 *   WASM  → local compute, lowest latency
 *   REST  → PowerShell server at /graph/infer (JSON exchange)
 *   JS    → pure JS fallback, always available
 *
 * CACHE:
 *   In-memory Map (bounded by maxEntries).
 *   Optionally persisted to localStorage or IndexedDB.
 */

'use strict';

import { TensorGraph } from './tensor_graph.js';
import { wasmStep, isWasmReady } from './wasm_sandbox.js';

const DEFAULT_REST_BASE  = 'http://localhost:3000';
const DEFAULT_MAX_CACHE  = 4096;

// ── Sha256Runtime ─────────────────────────────────────────────

export class Sha256Runtime {
  /**
   * @param {object} opts
   * @param {string}  [opts.restBase]    REST API base URL
   * @param {number}  [opts.maxCache]    Max cache entries (LRU evict on overflow)
   * @param {boolean} [opts.persist]     Persist cache to localStorage
   * @param {'wasm'|'rest'|'js'} [opts.preferBackend]
   */
  constructor({
    restBase      = DEFAULT_REST_BASE,
    maxCache      = DEFAULT_MAX_CACHE,
    persist       = false,
    preferBackend = 'wasm'
  } = {}) {
    this.restBase      = restBase;
    this.maxCache      = maxCache;
    this.persist       = persist;
    this.preferBackend = preferBackend;

    // SHA256 → { outputJSON, hash, step, backend, latencyMs }
    this._cache    = new Map();
    this._lruKeys  = [];       // front = oldest

    // Stats
    this.stats = { hits: 0, misses: 0, wasmRuns: 0, restRuns: 0, jsRuns: 0 };

    if (persist) this._loadFromStorage();
  }

  // ── Hash ────────────────────────────────────────────────────

  /**
   * Compute SHA256 of a TensorGraph.
   * @returns {Promise<string>} hex hash
   */
  async hash(graph) {
    if (!graph.sha256) await graph.computeHash();
    return graph.sha256;
  }

  // ── Execute ─────────────────────────────────────────────────

  /**
   * Run one inference step with content-addressed caching.
   *
   * @param {TensorGraph} graph
   * @returns {Promise<{ graph: TensorGraph, hash: string, fromCache: boolean, backend: string, latencyMs: number }>}
   */
  async step(graph) {
    const t0   = performance.now();
    const hash = await this.hash(graph);

    // Cache hit
    if (this._cache.has(hash)) {
      this.stats.hits++;
      const entry = this._cache.get(hash);
      this._touchLRU(hash);
      return {
        graph:     TensorGraph.fromJSON(entry.outputJSON),
        hash:      entry.hash,
        fromCache: true,
        backend:   entry.backend,
        latencyMs: performance.now() - t0
      };
    }

    this.stats.misses++;

    // Cache miss — run inference
    const backend = this._selectBackend();
    let outGraph;

    try {
      if (backend === 'wasm') {
        // Deep-copy graph so wasm mutates the copy
        const copy = TensorGraph.fromJSON(graph.toJSON());
        wasmStep(copy);
        outGraph = copy;
        this.stats.wasmRuns++;
      }
      else if (backend === 'rest') {
        outGraph = await this._restStep(graph);
        this.stats.restRuns++;
      }
      else {
        const copy = TensorGraph.fromJSON(graph.toJSON());
        copy.stepJS();
        outGraph = copy;
        this.stats.jsRuns++;
      }
    } catch (e) {
      console.warn(`[sha256_runtime] ${backend} failed, falling back to JS:`, e.message);
      const copy = TensorGraph.fromJSON(graph.toJSON());
      copy.stepJS();
      outGraph = copy;
      this.stats.jsRuns++;
    }

    const outHash = await this.hash(outGraph);
    const latencyMs = performance.now() - t0;

    // Store result
    this._storeCache(hash, {
      outputJSON: outGraph.toJSON(),
      hash:       outHash,
      backend,
      latencyMs
    });

    return { graph: outGraph, hash: outHash, fromCache: false, backend, latencyMs };
  }

  /**
   * Run N steps, chaining hashes.
   */
  async runN(graph, n = 1) {
    let current = graph;
    const trace = [];
    for (let i = 0; i < n; i++) {
      const result = await this.step(current);
      trace.push({ step: current.step, hash: result.hash,
                   fromCache: result.fromCache, backend: result.backend });
      current = result.graph;
    }
    return { graph: current, trace };
  }

  // ── REST Backend ─────────────────────────────────────────────

  async _restStep(graph) {
    const body = JSON.stringify(graph.toJSON());
    const res  = await fetch(`${this.restBase}/graph/infer`, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body
    });
    if (!res.ok) throw new Error(`REST /graph/infer ${res.status}`);
    const json = await res.json();
    return TensorGraph.fromJSON(json);
  }

  // ── Cache Management ─────────────────────────────────────────

  _selectBackend() {
    if (this.preferBackend === 'wasm' && isWasmReady()) return 'wasm';
    if (this.preferBackend === 'rest') return 'rest';
    return 'js';
  }

  _storeCache(hash, entry) {
    if (this._cache.size >= this.maxCache) {
      const oldest = this._lruKeys.shift();
      this._cache.delete(oldest);
    }
    this._cache.set(hash, entry);
    this._lruKeys.push(hash);
    if (this.persist) this._saveToStorage(hash, entry);
  }

  _touchLRU(hash) {
    const idx = this._lruKeys.indexOf(hash);
    if (idx !== -1) this._lruKeys.splice(idx, 1);
    this._lruKeys.push(hash);
  }

  /** Get cached output by hash (synchronous) */
  get(hash) { return this._cache.get(hash) || null; }

  /** Check if hash is cached */
  has(hash) { return this._cache.has(hash); }

  /** Clear all cache */
  clear() {
    this._cache.clear();
    this._lruKeys = [];
    if (this.persist) localStorage.removeItem('mx2lm_sha256_cache');
  }

  /** Dump stats */
  report() {
    const { hits, misses, wasmRuns, restRuns, jsRuns } = this.stats;
    const total = hits + misses;
    return {
      cacheSize:  this._cache.size,
      hitRate:    total ? (hits / total).toFixed(3) : '0',
      hits, misses, wasmRuns, restRuns, jsRuns
    };
  }

  // ── localStorage persistence ──────────────────────────────

  _saveToStorage(hash, entry) {
    try {
      const key = `mx2lm_sha256_${hash}`;
      localStorage.setItem(key, JSON.stringify(entry));
    } catch (e) { /* quota exceeded — ignore */ }
  }

  _loadFromStorage() {
    try {
      for (let i = 0; i < localStorage.length; i++) {
        const k = localStorage.key(i);
        if (k && k.startsWith('mx2lm_sha256_')) {
          const hash  = k.slice('mx2lm_sha256_'.length);
          const entry = JSON.parse(localStorage.getItem(k));
          this._cache.set(hash, entry);
          this._lruKeys.push(hash);
        }
      }
      console.log(`[sha256_runtime] Restored ${this._cache.size} entries from localStorage`);
    } catch (e) { /* ignore */ }
  }
}

// ── REST /graph/hash helper ────────────────────────────────────

/**
 * Ask the server for the SHA256 of a graph state.
 * Used to verify determinism across WASM / REST / JS backends.
 */
export async function remoteHash(graph, restBase = DEFAULT_REST_BASE) {
  const res = await fetch(`${restBase}/graph/hash`, {
    method:  'POST',
    headers: { 'Content-Type': 'application/json' },
    body:    JSON.stringify(graph.toJSON())
  });
  if (!res.ok) throw new Error(`REST /graph/hash ${res.status}`);
  const { sha256 } = await res.json();
  return sha256;
}

/**
 * Fetch a cached state by hash from the server.
 */
export async function fetchByHash(hash, restBase = DEFAULT_REST_BASE) {
  const res = await fetch(`${restBase}/graph/${hash}`);
  if (res.status === 404) return null;
  if (!res.ok) throw new Error(`REST /graph/${hash} ${res.status}`);
  return TensorGraph.fromJSON(await res.json());
}
