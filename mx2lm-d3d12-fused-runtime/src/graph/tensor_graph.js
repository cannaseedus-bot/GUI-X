/**
 * tensor_graph.js  —  MX2LM 8D Tensor Graph
 *
 * PURPOSE: Build, serialize, and parse 8D tensor node cluster graphs.
 *          NOT rendering. This is a compute-level data format.
 *
 * DIMENSIONS: d0..d7 = softmax gate scores over 8 MoE experts.
 *
 * Three runtimes supported:
 *   1. WASM sandbox  (TensorGraphWasm)
 *   2. JSON REST API (TensorGraphRest)
 *   3. SHA256 content-addressed (TensorGraphSha256)
 *
 * Usage:
 *   import { TensorGraph, sha256Node } from './tensor_graph.js';
 *   const g = new TensorGraph({ entityCount: 4096, step: 0 });
 *   g.addNode({ id: 0, d: [0.1,0.4,0.05,...], px:1, py:2, pz:3 });
 *   g.addEdge(0, 1, 0.72, 'attention');
 *   const xml = g.toXML();
 *   const json = g.toJSON();
 */

'use strict';

const NS = 'urn:mx2lm:tensor-graph:v0.1';
const DIM = 8;
const EPSILON = 1e-5;

// ── Helpers ───────────────────────────────────────────────────

/** Clamp a number */
function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }

/** Safe softmax over Array<number> */
function softmax(arr) {
  const max = Math.max(...arr);
  const exps = arr.map(x => Math.exp(x - max));
  const sum = exps.reduce((a, b) => a + b, 0) + EPSILON;
  return exps.map(x => x / sum);
}

/** SHA256 of a UTF-8 string → hex string (SubtleCrypto, async) */
async function sha256Str(str) {
  const buf = new TextEncoder().encode(str);
  const hash = await crypto.subtle.digest('SHA-256', buf);
  return Array.from(new Uint8Array(hash))
    .map(b => b.toString(16).padStart(2, '0'))
    .join('');
}

/** SHA256 of an 8D node's state (sync approximation via fnv32 for hot path) */
function fnv32(arr) {
  let h = 0x811c9dc5;
  for (const v of arr) {
    const bytes = new Float32Array([v]).buffer;
    for (const b of new Uint8Array(bytes)) {
      h ^= b;
      h = (h * 0x01000193) >>> 0;
    }
  }
  return h.toString(16).padStart(8, '0');
}

// ── Node ──────────────────────────────────────────────────────

class TensorNode {
  /**
   * @param {object} opts
   * @param {number} opts.id
   * @param {number[]} opts.d  - 8D gate score vector (will be softmax-normalized)
   * @param {number} opts.px opts.py opts.pz
   * @param {number} [opts.fx] [opts.fy] [opts.fz]
   * @param {number} [opts.signal]
   * @param {number} [opts.event]
   */
  constructor({ id, d, px = 0, py = 0, pz = 0,
                fx = 0, fy = 0, fz = 0,
                signal = 0, event = 0 }) {
    if (!Array.isArray(d) || d.length !== DIM)
      throw new Error(`Node ${id}: d must be float[8]`);

    this.id     = id;
    this.d      = softmax(d);   // normalize to probability simplex
    this.px = px; this.py = py; this.pz = pz;
    this.fx = fx; this.fy = fy; this.fz = fz;
    this.signal = signal;
    this.event  = event;

    // Expert assignment = argmax of gate scores
    this.expert = this.d.indexOf(Math.max(...this.d));

    // Content address: fnv32 of [id, d0..d7, px,py,pz]
    this.sha256 = fnv32([id, ...this.d, px, py, pz]);
  }

  /** Return as plain object (JSON-serializable) */
  toJSON() {
    return {
      id: this.id, sha256: this.sha256,
      d: this.d, expert: this.expert,
      px: this.px, py: this.py, pz: this.pz,
      fx: this.fx, fy: this.fy, fz: this.fz,
      signal: this.signal, event: this.event
    };
  }

  /** Return as XML attribute string */
  toXMLAttrs() {
    const d = this.d.map((v, i) => `d${i}="${v.toFixed(6)}"`).join(' ');
    return [
      `id="${this.id}"`, d,
      `px="${this.px.toFixed(4)}" py="${this.py.toFixed(4)}" pz="${this.pz.toFixed(4)}"`,
      `fx="${this.fx.toFixed(4)}" fy="${this.fy.toFixed(4)}" fz="${this.fz.toFixed(4)}"`,
      `signal="${this.signal.toFixed(4)}" event="${this.event}"`,
      `sha256="${this.sha256}"`
    ].join(' ');
  }
}

// ── Edge ──────────────────────────────────────────────────────

class TensorEdge {
  constructor(from, to, weight, type = 'attention') {
    this.from   = from;
    this.to     = to;
    this.weight = clamp(weight, 0, 1);
    this.type   = type;   // 'attention' | 'force' | 'event'
  }
  toJSON() {
    return { from: this.from, to: this.to, weight: this.weight, type: this.type };
  }
  toXMLAttrs() {
    return `from="${this.from}" to="${this.to}" weight="${this.weight.toFixed(6)}" type="${this.type}"`;
  }
}

// ── TensorGraph ───────────────────────────────────────────────

export class TensorGraph {
  /**
   * @param {object} opts
   * @param {number} opts.entityCount
   * @param {number} [opts.step]
   * @param {object} [opts.meta]     - key→value metadata dict
   */
  constructor({ entityCount, step = 0, meta = {} }) {
    this.entityCount = entityCount;
    this.step        = step;
    this.meta        = meta;
    this.nodes       = new Map();   // id → TensorNode
    this.edges       = [];          // TensorEdge[]
    this.sha256      = null;        // filled by computeHash()
  }

  // ── Mutation ───────────────────────────────────────────────

  addNode(opts) {
    const node = new TensorNode(opts);
    this.nodes.set(opts.id, node);
    return node;
  }

  addEdge(from, to, weight, type = 'attention') {
    this.edges.push(new TensorEdge(from, to, weight, type));
  }

  // ── Serialization ─────────────────────────────────────────

  /** Cluster nodes by expert (argmax of gate scores) */
  _clusters() {
    const clusters = new Map();
    for (let e = 0; e < DIM; e++) clusters.set(e, []);
    for (const node of this.nodes.values())
      clusters.get(node.expert).push(node);
    return clusters;
  }

  /** Export as SVG/XML string (compute graph format) */
  toXML() {
    const clusters = this._clusters();
    const lines = [];

    lines.push(`<?xml version="1.0" encoding="UTF-8"?>`);
    lines.push(`<tensor-graph xmlns="${NS}" version="0.1" dim="8"`
             + ` entity-count="${this.entityCount}" step="${this.step}"`
             + (this.sha256 ? ` sha256="${this.sha256}"` : '') + `>`);

    // Meta
    const metaEntries = Object.entries(this.meta);
    if (metaEntries.length) {
      lines.push('  <meta>');
      for (const [k, v] of metaEntries)
        lines.push(`    <entry k="${k}" v="${String(v).replace(/"/g, '&quot;')}"/>`);
      lines.push('  </meta>');
    }

    // Clusters
    for (const [expert, nodes] of clusters) {
      const gateMean = nodes.length
        ? (nodes.reduce((s, n) => s + n.d[expert], 0) / nodes.length).toFixed(6)
        : '0';
      const clusterHash = fnv32(nodes.map(n => n.sha256).join(''));
      lines.push(`  <cluster expert="${expert}" gate-mean="${gateMean}" sha256="${clusterHash}">`);

      for (const node of nodes)
        lines.push(`    <node ${node.toXMLAttrs()}/>`);

      // Include edges that connect nodes within this cluster
      for (const e of this.edges) {
        const fn = this.nodes.get(e.from);
        const tn = this.nodes.get(e.to);
        if (fn && tn && fn.expert === expert && tn.expert === expert)
          lines.push(`    <edge ${e.toXMLAttrs()}/>`);
      }

      lines.push('  </cluster>');
    }

    lines.push('</tensor-graph>');
    return lines.join('\n');
  }

  /** Export as structured JSON (for REST API / JS processing) */
  toJSON() {
    const clusters = {};
    for (const [expert, nodes] of this._clusters())
      clusters[expert] = nodes.map(n => n.toJSON());

    return {
      version: '0.1', dim: DIM,
      entityCount: this.entityCount,
      step: this.step,
      sha256: this.sha256,
      meta: this.meta,
      clusters,
      edges: this.edges.map(e => e.toJSON())
    };
  }

  /** Parse from XML string → TensorGraph */
  static fromXML(xmlStr) {
    const parser = new DOMParser();
    const doc    = parser.parseFromString(xmlStr, 'application/xml');
    const root   = doc.documentElement;

    const g = new TensorGraph({
      entityCount: parseInt(root.getAttribute('entity-count') || '0'),
      step:        parseInt(root.getAttribute('step') || '0'),
    });
    g.sha256 = root.getAttribute('sha256');

    // Parse meta
    for (const entry of doc.querySelectorAll('entry'))
      g.meta[entry.getAttribute('k')] = entry.getAttribute('v');

    // Parse nodes & edges
    for (const nodeEl of doc.querySelectorAll('node')) {
      const d = [];
      for (let i = 0; i < DIM; i++)
        d.push(parseFloat(nodeEl.getAttribute(`d${i}`) || '0'));

      g.addNode({
        id:     parseInt(nodeEl.getAttribute('id')),
        d,
        px:     parseFloat(nodeEl.getAttribute('px') || '0'),
        py:     parseFloat(nodeEl.getAttribute('py') || '0'),
        pz:     parseFloat(nodeEl.getAttribute('pz') || '0'),
        fx:     parseFloat(nodeEl.getAttribute('fx') || '0'),
        fy:     parseFloat(nodeEl.getAttribute('fy') || '0'),
        fz:     parseFloat(nodeEl.getAttribute('fz') || '0'),
        signal: parseFloat(nodeEl.getAttribute('signal') || '0'),
        event:  parseInt(nodeEl.getAttribute('event') || '0'),
      });
    }

    for (const edgeEl of doc.querySelectorAll('edge')) {
      g.addEdge(
        parseInt(edgeEl.getAttribute('from')),
        parseInt(edgeEl.getAttribute('to')),
        parseFloat(edgeEl.getAttribute('weight')),
        edgeEl.getAttribute('type') || 'attention'
      );
    }

    return g;
  }

  /** Parse from JSON object → TensorGraph */
  static fromJSON(obj) {
    const g = new TensorGraph({
      entityCount: obj.entityCount,
      step:        obj.step,
      meta:        obj.meta || {}
    });
    g.sha256 = obj.sha256 || null;

    for (const nodes of Object.values(obj.clusters || {}))
      for (const n of nodes) g.addNode(n);

    for (const e of (obj.edges || []))
      g.addEdge(e.from, e.to, e.weight, e.type);

    return g;
  }

  /** Compute async SHA256 of the full graph state */
  async computeHash() {
    const canonical = JSON.stringify({
      entityCount: this.entityCount,
      step: this.step,
      nodes: [...this.nodes.values()]
              .sort((a, b) => a.id - b.id)
              .map(n => n.toJSON()),
      edges: this.edges.map(e => e.toJSON())
    });
    this.sha256 = await sha256Str(canonical);
    return this.sha256;
  }

  /** 8D force propagation step (pure JS, no WASM / no REST) */
  stepJS(neighborCap = 64) {
    for (const node of this.nodes.values()) {
      // Gather attention edges from this node
      const outEdges = this.edges
        .filter(e => e.from === node.id && e.type === 'attention')
        .slice(0, neighborCap);

      if (!outEdges.length) continue;

      // Softmax already applied on weights, accumulate context
      const context = new Float64Array(DIM);
      for (const edge of outEdges) {
        const neighbor = this.nodes.get(edge.to);
        if (!neighbor) continue;
        for (let k = 0; k < DIM; k++)
          context[k] += edge.weight * neighbor.d[k];
      }

      // Force vector from context (use first 3 dims → xyz)
      node.fx = clamp(context[0] - node.px * 0.01, -1e4, 1e4);
      node.fy = clamp(context[1] - node.py * 0.01, -1e4, 1e4);
      node.fz = clamp(context[2] - node.pz * 0.01, -1e4, 1e4);

      // Update signal = argmax of accumulated context
      const maxIdx = [...context].reduce((mi, v, i, a) => v > a[mi] ? i : mi, 0);
      node.signal = maxIdx;

      // Re-hash
      node.sha256 = fnv32([node.id, ...node.d, node.px, node.py, node.pz]);
    }

    this.step++;
    this.sha256 = null;   // invalidate — call computeHash() if needed
    return this;
  }
}

// ── Builder from flat arrays (GPU readback format) ────────────

/**
 * Build a TensorGraph from GPU readback buffers.
 * All arrays must be Float32Array / Uint32Array of length entityCount.
 */
export function graphFromBuffers({
  entityCount, step = 0,
  positionArr,    // Float32Array, stride 4 (xyzw per entity)
  forceArr,       // Float32Array, stride 4
  signalArr,      // Float32Array, stride 1
  eventArr,       // Uint32Array,  stride 1
  gateScoresArr,  // Float32Array, stride 8 (8 expert scores per entity)
  meta = {}
}) {
  const g = new TensorGraph({ entityCount, step, meta });

  for (let i = 0; i < entityCount; i++) {
    const p4 = i * 4;
    const p8 = i * 8;

    const d = Array.from(gateScoresArr.subarray(p8, p8 + 8));

    g.addNode({
      id: i,
      d,
      px: positionArr[p4],
      py: positionArr[p4 + 1],
      pz: positionArr[p4 + 2],
      fx: forceArr[p4],
      fy: forceArr[p4 + 1],
      fz: forceArr[p4 + 2],
      signal: signalArr[i],
      event:  eventArr[i],
    });
  }

  return g;
}

/**
 * Build a minimal TensorGraph from a JSON snapshot
 * (as returned by the REST /graph/infer endpoint).
 */
export function graphFromSnapshot(snapshot) {
  return TensorGraph.fromJSON(snapshot);
}
