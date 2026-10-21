/// <reference lib="webworker" />
/**
 * Sinsan in the browser: the Rust engine's search (WebAssembly) driven by the distilled student
 * network (onnxruntime-web). Answers the same requests as training/play_server.py's /api/move with
 * the same reply shape, so the rest of the app cannot tell the two apart.
 */
import initEngine, { WasmSearch, startFen } from '@sinsan/engine-wasm';
import * as ort from 'onnxruntime-web/wasm';
import ortWasmUrl from 'onnxruntime-web/ort-wasm-simd-threaded.wasm?url';
import ortMjsUrl from 'onnxruntime-web/ort-wasm-simd-threaded.mjs?url';

const MODEL_URL = `${import.meta.env.BASE_URL}student/student.onnx`;
const META_URL = `${import.meta.env.BASE_URL}student/student.json`;
const CP_SCALE = 105; // as uci_engine.value_to_cp: P(win) = sigmoid(cp / 105)
const LEAVES = 8; // leaves evaluated per network call
const MAX_VISITS = 20000;
const MAX_TIME_MS = 60000;

export interface StudentRequest {
  cho: string;
  han: string;
  moves: string[];
  visits?: number;
  time_ms?: number;
  cpuct?: number;
  fpu?: number;
  ptemp?: number;
  util?: number;
  temperature?: number;
  temp_plies?: number;
}

export type ToWorker = { type: 'init' } | { type: 'move'; id: number; req: StudentRequest };
export type FromWorker =
  | { type: 'ready'; meta: Record<string, unknown>; threads: number }
  | { type: 'error'; id?: number; message: string }
  | { type: 'move'; id: number; reply: Record<string, unknown> };

let session: ort.InferenceSession | null = null;
let search: WasmSearch | null = null;
let searchParams = '';
let util = 0;
let ready: Promise<void> | null = null;

async function init(): Promise<void> {
  await initEngine();
  // Both files explicitly: without the .mjs, onnxruntime's thread workers would load this bundled
  // worker script instead of its own glue code, and each one would start a second engine.
  ort.env.wasm.wasmPaths = { wasm: ortWasmUrl, mjs: ortMjsUrl };
  // Threads need cross-origin isolation (COOP/COEP headers); without it onnxruntime runs on one.
  const threads = self.crossOriginIsolated ? Math.max(1, Math.min(8, (navigator.hardwareConcurrency || 4) - 1)) : 1;
  ort.env.wasm.numThreads = threads;
  const [model, meta] = await Promise.all([
    fetch(MODEL_URL).then((r) => {
      if (!r.ok) throw new Error(`student model missing (${r.status})`);
      return r.arrayBuffer();
    }),
    fetch(META_URL).then((r) => (r.ok ? r.json() : {})),
  ]);
  session = await ort.InferenceSession.create(model, { executionProviders: ['wasm'], graphOptimizationLevel: 'all' });
  post({ type: 'ready', meta, threads });
}

function post(msg: FromWorker): void {
  self.postMessage(msg);
}

/** One network call over the collected leaves: legal-action logits and side-to-move values. */
async function evaluate(s: WasmSearch, n: number): Promise<void> {
  const boards = new ort.Tensor('uint8', s.boards(), [n, 4, 90]);
  const scalars = new ort.Tensor('float32', s.scalars(), [n, 10]);
  const out = await session!.run({ boards, scalars });
  const policy = out['policy']!.data as Float32Array;
  const wdl = out['wdl']!.data as Float32Array;
  const margin = out['margin_mean']!.data as Float32Array;
  const actions = s.actions();
  const offsets = s.offsets();
  const width = policy.length / n;
  const logits = new Float32Array(actions.length);
  const values = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    for (let j = offsets[i]!; j < offsets[i + 1]!; j++) logits[j] = policy[i * width + actions[j]!]!;
    let v = wdl[i * 3]! - wdl[i * 3 + 2]!;
    if (util) v = Math.max(-1, Math.min(1, v + util * Math.tanh(margin[i]! / 20)));
    values[i] = v;
  }
  s.apply(logits, values);
}

const valueToCp = (v: number): number => {
  const p = Math.min(Math.max((v + 1) / 2, 1e-4), 1 - 1e-4);
  return Math.round(CP_SCALE * Math.log(p / (1 - p)));
};

type Child = [string, number, number | null, number];

async function move(req: StudentRequest): Promise<Record<string, unknown>> {
  const params = [req.cpuct ?? 1.2, req.fpu ?? 0.33, req.ptemp ?? 1.36] as const;
  const key = params.join(',');
  if (!search || key !== searchParams) {
    search?.free();
    search = new WasmSearch(...params);
    searchParams = key;
  }
  const s = search;
  const u = Math.max(0, Math.min(1, req.util ?? 0));
  if (u !== util) {
    util = u;
    s.reset(); // cached values were computed with the old utility
  }
  const fen = startFen(req.cho, req.han);
  s.setPosition(fen, req.moves);

  const budget = req.time_ms ? Math.max(50, Math.min(MAX_TIME_MS, req.time_ms)) : null;
  const visits = req.visits !== undefined || budget === null ? Math.max(1, Math.min(MAX_VISITS, req.visits ?? 800)) : MAX_VISITS;
  const start = performance.now();
  for (;;) {
    const n = s.collect(LEAVES);
    if (n) await evaluate(s, n);
    if (!s.rootExpanded()) {
      if (!n) break; // terminal root
      continue;
    }
    const total = s.rootVisits();
    const elapsed = performance.now() - start;
    if (total >= visits) break;
    if (budget !== null && elapsed >= budget) break;
    // With a time limit, stop once the best move cannot change in the remaining time.
    if (budget !== null && req.visits === undefined && elapsed > 100) {
      const top = (JSON.parse(s.rootChildren()) as Child[]).map((c) => c[1]).sort((a, b) => b - a);
      if (top.length > 1 && top[0]! - top[1]! > (total / elapsed) * (budget - elapsed)) break;
    }
  }
  const ms = performance.now() - start;

  const children = (JSON.parse(s.rootChildren()) as Child[]).sort((a, b) => b[1] - a[1] || b[3] - a[3]);
  let best = s.bestMove() ?? children[0]?.[0] ?? s.legalMoves()[0] ?? '0000';
  if (children.length && children[0]![1] === 0) best = children[0]![0]; // too few visits: the network's top choice
  let chosen = best;
  const temperature = req.temperature ?? 0;
  if (temperature > 0 && req.moves.length < (req.temp_plies ?? 1000) && children.length) {
    const searched = children[0]![1] > 0;
    const w = children.map((c) => (searched ? c[1] : c[3]) ** (1 / temperature));
    let r = Math.random() * w.reduce((a, b) => a + b, 0);
    chosen = children.find((_, i) => (r -= w[i]!) <= 0)?.[0] ?? best;
  }
  const value = s.rootValue();
  const total = s.rootVisits();
  const [depthMax, depthMean] = s.depthStats() as unknown as [number, number];
  const netValue = s.rootNetValue();
  return {
    move: chosen, best, cp: valueToCp(value), value, net_value: Number.isNaN(netValue) ? null : netValue,
    visits: total, ms, depth: Math.round(depthMean * 10) / 10, seldepth: depthMax,
    pv: s.principalVariation(16), legal: children.length, net: 'browser',
    candidates: children.slice(0, 12).map(([m, n, q, prior]) => ({ move: m, visits: n, q, prior })),
    tree: JSON.parse(s.dumpTree(6, 4, Math.max(2, Math.floor(total / 300)))),
  };
}

// Requests are handled one at a time, in order.
let queue: Promise<void> = Promise.resolve();
self.onmessage = (ev: MessageEvent<ToWorker>) => {
  const msg = ev.data;
  if (msg.type === 'init') {
    ready ??= init().catch((e) => {
      ready = null;
      post({ type: 'error', message: String(e) });
    });
    return;
  }
  queue = queue.then(async () => {
    try {
      ready ??= init();
      await ready;
      post({ type: 'move', id: msg.id, reply: await move(msg.req) });
    } catch (e) {
      post({ type: 'error', id: msg.id, message: String(e) });
    }
  });
};
