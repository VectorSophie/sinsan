/** Main-thread side of the in-browser Sinsan (student-worker.ts): one shared worker, created lazily. */
import type { FromWorker, StudentRequest } from './student-worker';

export interface StudentInfo {
  meta: Record<string, unknown>;
  threads: number;
}

let worker: Worker | null = null;
let nextId = 1;
const waiting = new Map<number, { resolve: (r: unknown) => void; reject: (e: Error) => void }>();
let readyPromise: Promise<StudentInfo> | null = null;
let readyResolve: ((i: StudentInfo) => void) | null = null;
let readyReject: ((e: Error) => void) | null = null;

function start(): Worker {
  if (worker) return worker;
  worker = new Worker(new URL('./student-worker.ts', import.meta.url), { type: 'module' });
  worker.onmessage = (ev: MessageEvent<FromWorker>) => {
    const msg = ev.data;
    if (msg.type === 'ready') {
      readyResolve?.({ meta: msg.meta, threads: msg.threads });
    } else if (msg.type === 'move') {
      waiting.get(msg.id)?.resolve(msg.reply);
      waiting.delete(msg.id);
    } else if (msg.id !== undefined) {
      waiting.get(msg.id)?.reject(new Error(msg.message));
      waiting.delete(msg.id);
    } else {
      readyReject?.(new Error(msg.message));
      readyPromise = null;
    }
  };
  return worker;
}

/** Loads the engine and model (once); resolves with the model's metadata. */
export function studentReady(): Promise<StudentInfo> {
  readyPromise ??= new Promise<StudentInfo>((resolve, reject) => {
    readyResolve = resolve;
    readyReject = reject;
    start().postMessage({ type: 'init' });
  });
  return readyPromise;
}

/** Same request and reply as POST /api/move, answered in the browser. */
export function studentMove<T>(req: StudentRequest): Promise<T> {
  const id = nextId++;
  return new Promise<T>((resolve, reject) => {
    waiting.set(id, { resolve: resolve as (r: unknown) => void, reject });
    start().postMessage({ type: 'move', id, req });
  });
}
