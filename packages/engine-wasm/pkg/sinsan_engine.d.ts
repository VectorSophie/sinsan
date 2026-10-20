/* tslint:disable */
/* eslint-disable */

export class WasmSearch {
    free(): void;
    [Symbol.dispose](): void;
    /**
     * Canonical legal action ids of all leaves, concatenated.
     */
    actions(): Uint32Array;
    /**
     * Legal-action logits (in actions() order) and one side-to-move value per leaf.
     */
    apply(logits: Float32Array, values: Float32Array): void;
    bestMove(): string | undefined;
    /**
     * u8 [n, 4, 90] piece codes.
     */
    boards(): Uint8Array;
    /**
     * Collects up to `leaves` leaves to evaluate; returns how many. Read them with boards(),
     * scalars(), actions() and offsets(), then call apply().
     */
    collect(leaves: number): number;
    /**
     * [max depth, visit-weighted mean depth].
     */
    depthStats(): Float32Array;
    /**
     * JSON [[parent row, depth, uci, visits, q|null, prior, terminal], ...] - see Tree::dump.
     */
    dumpTree(max_depth: number, top_k: number, min_visits: number): string;
    /**
     * Legal moves in the current root position.
     */
    legalMoves(): string[];
    constructor(cpuct: number, fpu: number, policy_temp: number);
    /**
     * [n + 1] start of each leaf's actions.
     */
    offsets(): Uint32Array;
    principalVariation(max_len: number): string[];
    /**
     * Drops the tree (e.g. after the evaluation settings changed).
     */
    reset(): void;
    /**
     * JSON [[uci, visits, q|null, prior], ...] for the root's children.
     */
    rootChildren(): string;
    rootExpanded(): boolean;
    /**
     * Network value of the root before search, or NaN if not evaluated.
     */
    rootNetValue(): number;
    rootValue(): number;
    rootVisits(): number;
    /**
     * f32 [n, 10].
     */
    scalars(): Float32Array;
    /**
     * Sets the position, keeping the tree when it extends the previous one by up to two moves.
     */
    setPosition(fen: string, moves: string[]): void;
}

/**
 * FEN of the starting position for the given formations (names as in the web app).
 */
export function startFen(cho: string, han: string): string;

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
    readonly memory: WebAssembly.Memory;
    readonly __wbg_wasmsearch_free: (a: number, b: number) => void;
    readonly startFen: (a: number, b: number, c: number, d: number) => [number, number, number, number];
    readonly wasmsearch_actions: (a: number) => [number, number];
    readonly wasmsearch_apply: (a: number, b: number, c: number, d: number, e: number) => [number, number];
    readonly wasmsearch_bestMove: (a: number) => [number, number];
    readonly wasmsearch_boards: (a: number) => [number, number];
    readonly wasmsearch_collect: (a: number, b: number) => number;
    readonly wasmsearch_depthStats: (a: number) => [number, number];
    readonly wasmsearch_dumpTree: (a: number, b: number, c: number, d: number) => [number, number];
    readonly wasmsearch_legalMoves: (a: number) => [number, number];
    readonly wasmsearch_new: (a: number, b: number, c: number) => number;
    readonly wasmsearch_offsets: (a: number) => [number, number];
    readonly wasmsearch_principalVariation: (a: number, b: number) => [number, number];
    readonly wasmsearch_reset: (a: number) => void;
    readonly wasmsearch_rootChildren: (a: number) => [number, number];
    readonly wasmsearch_rootExpanded: (a: number) => number;
    readonly wasmsearch_rootNetValue: (a: number) => number;
    readonly wasmsearch_rootValue: (a: number) => number;
    readonly wasmsearch_rootVisits: (a: number) => number;
    readonly wasmsearch_scalars: (a: number) => [number, number];
    readonly wasmsearch_setPosition: (a: number, b: number, c: number, d: number, e: number) => [number, number];
    readonly __wbindgen_malloc: (a: number, b: number) => number;
    readonly __wbindgen_realloc: (a: number, b: number, c: number, d: number) => number;
    readonly __wbindgen_externrefs: WebAssembly.Table;
    readonly __externref_table_dealloc: (a: number) => void;
    readonly __wbindgen_free: (a: number, b: number, c: number) => void;
    readonly __externref_drop_slice: (a: number, b: number) => void;
    readonly __externref_table_alloc: () => number;
    readonly __wbindgen_start: () => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;

/**
 * Instantiates the given `module`, which can either be bytes or
 * a precompiled `WebAssembly.Module`.
 *
 * @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
 *
 * @returns {InitOutput}
 */
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
 * If `module_or_path` is {RequestInfo} or {URL}, makes a request and
 * for everything else, calls `WebAssembly.instantiate` directly.
 *
 * @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
 *
 * @returns {Promise<InitOutput>}
 */
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
