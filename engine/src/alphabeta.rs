//! Alpha-beta search with a small NNUE evaluation: the browser-student candidate (F1).
//!
//! NNUE: plain piece-square inputs per perspective (14 relative piece codes x 90 canonical
//! squares = 1,260 features), an accumulator of width H shared by both perspectives, then two
//! small dense layers. Output is a win logit for the side to move (P(win) = sigmoid(z)); scores
//! are z * SCORE_SCALE "centipawns" (SCORE_SCALE = 105 matches the fitted teacher scale).
//!
//! Search: iterative deepening negamax with PVS, a transposition table, quiescence on captures,
//! null-move pruning via the pass move (always legal unless in check), late-move reductions,
//! killer and history ordering. Game endings come from `Game` (FSF janggi rules), so bikjang,
//! double pass, the move limit and repetition resolve exactly; decisive results score as mate.

use crate::board::{Board, Move, Piece, PieceKind, Side};
use crate::game::Game;
use crate::SQUARES;

pub const NUM_FEATURES: usize = 14 * SQUARES;
pub const SCORE_SCALE: f32 = 105.0;
const MATE: i32 = 30_000;
const INF: i32 = 32_000;

pub struct Nnue {
    pub h: usize,
    l1_w: Vec<f32>, // [NUM_FEATURES][h]
    l1_b: Vec<f32>, // [h]
    l2_w: Vec<f32>, // [32][2h]
    l2_b: Vec<f32>,
    l3_w: Vec<f32>, // [32][32]
    l3_b: Vec<f32>,
    out_w: Vec<f32>, // [32]
    out_b: f32,
}

impl Nnue {
    /// File format: magic "SNNU", u32 version (1), u32 h, then little-endian f32 arrays in order
    /// l1_w, l1_b, l2_w, l2_b, l3_w, l3_b, out_w, out_b.
    pub fn from_bytes(bytes: &[u8]) -> Result<Nnue, String> {
        if bytes.len() < 12 || &bytes[..4] != b"SNNU" {
            return Err("not an SNNU file".into());
        }
        let h = u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize;
        let mut floats = bytes[12..].chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap()));
        let mut take = |n: usize| -> Result<Vec<f32>, String> {
            let v: Vec<f32> = floats.by_ref().take(n).collect();
            if v.len() == n {
                Ok(v)
            } else {
                Err("truncated SNNU file".into())
            }
        };
        let l1_w = take(NUM_FEATURES * h)?;
        let l1_b = take(h)?;
        let l2_w = take(32 * 2 * h)?;
        let l2_b = take(32)?;
        let l3_w = take(32 * 32)?;
        let l3_b = take(32)?;
        let out_w = take(32)?;
        let out_b = take(1)?[0];
        Ok(Nnue { h, l1_w, l1_b, l2_w, l2_b, l3_w, l3_b, out_w, out_b })
    }

    #[inline]
    fn feature(persp: Side, piece: Piece, sq: usize) -> usize {
        let rel = piece.kind() as usize + if piece.side() == persp { 0 } else { 7 };
        let csq = if persp == Side::Han { SQUARES - 1 - sq } else { sq };
        rel * SQUARES + csq
    }

    fn refresh(&self, board: &Board, persp: Side, acc: &mut [f32]) {
        acc.copy_from_slice(&self.l1_b);
        for (sq, &p) in board.squares.iter().enumerate() {
            if !p.is_empty() {
                let f = Self::feature(persp, p, sq) * self.h;
                for (a, w) in acc.iter_mut().zip(&self.l1_w[f..f + self.h]) {
                    *a += w;
                }
            }
        }
    }

    #[inline]
    fn add(&self, acc: &mut [f32], f: usize, sign: f32) {
        let w = &self.l1_w[f * self.h..(f + 1) * self.h];
        for (a, x) in acc.iter_mut().zip(w) {
            *a += sign * x;
        }
    }

    /// Win logit for the side to move, given both perspectives' accumulators.
    fn evaluate(&self, ours: &[f32], theirs: &[f32]) -> f32 {
        let h = self.h;
        let crelu = |x: f32| x.clamp(0.0, 1.0);
        let mut l2 = [0f32; 32];
        for (j, out) in l2.iter_mut().enumerate() {
            let w = &self.l2_w[j * 2 * h..(j + 1) * 2 * h];
            let mut s = self.l2_b[j];
            for i in 0..h {
                s += w[i] * crelu(ours[i]) + w[h + i] * crelu(theirs[i]);
            }
            *out = crelu(s);
        }
        let mut l3 = [0f32; 32];
        for (j, out) in l3.iter_mut().enumerate() {
            let w = &self.l3_w[j * 32..(j + 1) * 32];
            *out = crelu(self.l3_b[j] + w.iter().zip(&l2).map(|(a, b)| a * b).sum::<f32>());
        }
        self.out_b + self.out_w.iter().zip(&l3).map(|(a, b)| a * b).sum::<f32>()
    }
}

#[derive(Clone, Copy, Default)]
struct TtEntry {
    key: u64,
    mv: Option<Move>,
    score: i32,
    depth: i8,
    bound: u8, // 0 exact, 1 lower, 2 upper
}

pub struct SearchResult {
    pub best: Option<Move>,
    pub score: i32,
    pub depth: i32,
    pub nodes: u64,
    pub pv: Vec<Move>,
}

pub struct AlphaBeta<'a> {
    nnue: &'a Nnue,
    tt: Vec<TtEntry>,
    killers: Vec<[Option<Move>; 2]>,
    history: Vec<i32>,
    acc: Vec<[Vec<f32>; 2]>,
    nodes: u64,
    node_limit: u64,
    stopped: bool,
}

fn piece_value(p: Piece) -> i32 {
    if p.is_empty() {
        0
    } else {
        match p.kind() {
            PieceKind::General => 10_000,
            k => k.points(),
        }
    }
}

impl<'a> AlphaBeta<'a> {
    pub fn new(nnue: &'a Nnue, tt_bits: u32) -> Self {
        AlphaBeta {
            nnue,
            tt: vec![TtEntry::default(); 1 << tt_bits],
            killers: vec![[None; 2]; 256],
            history: vec![0; SQUARES * SQUARES],
            acc: Vec::new(),
            nodes: 0,
            node_limit: u64::MAX,
            stopped: false,
        }
    }

    fn push(&mut self, game: &mut Game, m: Move) {
        let board = game.board();
        let ply = self.acc.len() - 1;
        let mut next = self.acc[ply].clone();
        if !m.is_pass() {
            let (from, to) = (m.from as usize, m.to as usize);
            let moving = board.squares[from];
            let captured = board.squares[to];
            for (pi, persp) in [Side::Cho, Side::Han].into_iter().enumerate() {
                self.nnue.add(&mut next[pi], Nnue::feature(persp, moving, from), -1.0);
                self.nnue.add(&mut next[pi], Nnue::feature(persp, moving, to), 1.0);
                if !captured.is_empty() {
                    self.nnue.add(&mut next[pi], Nnue::feature(persp, captured, to), -1.0);
                }
            }
        }
        self.acc.push(next);
        game.push(m);
    }

    fn pop(&mut self, game: &mut Game) {
        self.acc.pop();
        game.pop();
    }

    fn eval(&self, game: &Game) -> i32 {
        let acc = self.acc.last().unwrap();
        let us = game.board().side.idx();
        let z = self.nnue.evaluate(&acc[us], &acc[1 - us]);
        ((z * SCORE_SCALE) as i32).clamp(-MATE + 1000, MATE - 1000)
    }

    /// Exact game end at this node (ply > 0), from the side to move's view.
    fn terminal(&self, game: &Game, ply: i32) -> Option<i32> {
        let us = game.board().side;
        let r = game.immediate_end().or_else(|| game.optional_end())?;
        Some(if r.winner == us { MATE - ply } else { -(MATE - ply) })
    }

    fn order(&self, game: &Game, moves: &mut [Move], tt_move: Option<Move>, ply: usize) {
        let board = game.board();
        let mut scored: Vec<(i32, Move)> = moves
            .iter()
            .map(|&m| {
                let s = if Some(m) == tt_move {
                    1_000_000
                } else if m.is_pass() {
                    -1_000
                } else {
                    let victim = board.squares[m.to as usize];
                    if !victim.is_empty() {
                        100_000 + 16 * piece_value(victim) - piece_value(board.squares[m.from as usize]).min(100)
                    } else if self.killers[ply.min(255)].contains(&Some(m)) {
                        90_000
                    } else {
                        self.history[m.from as usize * SQUARES + m.to as usize]
                    }
                };
                (s, m)
            })
            .collect();
        scored.sort_by(|a, b| b.0.cmp(&a.0));
        for (dst, (_, m)) in moves.iter_mut().zip(scored) {
            *dst = m;
        }
    }

    fn qsearch(&mut self, game: &mut Game, mut alpha: i32, beta: i32, ply: i32) -> i32 {
        self.nodes += 1;
        if let Some(t) = self.terminal(game, ply) {
            return t;
        }
        let in_check = game.board().in_check();
        let stand = self.eval(game);
        if !in_check {
            if stand >= beta {
                return stand;
            }
            alpha = alpha.max(stand);
        }
        let mut moves = game.legal_moves();
        if moves.is_empty() {
            return -(MATE - ply);
        }
        if !in_check {
            let board = game.board();
            moves.retain(|m| !m.is_pass() && !board.squares[m.to as usize].is_empty());
        }
        self.order(game, &mut moves, None, ply as usize);
        let mut best = if in_check { -INF } else { stand };
        for m in moves {
            self.push(game, m);
            let score = -self.qsearch(game, -beta, -alpha, ply + 1);
            self.pop(game);
            if score > best {
                best = score;
                if score > alpha {
                    alpha = score;
                    if alpha >= beta {
                        break;
                    }
                }
            }
        }
        best
    }

    fn search(&mut self, game: &mut Game, depth: i32, mut alpha: i32, beta: i32, ply: i32, allow_null: bool) -> i32 {
        if depth <= 0 {
            return self.qsearch(game, alpha, beta, ply);
        }
        self.nodes += 1;
        if self.nodes >= self.node_limit {
            self.stopped = true;
            return 0;
        }
        if ply > 0 {
            if let Some(t) = self.terminal(game, ply) {
                return t;
            }
        }
        let key = game.board().key;
        let slot = (key as usize) & (self.tt.len() - 1);
        let entry = self.tt[slot];
        let tt_move = (entry.key == key).then_some(entry.mv).flatten();
        if entry.key == key && entry.depth as i32 >= depth && ply > 0 {
            let s = entry.score;
            match entry.bound {
                0 => return s,
                1 if s >= beta => return s,
                2 if s <= alpha => return s,
                _ => {}
            }
        }
        let in_check = game.board().in_check();
        let mut moves = game.legal_moves();
        if moves.is_empty() {
            return -(MATE - ply);
        }

        // Null move = pass (a legal Janggi move whenever not in check). Skip right after a pass,
        // since a second pass ends the game on points.
        let last_was_pass = game.moves().last().is_some_and(|m| m.is_pass());
        if allow_null && !in_check && depth >= 3 && ply > 0 && !last_was_pass && beta.abs() < MATE - 1000 {
            let static_eval = self.eval(game);
            if static_eval >= beta {
                let g = game.board().generals[game.board().side.idx()];
                let pass = Move { from: g, to: g };
                if moves.contains(&pass) {
                    self.push(game, pass);
                    let score = -self.search(game, depth - 3, -beta, -beta + 1, ply + 1, false);
                    self.pop(game);
                    if self.stopped {
                        return 0;
                    }
                    if score >= beta {
                        return score;
                    }
                }
            }
        }

        self.order(game, &mut moves, tt_move, ply as usize);
        let original_alpha = alpha;
        let mut best = -INF;
        let mut best_move = None;
        for (i, &m) in moves.iter().enumerate() {
            let quiet = m.is_pass() || game.board().squares[m.to as usize].is_empty();
            self.push(game, m);
            let gives_check = game.board().in_check();
            let mut score;
            if i == 0 {
                score = -self.search(game, depth - 1, -beta, -alpha, ply + 1, true);
            } else {
                let reduce = if depth >= 3 && i >= 4 && quiet && !in_check && !gives_check { 1 + (i >= 12) as i32 } else { 0 };
                score = -self.search(game, depth - 1 - reduce, -alpha - 1, -alpha, ply + 1, true);
                if score > alpha && (reduce > 0 || score < beta) {
                    score = -self.search(game, depth - 1, -beta, -alpha, ply + 1, true);
                }
            }
            self.pop(game);
            if self.stopped {
                return 0;
            }
            if score > best {
                best = score;
                best_move = Some(m);
                if score > alpha {
                    alpha = score;
                    if alpha >= beta {
                        if quiet {
                            let k = &mut self.killers[(ply as usize).min(255)];
                            if k[0] != Some(m) {
                                k[1] = k[0];
                                k[0] = Some(m);
                            }
                            if !m.is_pass() {
                                self.history[m.from as usize * SQUARES + m.to as usize] += depth * depth;
                            }
                        }
                        break;
                    }
                }
            }
        }
        let bound = if best <= original_alpha { 2 } else if best >= beta { 1 } else { 0 };
        self.tt[slot] = TtEntry { key, mv: best_move, score: best, depth: depth as i8, bound };
        best
    }

    fn pv(&self, game: &mut Game, max_len: usize) -> Vec<Move> {
        let mut pv = Vec::new();
        let mut pushed = 0;
        while pv.len() < max_len {
            let key = game.board().key;
            let e = self.tt[(key as usize) & (self.tt.len() - 1)];
            let Some(m) = (e.key == key).then_some(e.mv).flatten() else { break };
            if !game.legal_moves().contains(&m) {
                break;
            }
            pv.push(m);
            game.push(m);
            pushed += 1;
        }
        for _ in 0..pushed {
            game.pop();
        }
        pv
    }

    /// Iterative deepening up to `max_depth` or until `node_limit` nodes.
    pub fn think(&mut self, game: &mut Game, max_depth: i32, node_limit: u64) -> SearchResult {
        self.nodes = 0;
        self.node_limit = node_limit;
        self.stopped = false;
        let mut acc = [vec![0f32; self.nnue.h], vec![0f32; self.nnue.h]];
        self.nnue.refresh(game.board(), Side::Cho, &mut acc[0]);
        self.nnue.refresh(game.board(), Side::Han, &mut acc[1]);
        self.acc = vec![acc];
        let mut result = SearchResult { best: None, score: 0, depth: 0, nodes: 0, pv: Vec::new() };
        for depth in 1..=max_depth {
            let score = self.search(game, depth, -INF, INF, 0, false);
            if self.stopped && result.best.is_some() {
                break;
            }
            let key = game.board().key;
            let e = self.tt[(key as usize) & (self.tt.len() - 1)];
            if e.key == key && e.mv.is_some() {
                result.best = e.mv;
                result.score = score;
                result.depth = depth;
            }
            if self.stopped || score.abs() >= MATE - 100 {
                break;
            }
        }
        result.pv = self.pv(game, 16);
        result.nodes = self.nodes;
        if result.best.is_none() {
            result.best = game.legal_moves().first().copied();
        }
        result
    }
}
