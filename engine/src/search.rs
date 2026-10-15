//! Batched PUCT search: many independent game trees advanced together, with network evaluation
//! done outside (Python/GPU) in large batches.
//!
//! Protocol per iteration:
//!   1. `collect` walks each tree (PUCT with virtual loss) to up to `leaves_per_tree` new leaves,
//!      resolving terminal positions in place, and returns the leaves' encoded features plus each
//!      leaf's canonical legal actions (in child order).
//!   2. The caller evaluates them and calls `apply` with the legal-action logits (flattened in the
//!      same order) and one value per leaf (side to move's view, in [-1, 1]).
//!
//! Value bookkeeping: `Node::w` sums values from the view of the player who moved into the node,
//! so a parent picks children by their own w / n.

use crate::action::encode_move;
use crate::board::Move;
use crate::features::{canonical_action, encode, Encoded};
use crate::game::Game;

#[derive(Clone, Copy, Debug)]
pub struct SearchParams {
    pub cpuct: f32,
    pub cpuct_base: f32,
    pub cpuct_factor: f32,
    /// First-play urgency: unvisited children get Q = parent Q - reduction * sqrt(visited prior mass).
    pub fpu_reduction: f32,
    pub fpu_reduction_root: f32,
    /// Softmax temperature applied to the policy logits when setting priors.
    pub policy_temp: f32,
}

impl Default for SearchParams {
    fn default() -> Self {
        // Lc0's chess defaults as a starting point; tuned for Janggi by SPRT later.
        SearchParams { cpuct: 1.745, cpuct_base: 38739.0, cpuct_factor: 3.894, fpu_reduction: 0.33, fpu_reduction_root: 0.33, policy_temp: 1.36 }
    }
}

const UNEXPANDED: u8 = 0;
const PENDING: u8 = 1;
const EXPANDED: u8 = 2;
const TERMINAL: u8 = 3;

#[derive(Clone, Debug)]
struct Node {
    mv: Move,
    prior: f32,
    n: u32,
    w: f32,
    vl: u32,
    first_child: u32,
    num_children: u16,
    state: u8,
    /// For TERMINAL nodes: the value for the side to move at this node.
    terminal_value: f32,
}

impl Node {
    fn new(mv: Move, prior: f32) -> Node {
        Node { mv, prior, n: 0, w: 0.0, vl: 0, first_child: 0, num_children: 0, state: UNEXPANDED, terminal_value: 0.0 }
    }
}

pub struct Tree {
    pub game: Game,
    nodes: Vec<Node>,
    params: SearchParams,
}

pub struct Leaf {
    pub path: Vec<u32>,
    pub encoded: Encoded,
    pub actions: Vec<u32>,
}

impl Tree {
    pub fn new(game: Game, params: SearchParams) -> Tree {
        Tree { game, nodes: vec![Node::new(Move { from: 0, to: 0 }, 1.0)], params }
    }

    pub fn root_visits(&self) -> u32 {
        self.nodes[0].n
    }

    /// Value of the root position for its side to move.
    pub fn root_value(&self) -> f32 {
        let r = &self.nodes[0];
        if r.n == 0 {
            0.0
        } else {
            -r.w / r.n as f32
        }
    }

    /// (move, visits, Q for the mover, prior) for each root child.
    pub fn root_children(&self) -> Vec<(Move, u32, f32, f32)> {
        let r = &self.nodes[0];
        (0..r.num_children as usize)
            .map(|i| {
                let c = &self.nodes[r.first_child as usize + i];
                (c.mv, c.n, if c.n > 0 { c.w / c.n as f32 } else { f32::NAN }, c.prior)
            })
            .collect()
    }

    pub fn best_move(&self) -> Option<Move> {
        self.root_children()
            .into_iter()
            .max_by(|a, b| a.1.cmp(&b.1).then(a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal)))
            .map(|c| c.0)
    }

    fn cpuct(&self, parent_n: u32) -> f32 {
        let p = &self.params;
        p.cpuct + p.cpuct_factor * ((parent_n as f32 + p.cpuct_base) / p.cpuct_base).ln()
    }

    fn select_child(&self, idx: usize, is_root: bool) -> usize {
        let node = &self.nodes[idx];
        let parent_n = node.n + node.vl;
        let parent_q = if node.n > 0 { -node.w / node.n as f32 } else { 0.0 };
        let first = node.first_child as usize;
        let children = &self.nodes[first..first + node.num_children as usize];
        let visited_mass: f32 = children.iter().filter(|c| c.n + c.vl > 0).map(|c| c.prior).sum();
        let reduction = if is_root { self.params.fpu_reduction_root } else { self.params.fpu_reduction };
        let fpu = parent_q - reduction * visited_mass.sqrt();
        let u_scale = self.cpuct(parent_n) * (parent_n.max(1) as f32).sqrt();
        let mut best = (first, f32::NEG_INFINITY);
        for (i, c) in children.iter().enumerate() {
            let n = c.n + c.vl;
            let q = if n > 0 { (c.w - c.vl as f32) / n as f32 } else { fpu };
            let score = q + u_scale * c.prior / (1.0 + n as f32);
            if score > best.1 {
                best = (first + i, score);
            }
        }
        best.0
    }

    /// Adds `value` (side to move at the leaf's view) along `path`, removing virtual loss if asked.
    fn backup(&mut self, path: &[u32], mut value: f32, remove_vl: bool) {
        for &idx in path.iter().rev() {
            let node = &mut self.nodes[idx as usize];
            // w is from the view of the player who moved into this node: the leaf's opponent first.
            value = -value;
            node.w += value;
            node.n += 1;
            if remove_vl {
                node.vl -= 1;
            }
        }
    }

    /// Walks to one new leaf. Terminal leaves are backed up immediately (returns None); a leaf
    /// already awaiting evaluation is a collision (returns None, no state change).
    fn select_leaf(&mut self) -> Option<Leaf> {
        let mut path = vec![0u32];
        let mut idx = 0usize;
        let mut pushed = 0;
        loop {
            match self.nodes[idx].state {
                EXPANDED => {
                    let child = self.select_child(idx, idx == 0);
                    self.game.push(self.nodes[child].mv);
                    pushed += 1;
                    path.push(child as u32);
                    idx = child;
                }
                _ => break,
            }
        }
        let state = self.nodes[idx].state;
        let leaf = if state == TERMINAL {
            let v = self.nodes[idx].terminal_value;
            self.backup(&path, v, false);
            None
        } else if state == PENDING {
            None
        } else {
            let us = self.game.board().side;
            let legal = self.game.legal_moves();
            let result = if legal.is_empty() {
                Some(us.other())
            } else {
                self.game.immediate_end().or_else(|| self.game.optional_end()).map(|r| r.winner)
            };
            if let Some(winner) = result {
                let v = if winner == us { 1.0 } else { -1.0 };
                let node = &mut self.nodes[idx];
                node.state = TERMINAL;
                node.terminal_value = v;
                self.backup(&path, v, false);
                None
            } else {
                let first = self.nodes.len() as u32;
                let mut actions = Vec::with_capacity(legal.len());
                for &m in &legal {
                    self.nodes.push(Node::new(m, 0.0));
                    actions.push(canonical_action(encode_move(m).expect("legal move has a template"), us) as u32);
                }
                let node = &mut self.nodes[idx];
                node.first_child = first;
                node.num_children = legal.len() as u16;
                node.state = PENDING;
                for &p in &path {
                    self.nodes[p as usize].vl += 1;
                }
                let plies = self.game.plies();
                Some(Leaf { path, encoded: encode(&self.game, plies), actions })
            }
        };
        for _ in 0..pushed {
            self.game.pop();
        }
        leaf
    }

    /// Collects up to `max_leaves` leaves needing evaluation, stopping early after repeated
    /// collisions or when no further progress is possible.
    pub fn collect(&mut self, max_leaves: usize, max_attempts: usize) -> Vec<Leaf> {
        let mut leaves = Vec::new();
        for _ in 0..max_attempts {
            if leaves.len() >= max_leaves {
                break;
            }
            if let Some(leaf) = self.select_leaf() {
                leaves.push(leaf);
            }
        }
        leaves
    }

    /// Sets priors from legal-action logits and backs up the value for one evaluated leaf.
    pub fn apply(&mut self, leaf: &Leaf, logits: &[f32], value: f32) {
        let idx = *leaf.path.last().unwrap() as usize;
        let (first, count) = (self.nodes[idx].first_child as usize, self.nodes[idx].num_children as usize);
        debug_assert_eq!(count, logits.len());
        let t = self.params.policy_temp;
        let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = logits.iter().map(|&l| ((l - max) / t).exp()).collect();
        let sum: f32 = exps.iter().sum();
        for (i, e) in exps.iter().enumerate() {
            self.nodes[first + i].prior = e / sum;
        }
        self.nodes[idx].state = EXPANDED;
        self.backup(&leaf.path, value, true);
    }

    /// Mixes Dirichlet-style noise into the root priors (self-play exploration).
    pub fn add_root_noise(&mut self, noise: &[f32], fraction: f32) {
        let r = &self.nodes[0];
        let first = r.first_child as usize;
        for (i, &z) in noise.iter().enumerate().take(r.num_children as usize) {
            let c = &mut self.nodes[first + i];
            c.prior = (1.0 - fraction) * c.prior + fraction * z;
        }
    }

    pub fn root_expanded(&self) -> bool {
        self.nodes[0].state == EXPANDED
    }

    pub fn root_terminal(&self) -> bool {
        self.nodes[0].state == TERMINAL
    }
}
