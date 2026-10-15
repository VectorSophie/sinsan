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
    /// Network value of the root position (its side to move), once evaluated.
    root_net_value: Option<f32>,
    gumbel: Option<GumbelRoot>,
}

/// Gumbel root search state (Danihelka et al. 2022): sequential halving over m sampled actions.
#[derive(Clone, Debug)]
struct GumbelRoot {
    /// Per root child (by offset): Gumbel noise and log prior.
    g: Vec<f32>,
    logp: Vec<f32>,
    /// Remaining candidates (root child offsets).
    remaining: Vec<usize>,
    /// Cumulative visit target per remaining candidate in the current phase.
    target: u32,
    budget: u32,
    /// Root visits when the search started (tree reuse may carry visits over).
    start_n: u32,
    phases: u32,
    c_visit: f32,
    c_scale: f32,
}

pub struct Leaf {
    pub path: Vec<u32>,
    pub encoded: Encoded,
    pub actions: Vec<u32>,
}

impl Tree {
    pub fn new(game: Game, params: SearchParams) -> Tree {
        Tree { game, nodes: vec![Node::new(Move { from: 0, to: 0 }, 1.0)], params, root_net_value: None, gumbel: None }
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
                    let child = if idx == 0 && self.gumbel.is_some() {
                        match self.gumbel_pick() {
                            Some(c) => c,
                            None => return None,
                        }
                    } else {
                        self.select_child(idx, idx == 0)
                    };
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
        if idx == 0 {
            self.root_net_value = Some(value);
        }
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

    /// Plays `mv` at the root, keeping the searched subtree below it when there is one (tree
    /// reuse); otherwise the tree restarts from the new position.
    pub fn advance(&mut self, mv: Move) {
        let root = &self.nodes[0];
        let child = if root.state == EXPANDED {
            (0..root.num_children as usize).map(|i| root.first_child as usize + i).find(|&i| self.nodes[i].mv == mv)
        } else {
            None
        };
        self.game.push(mv);
        self.gumbel = None;
        self.root_net_value = None;
        let Some(child) = child else {
            self.nodes = vec![Node::new(Move { from: 0, to: 0 }, 1.0)];
            return;
        };
        if self.nodes[child].state == PENDING || self.nodes.iter().any(|n| n.vl > 0) {
            // Only reuse between batches, never with evaluations in flight.
            self.nodes = vec![Node::new(Move { from: 0, to: 0 }, 1.0)];
            return;
        }
        // Breadth-first copy of the subtree; children stay contiguous.
        let mut new_nodes = vec![self.nodes[child].clone()];
        let mut queue = std::collections::VecDeque::from([(child, 0usize)]);
        while let Some((old, new)) = queue.pop_front() {
            let (first, count) = (self.nodes[old].first_child as usize, self.nodes[old].num_children as usize);
            if count == 0 || self.nodes[old].state != EXPANDED && self.nodes[old].state != TERMINAL {
                new_nodes[new].num_children = 0;
                new_nodes[new].first_child = 0;
                if new_nodes[new].state != TERMINAL {
                    new_nodes[new].state = UNEXPANDED;
                }
                continue;
            }
            let start = new_nodes.len();
            new_nodes[new].first_child = start as u32;
            for i in 0..count {
                new_nodes.push(self.nodes[first + i].clone());
                queue.push_back((first + i, start + i));
            }
        }
        self.nodes = new_nodes;
    }

    /// Starts Gumbel sequential halving at an expanded root: picks the top `m` children by
    /// g + log prior and splits `budget` visits over ceil(log2 m) halving phases.
    pub fn gumbel_start(&mut self, m: usize, budget: u32, noise: &[f32], c_visit: f32, c_scale: f32) {
        let r = &self.nodes[0];
        assert!(r.state == EXPANDED, "gumbel_start needs an expanded root");
        let (first, count) = (r.first_child as usize, r.num_children as usize);
        let logp: Vec<f32> = (0..count).map(|i| self.nodes[first + i].prior.max(1e-12).ln()).collect();
        let g: Vec<f32> = (0..count).map(|i| noise.get(i).copied().unwrap_or(0.0)).collect();
        let mut order: Vec<usize> = (0..count).collect();
        order.sort_by(|&a, &b| (g[b] + logp[b]).partial_cmp(&(g[a] + logp[a])).unwrap());
        order.truncate(m.max(1).min(count));
        let phases = (order.len() as f32).log2().ceil().max(1.0) as u32;
        let per = (budget / (phases * order.len() as u32)).max(1);
        let base = order.iter().map(|&i| self.nodes[first + i].n).max().unwrap_or(0);
        let start_n = self.nodes[0].n;
        self.gumbel = Some(GumbelRoot { g, logp, remaining: order, target: base + per, budget, start_n, phases, c_visit, c_scale });
    }

    fn sigma(&self, gr: &GumbelRoot, q: f32) -> f32 {
        let r = &self.nodes[0];
        let first = r.first_child as usize;
        let max_n = (0..r.num_children as usize).map(|i| self.nodes[first + i].n).max().unwrap_or(0);
        (gr.c_visit + max_n as f32) * gr.c_scale * (q + 1.0) / 2.0
    }

    fn child_q(&self, offset: usize) -> Option<f32> {
        let c = &self.nodes[self.nodes[0].first_child as usize + offset];
        (c.n > 0).then(|| c.w / c.n as f32)
    }

    /// Root child to visit next under sequential halving, or None while the phase waits for
    /// in-flight evaluations (or the search is finished).
    fn gumbel_pick(&mut self) -> Option<usize> {
        let first = self.nodes[0].first_child as usize;
        loop {
            let gr = self.gumbel.as_ref()?;
            if gr.remaining.len() <= 1 || self.nodes[0].n >= gr.start_n + gr.budget {
                return None;
            }
            let pick = gr
                .remaining
                .iter()
                .copied()
                .filter(|&i| self.nodes[first + i].n + self.nodes[first + i].vl < gr.target)
                .min_by_key(|&i| self.nodes[first + i].n + self.nodes[first + i].vl);
            if let Some(i) = pick {
                return Some(first + i);
            }
            if gr.remaining.iter().any(|&i| self.nodes[first + i].vl > 0) {
                return None;
            }
            // Phase complete: keep the better half by g + log p + sigma(q).
            let mut gr = self.gumbel.take().unwrap();
            let mut scored: Vec<(f32, usize)> = gr
                .remaining
                .iter()
                .map(|&i| (gr.g[i] + gr.logp[i] + self.sigma(&gr, self.child_q(i).unwrap_or(-1.0)), i))
                .collect();
            scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
            let keep = scored.len().div_ceil(2);
            gr.remaining = scored.into_iter().take(keep).map(|x| x.1).collect();
            let per = (gr.budget / (gr.phases * gr.remaining.len() as u32)).max(1);
            gr.target += per;
            self.gumbel = Some(gr);
        }
    }

    pub fn gumbel_done(&self) -> bool {
        match &self.gumbel {
            None => true,
            Some(gr) => gr.remaining.len() <= 1 || self.nodes[0].n >= gr.start_n + gr.budget,
        }
    }

    /// (chosen move, improved policy over root children, search value). The improved policy is
    /// softmax(log p + sigma(completed Q)), with unvisited children completed by v_mix.
    pub fn gumbel_result(&self) -> Option<(Move, Vec<(Move, f32)>, f32)> {
        let gr = self.gumbel.as_ref()?;
        let r = &self.nodes[0];
        let (first, count) = (r.first_child as usize, r.num_children as usize);
        let net_v = self.root_net_value.unwrap_or(0.0);
        let (mut num, mut den, mut total_n) = (0.0f32, 0.0f32, 0u32);
        for i in 0..count {
            let c = &self.nodes[first + i];
            total_n += c.n;
            if c.n > 0 {
                num += c.prior * c.w / c.n as f32;
                den += c.prior;
            }
        }
        let v_mix = if den > 0.0 { (net_v + total_n as f32 * num / den) / (1.0 + total_n as f32) } else { net_v };
        let scores: Vec<f32> =
            (0..count).map(|i| gr.logp[i] + self.sigma(gr, self.child_q(i).unwrap_or(v_mix))).collect();
        let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = scores.iter().map(|s| (s - max).exp()).collect();
        let sum: f32 = exps.iter().sum();
        let policy = (0..count).map(|i| (self.nodes[first + i].mv, exps[i] / sum)).collect();
        let best = gr
            .remaining
            .iter()
            .copied()
            .max_by(|&a, &b| {
                let sa = gr.g[a] + gr.logp[a] + self.sigma(gr, self.child_q(a).unwrap_or(-1.0));
                let sb = gr.g[b] + gr.logp[b] + self.sigma(gr, self.child_q(b).unwrap_or(-1.0));
                sa.partial_cmp(&sb).unwrap()
            })
            .unwrap();
        Some((self.nodes[first + best].mv, policy, self.root_value()))
    }

    pub fn principal_variation(&self, max_len: usize) -> Vec<Move> {
        let mut pv = Vec::new();
        let mut idx = 0usize;
        while pv.len() < max_len && self.nodes[idx].state == EXPANDED {
            let (first, count) = (self.nodes[idx].first_child as usize, self.nodes[idx].num_children as usize);
            let Some(best) = (first..first + count).filter(|&i| self.nodes[i].n > 0).max_by_key(|&i| self.nodes[i].n) else { break };
            pv.push(self.nodes[best].mv);
            idx = best;
        }
        pv
    }

    pub fn root_expanded(&self) -> bool {
        self.nodes[0].state == EXPANDED
    }

    pub fn root_terminal(&self) -> bool {
        self.nodes[0].state == TERMINAL
    }
}
