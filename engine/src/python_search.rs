//! Python bindings for batched search (`sinsan_engine.SearchBatch`).

use crate::board::{parse_uci, Formation};
use crate::features::{HISTORY, NUM_SCALARS};
use crate::game::Game;
use crate::search::{Leaf, SearchParams, Tree};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};
use rayon::prelude::*;

fn formation(name: &str) -> PyResult<Formation> {
    Formation::from_name(name).ok_or_else(|| PyValueError::new_err(format!("unknown formation {name:?}")))
}

#[pyclass(name = "SearchBatch", module = "sinsan_engine", unsendable)]
pub struct PySearchBatch {
    params: SearchParams,
    trees: Vec<Option<Tree>>,
    pending: Vec<(usize, Leaf)>,
}

#[pymethods]
impl PySearchBatch {
    /// SearchBatch(cpuct=1.745, cpuct_base=38739, cpuct_factor=3.894, fpu_reduction=0.33,
    ///             fpu_reduction_root=0.33, policy_temp=1.36)
    #[new]
    #[pyo3(signature = (cpuct=1.745, cpuct_base=38739.0, cpuct_factor=3.894, fpu_reduction=0.33, fpu_reduction_root=0.33, policy_temp=1.36))]
    fn new(cpuct: f32, cpuct_base: f32, cpuct_factor: f32, fpu_reduction: f32, fpu_reduction_root: f32, policy_temp: f32) -> Self {
        let params = SearchParams { cpuct, cpuct_base, cpuct_factor, fpu_reduction, fpu_reduction_root, policy_temp };
        PySearchBatch { params, trees: Vec::new(), pending: Vec::new() }
    }

    /// Starts a fresh search in `slot` (growing the batch as needed) from a game given by its
    /// formations and move list.
    fn set_root(&mut self, slot: usize, cho: &str, han: &str, moves: Vec<String>) -> PyResult<()> {
        let mut game = Game::startpos(formation(cho)?, formation(han)?);
        for m in &moves {
            let mv = parse_uci(m).ok_or_else(|| PyValueError::new_err(format!("bad move {m:?}")))?;
            game.push(mv);
        }
        if slot >= self.trees.len() {
            self.trees.resize_with(slot + 1, || None);
        }
        self.trees[slot] = Some(Tree::new(game, self.params));
        Ok(())
    }

    /// Plays a move in `slot`'s game, keeping the subtree below it (tree reuse).
    fn advance(&mut self, slot: usize, uci: &str) -> PyResult<()> {
        if !self.pending.is_empty() {
            return Err(PyValueError::new_err("apply() the pending batch before advancing"));
        }
        let mv = parse_uci(uci).ok_or_else(|| PyValueError::new_err(format!("bad move {uci:?}")))?;
        let tree = self.trees.get_mut(slot).and_then(|t| t.as_mut()).ok_or_else(|| PyValueError::new_err("empty slot"))?;
        if !tree.game.legal_moves().contains(&mv) {
            return Err(PyValueError::new_err(format!("illegal move {uci}")));
        }
        tree.advance(mv);
        Ok(())
    }

    fn has_slot(&self, slot: usize) -> bool {
        self.trees.get(slot).is_some_and(|t| t.is_some())
    }

    /// Like set_root, but from a FEN (any position, e.g. a GUI's custom setup) plus moves.
    fn set_root_fen(&mut self, slot: usize, fen: &str, moves: Vec<String>) -> PyResult<()> {
        let mut game = Game::from_fen(fen).map_err(PyValueError::new_err)?;
        for m in &moves {
            let mv = parse_uci(m).ok_or_else(|| PyValueError::new_err(format!("bad move {m:?}")))?;
            if !game.legal_moves().contains(&mv) {
                return Err(PyValueError::new_err(format!("illegal move {m}")));
            }
            game.push(mv);
        }
        if slot >= self.trees.len() {
            self.trees.resize_with(slot + 1, || None);
        }
        self.trees[slot] = Some(Tree::new(game, self.params));
        Ok(())
    }

    /// Principal variation from the root: most-visited child at each step.
    fn principal_variation(&self, slot: usize, max_len: usize) -> Vec<String> {
        self.trees[slot].as_ref().map_or(Vec::new(), |t| t.principal_variation(max_len).into_iter().map(|m| m.uci()).collect())
    }

    fn clear_slot(&mut self, slot: usize) {
        if slot < self.trees.len() {
            self.trees[slot] = None;
        }
    }

    /// Collects leaves from `slots` (all active slots if None). Returns a dict of bytes:
    /// boards u8 [L,4,90], scalars f32 [L,10], actions i32 [sum], offsets i32 [L+1], slots i32 [L].
    #[pyo3(signature = (leaves_per_tree, slots=None, max_attempts=None))]
    fn collect<'py>(
        &mut self,
        py: Python<'py>,
        leaves_per_tree: usize,
        slots: Option<Vec<usize>>,
        max_attempts: Option<usize>,
    ) -> PyResult<Bound<'py, PyDict>> {
        if !self.pending.is_empty() {
            return Err(PyValueError::new_err("apply() the previous batch first"));
        }
        let wanted: std::collections::HashSet<usize> = match slots {
            Some(v) => v.into_iter().collect(),
            None => (0..self.trees.len()).collect(),
        };
        let attempts = max_attempts.unwrap_or(leaves_per_tree * 2 + 2);
        // Trees are independent: walk them in parallel without holding the GIL.
        let trees = &mut self.trees;
        let collected: Vec<(usize, Vec<Leaf>)> = py.allow_threads(|| {
            trees
                .par_iter_mut()
                .enumerate()
                .filter(|(s, t)| t.is_some() && wanted.contains(s))
                .map(|(s, t)| (s, t.as_mut().unwrap().collect(leaves_per_tree, attempts)))
                .collect()
        });
        for (s, leaves) in collected {
            for leaf in leaves {
                self.pending.push((s, leaf));
            }
        }
        let n = self.pending.len();
        let mut boards = Vec::with_capacity(n * HISTORY * 90);
        let mut scalars: Vec<u8> = Vec::with_capacity(n * NUM_SCALARS * 4);
        let mut actions: Vec<u8> = Vec::new();
        let mut offsets: Vec<u8> = Vec::with_capacity((n + 1) * 4);
        let mut slot_ids: Vec<u8> = Vec::with_capacity(n * 4);
        let mut total: i32 = 0;
        offsets.extend_from_slice(&total.to_le_bytes());
        for (s, leaf) in &self.pending {
            for b in &leaf.encoded.boards {
                boards.extend_from_slice(b);
            }
            for v in leaf.encoded.scalars {
                scalars.extend_from_slice(&v.to_le_bytes());
            }
            for &a in &leaf.actions {
                actions.extend_from_slice(&(a as i32).to_le_bytes());
            }
            total += leaf.actions.len() as i32;
            offsets.extend_from_slice(&total.to_le_bytes());
            slot_ids.extend_from_slice(&(*s as i32).to_le_bytes());
        }
        let out = PyDict::new(py);
        out.set_item("count", n)?;
        out.set_item("boards", PyBytes::new(py, &boards))?;
        out.set_item("scalars", PyBytes::new(py, &scalars))?;
        out.set_item("actions", PyBytes::new(py, &actions))?;
        out.set_item("offsets", PyBytes::new(py, &offsets))?;
        out.set_item("slots", PyBytes::new(py, &slot_ids))?;
        Ok(out)
    }

    /// Applies evaluations for the last collected batch: legal-action logits flattened in the
    /// same order as `actions` (f32 bytes), and one value per leaf for its side to move (f32 bytes).
    fn apply(&mut self, logits: &[u8], values: &[u8]) -> PyResult<()> {
        let logits: Vec<f32> = logits.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
        let values: Vec<f32> = values.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
        if values.len() != self.pending.len() {
            return Err(PyValueError::new_err(format!("expected {} values, got {}", self.pending.len(), values.len())));
        }
        let mut offset = 0;
        for (i, (s, leaf)) in std::mem::take(&mut self.pending).into_iter().enumerate() {
            let k = leaf.actions.len();
            if offset + k > logits.len() {
                return Err(PyValueError::new_err("not enough logits"));
            }
            if let Some(tree) = self.trees[s].as_mut() {
                tree.apply(&leaf, &logits[offset..offset + k], values[i]);
            }
            offset += k;
        }
        Ok(())
    }

    fn root_visits(&self, slot: usize) -> u32 {
        self.trees[slot].as_ref().map_or(0, |t| t.root_visits())
    }

    fn root_value(&self, slot: usize) -> f32 {
        self.trees[slot].as_ref().map_or(0.0, |t| t.root_value())
    }

    /// [(uci, visits, q, prior)] for the root's children.
    fn root_children(&self, slot: usize) -> Vec<(String, u32, f32, f32)> {
        self.trees[slot]
            .as_ref()
            .map_or(Vec::new(), |t| t.root_children().into_iter().map(|(m, n, q, p)| (m.uci(), n, q, p)).collect())
    }

    /// Network value of the root before search (side to move), if evaluated.
    fn root_net_value(&self, slot: usize) -> Option<f32> {
        self.trees[slot].as_ref().and_then(|t| t.root_net_value())
    }

    /// (max depth, visit-weighted mean depth) of the searched tree.
    fn depth_stats(&self, slot: usize) -> (u32, f32) {
        self.trees[slot].as_ref().map_or((0, 0.0), |t| t.depth_stats())
    }

    /// [(parent row, depth, uci, visits, q, prior, terminal)] - see Tree::dump.
    #[pyo3(signature = (slot, max_depth=4, top_k=4, min_visits=1))]
    fn dump_tree(&self, slot: usize, max_depth: usize, top_k: usize, min_visits: u32) -> Vec<(i32, u32, String, u32, f32, f32, bool)> {
        self.trees[slot].as_ref().map_or(Vec::new(), |t| {
            t.dump(max_depth, top_k, min_visits).into_iter().map(|(p, d, m, n, q, pr, term)| (p, d, m.uci(), n, q, pr, term)).collect()
        })
    }

    fn best_move(&self, slot: usize) -> Option<String> {
        self.trees[slot].as_ref().and_then(|t| t.best_move()).map(|m| m.uci())
    }

    fn add_root_noise(&mut self, slot: usize, noise: Vec<f32>, fraction: f32) {
        if let Some(t) = self.trees[slot].as_mut() {
            t.add_root_noise(&noise, fraction);
        }
    }

    /// Starts Gumbel sequential halving at `slot`'s (expanded) root over the top `m` actions with
    /// `budget` visits; `noise` holds one Gumbel(0,1) sample per root child (root_children order).
    #[pyo3(signature = (slot, m, budget, noise, c_visit=50.0, c_scale=1.0))]
    fn gumbel_start(&mut self, slot: usize, m: usize, budget: u32, noise: Vec<f32>, c_visit: f32, c_scale: f32) -> PyResult<()> {
        let tree = self.trees.get_mut(slot).and_then(|t| t.as_mut()).ok_or_else(|| PyValueError::new_err("empty slot"))?;
        if !tree.root_expanded() {
            return Err(PyValueError::new_err("root not expanded yet"));
        }
        tree.gumbel_start(m, budget, &noise, c_visit, c_scale);
        Ok(())
    }

    fn gumbel_done(&self, slot: usize) -> bool {
        self.trees[slot].as_ref().is_none_or(|t| t.gumbel_done())
    }

    /// (chosen uci, [(uci, improved prob)], root value) after a Gumbel search.
    fn gumbel_result(&self, slot: usize) -> Option<(String, Vec<(String, f32)>, f32)> {
        let t = self.trees[slot].as_ref()?;
        t.gumbel_result().map(|(m, pol, v)| (m.uci(), pol.into_iter().map(|(m, p)| (m.uci(), p)).collect(), v))
    }

    fn root_expanded(&self, slot: usize) -> bool {
        self.trees[slot].as_ref().is_some_and(|t| t.root_expanded())
    }
}
