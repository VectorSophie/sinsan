//! Python bindings (`import sinsan_engine`), built with maturin (`--features python`).

use crate::action::{encode_move, PASS_ACTION};
use crate::board::{parse_uci, Board, Formation, Move, Side};
use crate::game::{perft as perft_impl, perft_game, EndReason, Game, GameResult};
use crate::features::{canonical_action, encode, HISTORY, LEGAL_BYTES, NUM_SCALARS, SCALAR_NAMES};
use pyo3::exceptions::PyValueError;
use pyo3::types::{PyBytes, PyDict};
use pyo3::prelude::*;

fn side_str(s: Side) -> &'static str {
    match s {
        Side::Cho => "w",
        Side::Han => "b",
    }
}

fn reason_str(r: EndReason) -> &'static str {
    match r {
        EndReason::Checkmate => "checkmate",
        EndReason::Stalemate => "stalemate",
        EndReason::Bikjang => "bikjang",
        EndReason::DoublePass => "double-pass",
        EndReason::MoveLimit => "move-limit",
        EndReason::Repetition => "repetition",
        EndReason::PerpetualCheck => "perpetual-check",
    }
}

fn result_tuple(r: Option<GameResult>) -> Option<(&'static str, &'static str)> {
    r.map(|r| (side_str(r.winner), reason_str(r.reason)))
}

fn formation(name: &str) -> PyResult<Formation> {
    Formation::from_name(name).ok_or_else(|| PyValueError::new_err(format!("unknown formation {name:?}")))
}

#[pyclass(name = "Game", module = "sinsan_engine")]
#[derive(Clone)]
struct PyGame {
    inner: Game,
}

#[pymethods]
impl PyGame {
    /// Game(fen=None, cho="masang-sangma", han="masang-sangma")
    #[new]
    #[pyo3(signature = (fen=None, cho="masang-sangma", han="masang-sangma"))]
    fn new(fen: Option<&str>, cho: &str, han: &str) -> PyResult<Self> {
        let inner = match fen {
            Some(f) => Game::from_fen(f).map_err(PyValueError::new_err)?,
            None => Game::startpos(formation(cho)?, formation(han)?),
        };
        Ok(PyGame { inner })
    }

    fn copy(&self) -> Self {
        self.clone()
    }

    fn fen(&self) -> String {
        self.inner.board().fen()
    }

    fn root_fen(&self) -> String {
        self.inner.root_fen()
    }

    #[getter]
    fn side_to_move(&self) -> &'static str {
        side_str(self.inner.board().side)
    }

    #[getter]
    fn plies(&self) -> usize {
        self.inner.plies()
    }

    #[getter]
    fn key(&self) -> u64 {
        self.inner.board().key
    }

    fn moves(&self) -> Vec<String> {
        self.inner.moves().iter().map(|m| m.uci()).collect()
    }

    fn legal_moves(&self) -> Vec<String> {
        self.inner.legal_moves().iter().map(|m| m.uci()).collect()
    }

    fn legal_actions(&self) -> Vec<usize> {
        self.inner.legal_moves().iter().map(|&m| encode_move(m).unwrap()).collect()
    }

    fn push(&mut self, uci: &str) -> PyResult<()> {
        self.inner.push_uci(uci).map_err(PyValueError::new_err)
    }

    fn push_action(&mut self, action: usize) -> PyResult<String> {
        let legal = self.inner.legal_moves();
        let m = legal
            .into_iter()
            .find(|&m| encode_move(m) == Some(action))
            .ok_or_else(|| PyValueError::new_err(format!("action {action} is not legal here")))?;
        self.inner.push(m);
        Ok(m.uci())
    }

    fn pop(&mut self) -> Option<String> {
        self.inner.pop().map(|m| m.uci())
    }

    fn in_check(&self) -> bool {
        self.inner.board().in_check()
    }

    fn bikjang(&self) -> bool {
        self.inner.board().bikjang()
    }

    fn point_balance(&self) -> i32 {
        self.inner.board().point_balance()
    }

    /// None while the game continues, else (winner "w"/"b", reason).
    fn result(&self) -> Option<(&'static str, &'static str)> {
        result_tuple(self.inner.result())
    }

    fn immediate_end(&self) -> Option<(&'static str, &'static str)> {
        result_tuple(self.inner.immediate_end())
    }

    fn optional_end(&self) -> Option<(&'static str, &'static str)> {
        result_tuple(self.inner.optional_end())
    }
}

/// Encodes every position of a game for training.
///
/// encode_game(cho, han, moves, candidates) -> dict of bytes buffers (wrap with numpy.frombuffer):
///   boards  uint8   [P, HISTORY, 90]  canonical piece codes (see engine/src/features.rs)
///   scalars float32 [P, NUM_SCALARS]
///   legal   uint8   [P, LEGAL_BYTES]  packed canonical legal-action mask
///   stm     uint8   [P]               side to move (0 Cho, 1 Han)
///   cand_actions int16 [P, K]         canonical action of each candidate (-1 = none)
/// P = len(candidates) (positions 0..P-1, one candidate list per position); K = max candidates.
#[pyfunction]
fn encode_game<'py>(
    py: Python<'py>,
    cho: &str,
    han: &str,
    moves: Vec<String>,
    candidates: Vec<Vec<String>>,
) -> PyResult<Bound<'py, PyDict>> {
    let mut game = Game::startpos(formation(cho)?, formation(han)?);
    for m in &moves {
        let mv = parse_uci(m).ok_or_else(|| PyValueError::new_err(format!("bad move {m:?}")))?;
        game.push(mv);
    }
    let positions = candidates.len();
    if positions > moves.len() + 1 {
        return Err(PyValueError::new_err("more label rows than positions"));
    }
    let k = candidates.iter().map(|c| c.len()).max().unwrap_or(0);
    let mut boards = Vec::with_capacity(positions * HISTORY * 90);
    let mut scalars: Vec<u8> = Vec::with_capacity(positions * NUM_SCALARS * 4);
    let mut legal = Vec::with_capacity(positions * LEGAL_BYTES);
    let mut stm = Vec::with_capacity(positions);
    let mut cand_actions: Vec<u8> = Vec::with_capacity(positions * k * 2);
    for (ply, cands) in candidates.iter().enumerate() {
        let e = encode(&game, ply);
        for b in &e.boards {
            boards.extend_from_slice(b);
        }
        for v in e.scalars {
            scalars.extend_from_slice(&v.to_le_bytes());
        }
        legal.extend_from_slice(&e.legal);
        let us = game.states[ply].board.side;
        stm.push(us as u8);
        for i in 0..k {
            let a: i16 = match cands.get(i) {
                Some(uci) => {
                    let mv = parse_uci(uci).ok_or_else(|| PyValueError::new_err(format!("bad candidate {uci:?}")))?;
                    let action = encode_move(mv).ok_or_else(|| PyValueError::new_err(format!("no template for {uci}")))?;
                    canonical_action(action, us) as i16
                }
                None => -1,
            };
            cand_actions.extend_from_slice(&a.to_le_bytes());
        }
    }
    let out = PyDict::new(py);
    out.set_item("boards", PyBytes::new(py, &boards))?;
    out.set_item("scalars", PyBytes::new(py, &scalars))?;
    out.set_item("legal", PyBytes::new(py, &legal))?;
    out.set_item("stm", PyBytes::new(py, &stm))?;
    out.set_item("cand_actions", PyBytes::new(py, &cand_actions))?;
    out.set_item("k", k)?;
    Ok(out)
}

/// perft(fen, depth, stop_at_game_end=False): leaf count of the legal-move tree.
#[pyfunction]
#[pyo3(signature = (fen, depth, stop_at_game_end=false))]
fn perft(fen: &str, depth: u32, stop_at_game_end: bool) -> PyResult<u64> {
    if stop_at_game_end {
        let mut game = Game::from_fen(fen).map_err(PyValueError::new_err)?;
        return Ok(perft_game(&mut game, depth));
    }
    let board = Board::from_fen(fen).map_err(PyValueError::new_err)?;
    Ok(perft_impl(&board, depth))
}

/// Action id for a UCI move string; pass (from == to) maps to the pass action.
#[pyfunction]
fn encode_uci(uci: &str) -> PyResult<usize> {
    let m: Move = parse_uci(uci).ok_or_else(|| PyValueError::new_err(format!("bad move {uci:?}")))?;
    encode_move(m).ok_or_else(|| PyValueError::new_err(format!("move {uci} has no template")))
}

/// Permutation mapping each action id to its left-right mirror image (for augmentation).
#[pyfunction]
fn mirror_table() -> Vec<usize> {
    (0..crate::action::ACTION_SPACE_SIZE).map(crate::action::mirror_action).collect()
}

#[pymodule]
fn sinsan_engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyGame>()?;
    m.add_class::<crate::python_search::PySearchBatch>()?;
    m.add_function(wrap_pyfunction!(perft, m)?)?;
    m.add_function(wrap_pyfunction!(encode_uci, m)?)?;
    m.add_function(wrap_pyfunction!(encode_game, m)?)?;
    m.add_function(wrap_pyfunction!(mirror_table, m)?)?;
    m.add("PASS_ACTION", PASS_ACTION)?;
    m.add("HISTORY", HISTORY)?;
    m.add("NUM_SCALARS", NUM_SCALARS)?;
    m.add("LEGAL_BYTES", LEGAL_BYTES)?;
    m.add("SCALAR_NAMES", SCALAR_NAMES.to_vec())?;
    m.add("ACTION_SPACE_SIZE", crate::action::ACTION_SPACE_SIZE)?;
    Ok(())
}
