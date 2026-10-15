//! Janggi rules core for Sinsan.
//!
//! The rules follow Fairy-Stockfish's `janggi` variant exactly (the teacher engine, and the rules
//! pychess.org plays), verified by differential tests against pyffish in
//! `tests/differential/`. In particular, game endings follow FSF rather than an independent
//! reading of the rules:
//!
//! - no legal moves: the side to move loses (checkmate, or stalemate, which FSF also scores as a loss)
//! - bikjang that persists for two consecutive positions, or two consecutive passes: decided by points
//! - 100 plies without a capture: decided by points
//! - threefold repetition: a side that gave check on every move of the cycle loses; otherwise points
//!
//! Point count: chariot 13, cannon 7, horse 5, elephant 3, guard 3, soldier 2, and Cho needs a
//! margin of at least 2 (Han's 1.5 compensation), so point decisions are never drawn.
//!
//! Coordinates: square = row * 9 + col, row 0 is Han's back rank (rank 10 in UCI), row 9 is Cho's
//! back rank (rank 1). This matches `packages/rules` and `packages/action-space`.

pub mod action;
pub mod alphabeta;
pub mod board;
pub mod features;
pub mod game;
pub mod search;
mod tables;

#[cfg(feature = "python")]
mod python;
#[cfg(feature = "python")]
mod python_search;

pub use action::{decode_action, encode_move, ACTION_SPACE_SIZE, PASS_ACTION, TEMPLATE_COUNT};
pub use board::{Board, Formation, Move, Piece, PieceKind, Side};
pub use game::{EndReason, Game, GameResult};

pub const ROWS: usize = 10;
pub const COLS: usize = 9;
pub const SQUARES: usize = 90;
