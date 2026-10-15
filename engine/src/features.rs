//! Network input encoding, always from the side to move's perspective.
//!
//! When Han is to move the board is rotated 180 degrees (square s -> 89 - s) and colors swapped,
//! so "us" always sits at the bottom with our palace on rows 7-9 and moves toward row 0. Actions
//! are rotated the same way (`action::rotate_action`). The encoding is compact; one-hot planes
//! are built on the GPU from the piece codes.
//!
//! Per position:
//! - `boards`: HISTORY x 90 canonical piece codes (0 empty, 1-7 ours, 8-14 theirs, by PieceKind
//!   order), current position first, then earlier ones (zeros before the game start), all in the
//!   current side to move's frame.
//! - `scalars`: NUM_SCALARS floats, see `SCALAR_NAMES`.
//! - `legal`: the canonical legal-action mask, bit-packed little-endian into LEGAL_BYTES bytes.

use crate::action::{encode_move, rotate_action, ACTION_SPACE_SIZE};
use crate::board::{Board, Side};
use crate::game::Game;
use crate::SQUARES;

pub const HISTORY: usize = 4;
pub const NUM_SCALARS: usize = 10;
pub const LEGAL_BYTES: usize = ACTION_SPACE_SIZE.div_ceil(8);
pub const SCALAR_NAMES: [&str; NUM_SCALARS] = [
    "repetitions",     // earlier occurrences of this position in the no-capture window, /2, capped at 1
    "in_check",        // side to move is in check
    "bikjang",         // generals face each other now
    "bikjang_before",  // ...and in the previous position (a second one ends the game)
    "last_move_pass",  // opponent just passed (passing now ends the game)
    "us_cho",          // side to move is Cho
    "rule50",          // plies since the last capture / 100 (game ends at 1.0)
    "point_margin",    // our points minus theirs, with Han's 1.5 compensation, / 20
    "ply",             // plies since the start / 200
    "history_plies",   // earlier positions available / (HISTORY - 1)
];

/// Piece code of `board`'s square `sq` seen by side `us` (no rotation applied here).
#[inline]
fn relative_code(board: &Board, sq: usize, us: Side) -> u8 {
    let p = board.squares[sq];
    if p.is_empty() {
        return 0;
    }
    1 + p.kind() as u8 + if p.side() == us { 0 } else { 7 }
}

#[inline]
pub fn canonical_square(sq: usize, us: Side) -> usize {
    if us == Side::Han {
        SQUARES - 1 - sq
    } else {
        sq
    }
}

#[inline]
pub fn canonical_action(action: usize, us: Side) -> usize {
    if us == Side::Han {
        rotate_action(action)
    } else {
        action
    }
}

pub struct Encoded {
    pub boards: [[u8; SQUARES]; HISTORY],
    pub scalars: [f32; NUM_SCALARS],
    pub legal: [u8; LEGAL_BYTES],
}

/// Encodes the position after `ply` moves of `game` (ply <= game.plies()).
pub fn encode(game: &Game, ply: usize) -> Encoded {
    let states = &game.states;
    let cur = &states[ply];
    let board = &cur.board;
    let us = board.side;

    let mut boards = [[0u8; SQUARES]; HISTORY];
    for (h, out) in boards.iter_mut().enumerate() {
        if h > ply {
            break;
        }
        let b = &states[ply - h].board;
        for sq in 0..SQUARES {
            out[canonical_square(sq, us)] = relative_code(b, sq, us);
        }
    }

    let window = (board.rule50 as usize).min(ply);
    let mut repetitions = 0;
    let mut back = 4;
    while back <= window {
        if states[ply - back].board.key == board.key {
            repetitions += 1;
        }
        back += 2;
    }
    let margin = board.point_balance() as f32 - 1.5;
    let scalars = [
        (repetitions as f32 / 2.0).min(1.0),
        cur.in_check as u8 as f32,
        cur.bikjang as u8 as f32,
        (ply > 0 && states[ply - 1].bikjang) as u8 as f32,
        cur.passed as u8 as f32,
        (us == Side::Cho) as u8 as f32,
        board.rule50 as f32 / 100.0,
        if us == Side::Cho { margin } else { -margin } / 20.0,
        ply as f32 / 200.0,
        ply.min(HISTORY - 1) as f32 / (HISTORY - 1) as f32,
    ];

    let mut legal = [0u8; LEGAL_BYTES];
    for m in board.legal_moves() {
        let a = canonical_action(encode_move(m).expect("legal move has a template"), us);
        legal[a / 8] |= 1 << (a % 8);
    }
    Encoded { boards, scalars, legal }
}
