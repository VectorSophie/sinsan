//! Move <-> action-id encoding, identical to packages/action-space: 90 origins x 60 templates + pass.
//! Verified against packages/action-space/golden-fixture.json in tests/action_fixture.rs.

use crate::board::Move;
use crate::{COLS, ROWS, SQUARES};
use std::sync::OnceLock;

pub const TEMPLATE_COUNT: usize = 60;
pub const PASS_ACTION: usize = SQUARES * TEMPLATE_COUNT;
pub const ACTION_SPACE_SIZE: usize = PASS_ACTION + 1;

/// Template order must match packages/action-space/src/templates.ts exactly.
pub fn templates() -> &'static [(i8, i8); TEMPLATE_COUNT] {
    static T: OnceLock<[(i8, i8); TEMPLATE_COUNT]> = OnceLock::new();
    T.get_or_init(|| {
        let mut out = Vec::with_capacity(TEMPLATE_COUNT);
        for (dr, dc) in [(-1, 0), (1, 0), (0, -1), (0, 1)] {
            for dist in 1..=9 {
                out.push((dr * dist, dc * dist));
            }
        }
        for (dr, dc) in [(-1, -1), (-1, 1), (1, -1), (1, 1)] {
            for dist in 1..=2 {
                out.push((dr * dist, dc * dist));
            }
        }
        out.extend([(-2, -1), (-2, 1), (2, -1), (2, 1), (-1, -2), (1, -2), (-1, 2), (1, 2)]);
        out.extend([(-3, -2), (-3, 2), (3, -2), (3, 2), (-2, -3), (2, -3), (-2, 3), (2, 3)]);
        out.try_into().unwrap()
    })
}

fn template_index(dr: i8, dc: i8) -> Option<usize> {
    templates().iter().position(|&t| t == (dr, dc))
}

/// Action id of a move. Pass maps to PASS_ACTION.
pub fn encode_move(m: Move) -> Option<usize> {
    if m.is_pass() {
        return Some(PASS_ACTION);
    }
    let (f, t) = (m.from as usize, m.to as usize);
    let dr = (t / COLS) as i8 - (f / COLS) as i8;
    let dc = (t % COLS) as i8 - (f % COLS) as i8;
    Some(f * TEMPLATE_COUNT + template_index(dr, dc)?)
}

/// Decodes an action id to (from, to); pass returns None (the caller supplies the general's square).
pub fn decode_action(action: usize) -> Option<(usize, usize)> {
    if action >= PASS_ACTION {
        return None;
    }
    let (from, ti) = (action / TEMPLATE_COUNT, action % TEMPLATE_COUNT);
    let (dr, dc) = templates()[ti];
    let r = (from / COLS) as i8 + dr;
    let c = (from % COLS) as i8 + dc;
    if r < 0 || r >= ROWS as i8 || c < 0 || c >= COLS as i8 {
        return None;
    }
    Some((from, r as usize * COLS + c as usize))
}

/// Maps an action to the same move seen from the other side (board rotated 180 degrees).
/// Used for side-to-move canonicalization: every template's negation is also a template.
pub fn rotate_action(action: usize) -> usize {
    if action == PASS_ACTION {
        return action;
    }
    let (from, ti) = (action / TEMPLATE_COUNT, action % TEMPLATE_COUNT);
    let (dr, dc) = templates()[ti];
    (SQUARES - 1 - from) * TEMPLATE_COUNT + template_index(-dr, -dc).expect("template set is closed under negation")
}

/// Maps an action to its left-right mirror image (column c -> 8 - c).
pub fn mirror_action(action: usize) -> usize {
    if action == PASS_ACTION {
        return action;
    }
    let (from, ti) = (action / TEMPLATE_COUNT, action % TEMPLATE_COUNT);
    let (dr, dc) = templates()[ti];
    let (r, c) = (from / COLS, from % COLS);
    (r * COLS + (COLS - 1 - c)) * TEMPLATE_COUNT + template_index(dr, -dc).expect("template set is closed under mirroring")
}
