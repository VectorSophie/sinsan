//! Precomputed geometry: palace membership, palace diagonal lines, and per-side soldier steps.

use crate::{COLS, ROWS, SQUARES};
use std::sync::OnceLock;

pub(crate) const ORTH: [(i8, i8); 4] = [(-1, 0), (1, 0), (0, -1), (0, 1)];

#[inline]
pub(crate) fn offset(sq: usize, dr: i8, dc: i8) -> Option<usize> {
    let r = (sq / COLS) as i8 + dr;
    let c = (sq % COLS) as i8 + dc;
    if r < 0 || r >= ROWS as i8 || c < 0 || c >= COLS as i8 {
        None
    } else {
        Some(r as usize * COLS + c as usize)
    }
}

pub(crate) struct Tables {
    /// Palace membership per side: [0] = Cho palace (rows 7-9), [1] = Han palace (rows 0-2).
    pub palace: [[bool; SQUARES]; 2],
    /// Squares one step away along a palace diagonal line (corner <-> center).
    pub diag_steps: [Vec<usize>; SQUARES],
    /// Palace diagonal rays for sliders: from a corner, [center, opposite corner];
    /// from the center, one ray per corner.
    pub diag_rays: [Vec<Vec<usize>>; SQUARES],
    /// Soldier destinations per side (forward, sideways, forward palace diagonals).
    pub soldier_steps: [[Vec<usize>; SQUARES]; 2],
    /// Reverse of soldier_steps: squares from which a soldier of that side reaches this square.
    pub soldier_attackers: [[Vec<usize>; SQUARES]; 2],
}

fn palace_center(side: usize) -> usize {
    if side == 0 {
        8 * COLS + 4
    } else {
        COLS + 4
    }
}

fn build() -> Tables {
    let mut palace = [[false; SQUARES]; 2];
    for sq in 0..SQUARES {
        let (r, c) = (sq / COLS, sq % COLS);
        if (3..=5).contains(&c) {
            if (7..=9).contains(&r) {
                palace[0][sq] = true;
            }
            if r <= 2 {
                palace[1][sq] = true;
            }
        }
    }

    let mut diag_steps: [Vec<usize>; SQUARES] = std::array::from_fn(|_| Vec::new());
    let mut diag_rays: [Vec<Vec<usize>>; SQUARES] = std::array::from_fn(|_| Vec::new());
    for side in 0..2 {
        let center = palace_center(side);
        for &(dr, dc) in &[(-1i8, -1i8), (-1, 1), (1, -1), (1, 1)] {
            let corner = offset(center, dr, dc).unwrap();
            let opposite = offset(center, -dr, -dc).unwrap();
            diag_steps[center].push(corner);
            diag_steps[corner].push(center);
            diag_rays[center].push(vec![corner]);
            diag_rays[corner].push(vec![center, opposite]);
        }
    }

    let mut soldier_steps: [[Vec<usize>; SQUARES]; 2] =
        std::array::from_fn(|_| std::array::from_fn(|_| Vec::new()));
    for side in 0..2 {
        let forward: i8 = if side == 0 { -1 } else { 1 };
        for sq in 0..SQUARES {
            let steps = &mut soldier_steps[side][sq];
            for (dr, dc) in [(forward, 0), (0, -1), (0, 1)] {
                if let Some(t) = offset(sq, dr, dc) {
                    steps.push(t);
                }
            }
            // Palace diagonals, forward only (a soldier can only ever reach the enemy palace).
            for &t in &diag_steps[sq] {
                let dr = (t / COLS) as i8 - (sq / COLS) as i8;
                if dr == forward {
                    steps.push(t);
                }
            }
        }
    }
    let mut soldier_attackers: [[Vec<usize>; SQUARES]; 2] =
        std::array::from_fn(|_| std::array::from_fn(|_| Vec::new()));
    for side in 0..2 {
        for from in 0..SQUARES {
            for &to in &soldier_steps[side][from] {
                soldier_attackers[side][to].push(from);
            }
        }
    }

    Tables { palace, diag_steps, diag_rays, soldier_steps, soldier_attackers }
}

pub(crate) fn tables() -> &'static Tables {
    static T: OnceLock<Tables> = OnceLock::new();
    T.get_or_init(build)
}

/// Zobrist keys: [piece code 1..=14][square], plus side to move.
pub(crate) struct Zobrist {
    pub piece: [[u64; SQUARES]; 15],
    pub side: u64,
}

pub(crate) fn zobrist() -> &'static Zobrist {
    static Z: OnceLock<Zobrist> = OnceLock::new();
    Z.get_or_init(|| {
        // splitmix64: deterministic keys so hashes are stable across runs and processes.
        let mut state: u64 = 0x5133_7a9e_0b5e_d00d;
        let mut next = || {
            state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        };
        let mut piece = [[0u64; SQUARES]; 15];
        for row in piece.iter_mut().skip(1) {
            for k in row.iter_mut() {
                *k = next();
            }
        }
        Zobrist { piece, side: next() }
    })
}
