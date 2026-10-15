//! Board representation, FEN, move generation and attack detection.

use crate::tables::{offset, tables, zobrist, ORTH};
use crate::{COLS, SQUARES};
use std::fmt;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Side {
    /// Moves first, uppercase in FEN ("w"), back rank is row 9.
    Cho = 0,
    /// Lowercase in FEN ("b"), back rank is row 0.
    Han = 1,
}

impl Side {
    #[inline]
    pub fn other(self) -> Side {
        match self {
            Side::Cho => Side::Han,
            Side::Han => Side::Cho,
        }
    }
    #[inline]
    pub fn idx(self) -> usize {
        self as usize
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PieceKind {
    General = 0,
    Guard = 1,
    Elephant = 2,
    Horse = 3,
    Chariot = 4,
    Cannon = 5,
    Soldier = 6,
}

impl PieceKind {
    pub const ALL: [PieceKind; 7] = [
        PieceKind::General,
        PieceKind::Guard,
        PieceKind::Elephant,
        PieceKind::Horse,
        PieceKind::Chariot,
        PieceKind::Cannon,
        PieceKind::Soldier,
    ];

    /// Janggi point values used by FSF's JANGGI_MATERIAL counting.
    pub fn points(self) -> i32 {
        match self {
            PieceKind::General => 0,
            PieceKind::Guard | PieceKind::Elephant => 3,
            PieceKind::Horse => 5,
            PieceKind::Chariot => 13,
            PieceKind::Cannon => 7,
            PieceKind::Soldier => 2,
        }
    }

    fn fen_char(self) -> char {
        match self {
            PieceKind::General => 'k',
            PieceKind::Guard => 'a',
            PieceKind::Elephant => 'b',
            PieceKind::Horse => 'n',
            PieceKind::Chariot => 'r',
            PieceKind::Cannon => 'c',
            PieceKind::Soldier => 'p',
        }
    }

    fn from_fen_char(c: char) -> Option<PieceKind> {
        Some(match c.to_ascii_lowercase() {
            'k' => PieceKind::General,
            'a' => PieceKind::Guard,
            'b' | 'e' => PieceKind::Elephant,
            'n' | 'h' => PieceKind::Horse,
            'r' => PieceKind::Chariot,
            'c' => PieceKind::Cannon,
            'p' => PieceKind::Soldier,
            _ => return None,
        })
    }
}

/// Packed piece code: 0 = empty, otherwise 1 + side * 7 + kind.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub struct Piece(pub u8);

impl Piece {
    pub const EMPTY: Piece = Piece(0);
    #[inline]
    pub fn new(side: Side, kind: PieceKind) -> Piece {
        Piece(1 + side as u8 * 7 + kind as u8)
    }
    #[inline]
    pub fn is_empty(self) -> bool {
        self.0 == 0
    }
    #[inline]
    pub fn side(self) -> Side {
        debug_assert!(!self.is_empty());
        if self.0 <= 7 {
            Side::Cho
        } else {
            Side::Han
        }
    }
    #[inline]
    pub fn kind(self) -> PieceKind {
        debug_assert!(!self.is_empty());
        PieceKind::ALL[((self.0 - 1) % 7) as usize]
    }
    #[inline]
    pub fn is(self, side: Side, kind: PieceKind) -> bool {
        self.0 == 1 + side as u8 * 7 + kind as u8
    }
    #[inline]
    pub fn kind_is(self, kind: PieceKind) -> bool {
        !self.is_empty() && self.kind() == kind
    }
    #[inline]
    fn is_cannon(self) -> bool {
        self.0 == 1 + PieceKind::Cannon as u8 || self.0 == 8 + PieceKind::Cannon as u8
    }
}

/// A move. A pass is encoded as from == to (the mover's general square), matching
/// Fairy-Stockfish's UCI notation for passing.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Move {
    pub from: u8,
    pub to: u8,
}

impl Move {
    #[inline]
    pub fn is_pass(self) -> bool {
        self.from == self.to
    }

    pub fn uci(self) -> String {
        format!("{}{}", square_name(self.from as usize), square_name(self.to as usize))
    }
}

impl fmt::Display for Move {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.uci())
    }
}

pub fn square_name(sq: usize) -> String {
    let (r, c) = (sq / COLS, sq % COLS);
    format!("{}{}", (b'a' + c as u8) as char, 10 - r)
}

pub fn parse_square(s: &str) -> Option<usize> {
    let mut chars = s.chars();
    let file = chars.next()?;
    if !('a'..='i').contains(&file) {
        return None;
    }
    let rank: usize = chars.as_str().parse().ok()?;
    if !(1..=10).contains(&rank) {
        return None;
    }
    Some((10 - rank) * COLS + (file as usize - 'a' as usize))
}

pub fn parse_uci(s: &str) -> Option<Move> {
    // Ranks are one or two digits, so split at the second file letter.
    let bytes = s.as_bytes();
    let split = (1..bytes.len()).find(|&i| bytes[i].is_ascii_alphabetic())?;
    let from = parse_square(&s[..split])?;
    let to = parse_square(&s[split..])?;
    Some(Move { from: from as u8, to: to as u8 })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Formation {
    MasangSangma,
    SangmaMasang,
    MasangMasang,
    SangmaSangma,
}

impl Formation {
    pub const ALL: [Formation; 4] = [
        Formation::MasangSangma,
        Formation::SangmaMasang,
        Formation::MasangMasang,
        Formation::SangmaSangma,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Formation::MasangSangma => "masang-sangma",
            Formation::SangmaMasang => "sangma-masang",
            Formation::MasangMasang => "masang-masang",
            Formation::SangmaSangma => "sangma-sangma",
        }
    }

    pub fn from_name(name: &str) -> Option<Formation> {
        Formation::ALL.into_iter().find(|f| f.name() == name)
    }

    /// Columns 1, 2, 6, 7 in board order, identical to packages/rules/src/formations.ts.
    fn layout(self) -> [PieceKind; 4] {
        use PieceKind::{Elephant as E, Horse as H};
        match self {
            Formation::MasangSangma => [H, E, E, H],
            Formation::SangmaMasang => [E, H, H, E],
            Formation::MasangMasang => [H, E, H, E],
            Formation::SangmaSangma => [E, H, E, H],
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Board {
    pub squares: [Piece; SQUARES],
    pub side: Side,
    pub generals: [u8; 2],
    /// Plies since the last capture (FEN halfmove clock).
    pub rule50: u16,
    pub fullmove: u16,
    pub key: u64,
}

impl Board {
    pub fn startpos(cho: Formation, han: Formation) -> Board {
        use PieceKind::*;
        let mut squares = [Piece::EMPTY; SQUARES];
        for (side, formation, back, general_row, cannon_row, soldier_row) in
            [(Side::Han, han, 0, 1, 2, 3), (Side::Cho, cho, 9, 8, 7, 6)]
        {
            let l = formation.layout();
            let back_rank = [Chariot, l[0], l[1], Guard, General, Guard, l[2], l[3], Chariot];
            for (col, kind) in back_rank.into_iter().enumerate() {
                if kind != General {
                    squares[back * COLS + col] = Piece::new(side, kind);
                }
            }
            squares[general_row * COLS + 4] = Piece::new(side, General);
            squares[cannon_row * COLS + 1] = Piece::new(side, Cannon);
            squares[cannon_row * COLS + 7] = Piece::new(side, Cannon);
            for col in [0, 2, 4, 6, 8] {
                squares[soldier_row * COLS + col] = Piece::new(side, Soldier);
            }
        }
        Board::from_parts(squares, Side::Cho, 0, 1).expect("valid start position")
    }

    fn from_parts(squares: [Piece; SQUARES], side: Side, rule50: u16, fullmove: u16) -> Result<Board, String> {
        let mut generals = [u8::MAX; 2];
        for (sq, p) in squares.iter().enumerate() {
            if !p.is_empty() && p.kind() == PieceKind::General {
                let s = p.side().idx();
                if generals[s] != u8::MAX {
                    return Err("more than one general per side".into());
                }
                generals[s] = sq as u8;
            }
        }
        if generals.contains(&u8::MAX) {
            return Err("each side needs exactly one general".into());
        }
        let mut b = Board { squares, side, generals, rule50, fullmove, key: 0 };
        b.key = b.compute_key();
        Ok(b)
    }

    fn compute_key(&self) -> u64 {
        let z = zobrist();
        let mut k = if self.side == Side::Han { z.side } else { 0 };
        for (sq, p) in self.squares.iter().enumerate() {
            if !p.is_empty() {
                k ^= z.piece[p.0 as usize][sq];
            }
        }
        k
    }

    pub fn from_fen(fen: &str) -> Result<Board, String> {
        let mut fields = fen.split_whitespace();
        let placement = fields.next().ok_or("empty FEN")?;
        let mut squares = [Piece::EMPTY; SQUARES];
        let rows: Vec<&str> = placement.split('/').collect();
        if rows.len() != 10 {
            return Err(format!("expected 10 ranks, got {}", rows.len()));
        }
        for (r, row) in rows.iter().enumerate() {
            let mut c = 0usize;
            let mut digits = String::new();
            let flush = |digits: &mut String, c: &mut usize| -> Result<(), String> {
                if !digits.is_empty() {
                    *c += digits.parse::<usize>().map_err(|e| e.to_string())?;
                    digits.clear();
                }
                Ok(())
            };
            for ch in row.chars() {
                if ch.is_ascii_digit() {
                    digits.push(ch);
                    continue;
                }
                flush(&mut digits, &mut c)?;
                let kind = PieceKind::from_fen_char(ch).ok_or_else(|| format!("bad piece char {ch:?}"))?;
                let side = if ch.is_ascii_uppercase() { Side::Cho } else { Side::Han };
                if c >= COLS {
                    return Err(format!("rank {} overflows", r));
                }
                squares[r * COLS + c] = Piece::new(side, kind);
                c += 1;
            }
            flush(&mut digits, &mut c)?;
            if c != COLS {
                return Err(format!("rank {} has {} files", r, c));
            }
        }
        let side = match fields.next().unwrap_or("w") {
            "w" => Side::Cho,
            "b" => Side::Han,
            other => return Err(format!("bad side {other:?}")),
        };
        let _castling = fields.next();
        let _ep = fields.next();
        let rule50 = fields.next().map(|s| s.parse().map_err(|_| "bad halfmove")).transpose()?.unwrap_or(0);
        let fullmove = fields.next().map(|s| s.parse().map_err(|_| "bad fullmove")).transpose()?.unwrap_or(1);
        Board::from_parts(squares, side, rule50, fullmove)
    }

    pub fn placement(&self) -> String {
        let mut out = String::new();
        for r in 0..10 {
            if r > 0 {
                out.push('/');
            }
            let mut empty = 0;
            for c in 0..COLS {
                let p = self.squares[r * COLS + c];
                if p.is_empty() {
                    empty += 1;
                } else {
                    if empty > 0 {
                        out.push_str(&empty.to_string());
                        empty = 0;
                    }
                    let ch = p.kind().fen_char();
                    out.push(if p.side() == Side::Cho { ch.to_ascii_uppercase() } else { ch });
                }
            }
            if empty > 0 {
                out.push_str(&empty.to_string());
            }
        }
        out
    }

    pub fn fen(&self) -> String {
        format!(
            "{} {} - - {} {}",
            self.placement(),
            if self.side == Side::Cho { "w" } else { "b" },
            self.rule50,
            self.fullmove
        )
    }

    #[inline]
    pub fn at(&self, sq: usize) -> Piece {
        self.squares[sq]
    }

    /// Both generals on one file with nothing between them.
    pub fn bikjang(&self) -> bool {
        let (a, b) = (self.generals[0] as usize, self.generals[1] as usize);
        if a % COLS != b % COLS {
            return false;
        }
        let (lo, hi) = if a < b { (a, b) } else { (b, a) };
        let mut sq = lo + COLS;
        while sq < hi {
            if !self.squares[sq].is_empty() {
                return false;
            }
            sq += COLS;
        }
        true
    }

    /// Is `sq` attacked by a piece of side `by`? Generals and guards never reach the enemy
    /// palace, so only the five field pieces can give check.
    pub fn attacked_by(&self, sq: usize, by: Side) -> bool {
        let t = tables();
        let enemy = |p: Piece, k: PieceKind| p.is(by, k);

        // Chariots and cannons along files and ranks.
        for &(dr, dc) in &ORTH {
            let mut cur = sq;
            let mut screened = false;
            while let Some(next) = offset(cur, dr, dc) {
                cur = next;
                let p = self.squares[cur];
                if p.is_empty() {
                    continue;
                }
                if !screened {
                    if enemy(p, PieceKind::Chariot) {
                        return true;
                    }
                    if p.is_cannon() {
                        break; // a cannon cannot serve as a screen
                    }
                    screened = true;
                } else {
                    if enemy(p, PieceKind::Cannon) {
                        return true;
                    }
                    break;
                }
            }
        }
        // Chariots and cannons along palace diagonals.
        for ray in &t.diag_rays[sq] {
            let mut screened = false;
            for &cur in ray {
                let p = self.squares[cur];
                if p.is_empty() {
                    continue;
                }
                if !screened {
                    if enemy(p, PieceKind::Chariot) {
                        return true;
                    }
                    if p.is_cannon() {
                        break;
                    }
                    screened = true;
                } else {
                    if enemy(p, PieceKind::Cannon) {
                        return true;
                    }
                    break;
                }
            }
        }
        // Horses: attacker at sq - 2d - p, leg at sq - d - p.
        // Elephants: attacker at sq - 3d - 2p, legs at sq - 2d - 2p and sq - d - p.
        for &(dr, dc) in &ORTH {
            for (pr, pc) in [(dc, dr), (-dc, -dr)] {
                if let Some(leg) = offset(sq, -dr - pr, -dc - pc) {
                    if self.squares[leg].is_empty() {
                        if let Some(a) = offset(sq, -2 * dr - pr, -2 * dc - pc) {
                            if enemy(self.squares[a], PieceKind::Horse) {
                                return true;
                            }
                            if let Some(leg2) = offset(sq, -2 * dr - 2 * pr, -2 * dc - 2 * pc) {
                                if self.squares[leg2].is_empty() {
                                    if let Some(e) = offset(sq, -3 * dr - 2 * pr, -3 * dc - 2 * pc) {
                                        if enemy(self.squares[e], PieceKind::Elephant) {
                                            return true;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        for &a in &t.soldier_attackers[by.idx()][sq] {
            if enemy(self.squares[a], PieceKind::Soldier) {
                return true;
            }
        }
        false
    }

    #[inline]
    pub fn in_check(&self) -> bool {
        self.attacked_by(self.generals[self.side.idx()] as usize, self.side.other())
    }

    /// Pseudo-legal moves for the side to move (excluding pass).
    pub fn pseudo_moves(&self, out: &mut Vec<Move>) {
        let t = tables();
        let us = self.side;
        let push = |from: usize, to: usize, out: &mut Vec<Move>| {
            let p = self.squares[to];
            if p.is_empty() || p.side() != us {
                out.push(Move { from: from as u8, to: to as u8 });
            }
        };
        for from in 0..SQUARES {
            let piece = self.squares[from];
            if piece.is_empty() || piece.side() != us {
                continue;
            }
            match piece.kind() {
                PieceKind::General | PieceKind::Guard => {
                    let palace = &t.palace[us.idx()];
                    for &(dr, dc) in &ORTH {
                        if let Some(to) = offset(from, dr, dc) {
                            if palace[to] {
                                push(from, to, out);
                            }
                        }
                    }
                    for &to in &t.diag_steps[from] {
                        push(from, to, out);
                    }
                }
                PieceKind::Soldier => {
                    for &to in &t.soldier_steps[us.idx()][from] {
                        push(from, to, out);
                    }
                }
                PieceKind::Horse => {
                    for &(dr, dc) in &ORTH {
                        let Some(leg) = offset(from, dr, dc) else { continue };
                        if !self.squares[leg].is_empty() {
                            continue;
                        }
                        for (pr, pc) in [(dc, dr), (-dc, -dr)] {
                            if let Some(to) = offset(from, 2 * dr + pr, 2 * dc + pc) {
                                push(from, to, out);
                            }
                        }
                    }
                }
                PieceKind::Elephant => {
                    for &(dr, dc) in &ORTH {
                        let Some(leg) = offset(from, dr, dc) else { continue };
                        if !self.squares[leg].is_empty() {
                            continue;
                        }
                        for (pr, pc) in [(dc, dr), (-dc, -dr)] {
                            let Some(leg2) = offset(from, 2 * dr + pr, 2 * dc + pc) else { continue };
                            if !self.squares[leg2].is_empty() {
                                continue;
                            }
                            if let Some(to) = offset(from, 3 * dr + 2 * pr, 3 * dc + 2 * pc) {
                                push(from, to, out);
                            }
                        }
                    }
                }
                PieceKind::Chariot => {
                    for &(dr, dc) in &ORTH {
                        let mut cur = from;
                        while let Some(to) = offset(cur, dr, dc) {
                            cur = to;
                            push(from, to, out);
                            if !self.squares[to].is_empty() {
                                break;
                            }
                        }
                    }
                    for ray in &t.diag_rays[from] {
                        for &to in ray {
                            push(from, to, out);
                            if !self.squares[to].is_empty() {
                                break;
                            }
                        }
                    }
                }
                PieceKind::Cannon => {
                    let ray_moves = |squares: &mut dyn Iterator<Item = usize>, out: &mut Vec<Move>| {
                        let mut screened = false;
                        for to in squares {
                            let p = self.squares[to];
                            if !screened {
                                if p.is_empty() {
                                    continue;
                                }
                                if p.is_cannon() {
                                    return;
                                }
                                screened = true;
                            } else if p.is_empty() {
                                out.push(Move { from: from as u8, to: to as u8 });
                            } else {
                                if !p.is_cannon() && p.side() != us {
                                    out.push(Move { from: from as u8, to: to as u8 });
                                }
                                return;
                            }
                        }
                    };
                    for &(dr, dc) in &ORTH {
                        let mut it = std::iter::successors(offset(from, dr, dc), |&s| offset(s, dr, dc));
                        ray_moves(&mut it, out);
                    }
                    for ray in &t.diag_rays[from] {
                        ray_moves(&mut ray.iter().copied(), out);
                    }
                }
            }
        }
    }

    /// Plays a move (no legality check) and returns the new board.
    pub fn play(&self, m: Move) -> Board {
        let z = zobrist();
        let mut b = self.clone();
        b.rule50 += 1;
        if !m.is_pass() {
            let (from, to) = (m.from as usize, m.to as usize);
            let moving = b.squares[from];
            let captured = b.squares[to];
            if !captured.is_empty() {
                b.key ^= z.piece[captured.0 as usize][to];
                b.rule50 = 0;
            }
            b.key ^= z.piece[moving.0 as usize][from] ^ z.piece[moving.0 as usize][to];
            b.squares[to] = moving;
            b.squares[from] = Piece::EMPTY;
            if moving.kind() == PieceKind::General {
                b.generals[moving.side().idx()] = to as u8;
            }
        }
        if b.side == Side::Han {
            b.fullmove += 1;
        }
        b.side = b.side.other();
        b.key ^= z.side;
        b
    }

    /// Legal moves, including pass. Follows FSF's `Position::legal`:
    /// - no move may leave the mover's general attacked;
    /// - pass is legal unless in check, but always legal while bikjang stands;
    /// - while bikjang stands, every non-pass move must break it.
    pub fn legal_moves(&self) -> Vec<Move> {
        let mut pseudo = Vec::with_capacity(64);
        self.pseudo_moves(&mut pseudo);
        let us = self.side;
        let bikjang = self.bikjang();
        let mut legal = Vec::with_capacity(pseudo.len() + 1);
        for m in pseudo {
            let next = self.play(m);
            if next.attacked_by(next.generals[us.idx()] as usize, us.other()) {
                continue;
            }
            // Capturing the general only arises after a pass in check during bikjang, a position
            // that is already over (bikjang stood twice); there the bikjang test is meaningless.
            let captures_general = self.squares[m.to as usize].kind_is(PieceKind::General);
            if bikjang && !captures_general && next.bikjang() {
                continue;
            }
            legal.push(m);
        }
        let g = self.generals[us.idx()];
        if bikjang || !self.in_check() {
            legal.push(Move { from: g, to: g });
        }
        legal
    }

    /// Cho points minus Han points (no compensation).
    pub fn point_balance(&self) -> i32 {
        self.squares
            .iter()
            .filter(|p| !p.is_empty())
            .map(|p| if p.side() == Side::Cho { p.kind().points() } else { -p.kind().points() })
            .sum()
    }

    /// Winner under FSF's JANGGI_MATERIAL counting: Cho wins only with a margin of 2+.
    pub fn point_winner(&self) -> Side {
        if self.point_balance() - 1 > 0 {
            Side::Cho
        } else {
            Side::Han
        }
    }
}
