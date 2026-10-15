//! Game history and results, mirroring Fairy-Stockfish's `janggi` game-end logic
//! (`Position::is_immediate_game_end`, `is_optional_game_end`, and the no-legal-moves case).

use crate::board::{parse_uci, Board, Formation, Move, Side};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EndReason {
    /// No legal moves while in check.
    Checkmate,
    /// No legal moves while not in check (FSF scores this as a loss too).
    Stalemate,
    /// Bikjang stood in two consecutive positions; decided by points.
    Bikjang,
    /// Two consecutive passes; decided by points.
    DoublePass,
    /// 100 plies without a capture; decided by points.
    MoveLimit,
    /// Threefold repetition without perpetual check; decided by points.
    Repetition,
    /// Threefold repetition where one side gave check on every move of the cycle: that side loses.
    PerpetualCheck,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GameResult {
    pub winner: Side,
    pub reason: EndReason,
}

#[derive(Clone, Debug)]
struct State {
    board: Board,
    in_check: bool,
    bikjang: bool,
    /// The move that led to this state was a pass.
    passed: bool,
}

#[derive(Clone, Debug)]
pub struct Game {
    states: Vec<State>,
    moves: Vec<Move>,
}

/// Points-based result from the perspective of the side to move, as FSF's
/// `material_counting_result` (always decisive under janggi counting).
fn by_points(board: &Board, reason: EndReason) -> GameResult {
    GameResult { winner: board.point_winner(), reason }
}

impl Game {
    pub fn new(board: Board) -> Game {
        let in_check = board.in_check();
        let bikjang = board.bikjang();
        Game { states: vec![State { board, in_check, bikjang, passed: false }], moves: Vec::new() }
    }

    pub fn startpos(cho: Formation, han: Formation) -> Game {
        Game::new(Board::startpos(cho, han))
    }

    pub fn from_fen(fen: &str) -> Result<Game, String> {
        Ok(Game::new(Board::from_fen(fen)?))
    }

    #[inline]
    pub fn board(&self) -> &Board {
        &self.states.last().unwrap().board
    }

    pub fn moves(&self) -> &[Move] {
        &self.moves
    }

    /// Plies played since the root position (FSF's pliesFromNull for a game set up from FEN + moves).
    pub fn plies(&self) -> usize {
        self.moves.len()
    }

    pub fn root_fen(&self) -> String {
        self.states[0].board.fen()
    }

    pub fn legal_moves(&self) -> Vec<Move> {
        self.board().legal_moves()
    }

    /// Plays a move without checking legality.
    pub fn push(&mut self, m: Move) {
        let board = self.board().play(m);
        let in_check = board.in_check();
        let bikjang = board.bikjang();
        self.states.push(State { board, in_check, bikjang, passed: m.is_pass() });
        self.moves.push(m);
    }

    /// Plays a UCI move after checking it is legal.
    pub fn push_uci(&mut self, uci: &str) -> Result<(), String> {
        let m = parse_uci(uci).ok_or_else(|| format!("unparseable move {uci:?}"))?;
        if !self.legal_moves().contains(&m) {
            return Err(format!("illegal move {uci} in {}", self.board().fen()));
        }
        self.push(m);
        Ok(())
    }

    pub fn pop(&mut self) -> Option<Move> {
        if self.moves.is_empty() {
            return None;
        }
        self.states.pop();
        self.moves.pop()
    }

    /// Bikjang standing for two consecutive positions, or two consecutive passes.
    pub fn immediate_end(&self) -> Option<GameResult> {
        let n = self.states.len();
        if n < 2 {
            return None;
        }
        let (cur, prev) = (&self.states[n - 1], &self.states[n - 2]);
        if cur.bikjang && prev.bikjang {
            return Some(by_points(&cur.board, EndReason::Bikjang));
        }
        if cur.passed && prev.passed {
            return Some(by_points(&cur.board, EndReason::DoublePass));
        }
        None
    }

    /// The 100-ply rule and threefold repetition (FSF's is_optional_game_end, called with ply 0).
    pub fn optional_end(&self) -> Option<GameResult> {
        let cur = self.states.last().unwrap();
        let board = &cur.board;
        if board.rule50 > 99 && (!cur.in_check || !board.legal_moves().is_empty()) {
            return Some(by_points(board, EndReason::MoveLimit));
        }

        // Repetition: walk back two plies at a time within the no-capture window.
        let last = self.states.len() - 1;
        let end = (board.rule50 as usize).min(self.plies());
        if end < 4 {
            return None;
        }
        let at = |back: usize| &self.states[last - back];
        let mut perpetual_them = cur.in_check && at(2).in_check;
        let mut perpetual_us = at(1).in_check && at(3).in_check;
        let mut count = 0;
        let mut i = 4;
        while i <= end {
            let stp = at(i);
            perpetual_them &= stp.in_check;
            if stp.board.key == board.key {
                count += 1;
                if count + 1 >= 3 {
                    let us = board.side;
                    return Some(match (perpetual_them, perpetual_us) {
                        (true, false) => GameResult { winner: us, reason: EndReason::PerpetualCheck },
                        (false, true) => GameResult { winner: us.other(), reason: EndReason::PerpetualCheck },
                        _ => by_points(board, EndReason::Repetition),
                    });
                }
            }
            if i + 1 <= end {
                perpetual_us &= at(i + 1).in_check;
            }
            i += 2;
        }
        None
    }

    /// Full referee: no legal moves first, then immediate ends, then optional ends.
    pub fn result(&self) -> Option<GameResult> {
        let cur = self.states.last().unwrap();
        if cur.board.legal_moves().is_empty() {
            let reason = if cur.in_check { EndReason::Checkmate } else { EndReason::Stalemate };
            return Some(GameResult { winner: cur.board.side.other(), reason });
        }
        self.immediate_end().or_else(|| self.optional_end())
    }
}

/// Counts leaf nodes of the legal-move tree (rules-only, ignores game endings).
pub fn perft(board: &Board, depth: u32) -> u64 {
    if depth == 0 {
        return 1;
    }
    let moves = board.legal_moves();
    if depth == 1 {
        return moves.len() as u64;
    }
    moves.iter().map(|&m| perft(&board.play(m), depth - 1)).sum()
}

/// Perft that stops at immediate game ends (double pass, standing bikjang), like Fairy-Stockfish's
/// `go perft`, whose legal-move generation returns nothing once such an end has occurred.
pub fn perft_game(game: &mut Game, depth: u32) -> u64 {
    if depth == 0 {
        return 1;
    }
    if game.immediate_end().is_some() {
        return 0;
    }
    let moves = game.legal_moves();
    let mut total = 0;
    for m in moves {
        game.push(m);
        total += if depth == 1 { 1 } else { perft_game(game, depth - 1) };
        game.pop();
    }
    total
}
