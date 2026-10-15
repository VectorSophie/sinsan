//! Perft counts cross-checked against Fairy-Stockfish's `go perft` (janggi), which stops at
//! immediate game ends (double pass, standing bikjang).
use sinsan_engine::game::perft_game;
use sinsan_engine::Game;

fn check(fen: &str, depth: u32, expected: u64) {
    let mut game = Game::from_fen(fen).unwrap();
    assert_eq!(perft_game(&mut game, depth), expected, "{fen} depth {depth}");
}

#[test]
fn startpos() {
    check("rnba1abnr/4k4/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/4K4/RNBA1ABNR w - - 0 1", 4, 1_065_277);
}

#[test]
fn startpos_depth5() {
    check("rnba1abnr/4k4/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/4K4/RNBA1ABNR w - - 0 1", 5, 35_243_995);
}

#[test]
fn midgame() {
    check("1nba1abnr/4k4/1c5c1/3pp1p1p/p8/9/1Pr1P2PP/1CNK3CR/9/R1B1AABN1 w - - 0 1", 4, 1_743_189);
}
