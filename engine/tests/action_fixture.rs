//! Cross-checks the Rust action encoding against the TypeScript golden fixture, and the
//! rotation/mirror maps against geometry.
use sinsan_engine::action::{decode_action, encode_move, mirror_action, rotate_action, PASS_ACTION};
use sinsan_engine::Move;

#[test]
fn matches_typescript_golden_fixture() {
    let text = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/../packages/action-space/golden-fixture.json")).unwrap();
    // Minimal field extraction: the fixture is a flat array of objects with from/to/isPass/actionId.
    let mut checked = 0;
    for obj in text.split('{').skip(1) {
        let field = |name: &str| -> String {
            let start = obj.find(&format!("\"{name}\":")).unwrap() + name.len() + 3;
            obj[start..].split(|c| c == ',' || c == '}' || c == '\n').next().unwrap().trim().to_string()
        };
        let is_pass = field("isPass") == "true";
        let expected: usize = field("actionId").parse().unwrap();
        // Pass entries carry from = to = -1; any from == to move encodes as pass.
        let (from, to): (u8, u8) =
            if is_pass { (0, 0) } else { (field("from").parse().unwrap(), field("to").parse().unwrap()) };
        let m = Move { from, to };
        assert_eq!(encode_move(m), Some(expected), "move {from}->{to} pass={is_pass}");
        if !is_pass {
            assert_eq!(decode_action(expected), Some((from as usize, to as usize)));
        }
        checked += 1;
    }
    assert!(checked > 100, "fixture too small: {checked}");
}

#[test]
fn rotation_and_mirror_are_geometric_involutions() {
    for a in 0..PASS_ACTION {
        assert_eq!(rotate_action(rotate_action(a)), a);
        assert_eq!(mirror_action(mirror_action(a)), a);
        if let Some((f, t)) = decode_action(a) {
            assert_eq!(decode_action(rotate_action(a)), Some((89 - f, 89 - t)));
            let m = |s: usize| s / 9 * 9 + (8 - s % 9);
            assert_eq!(decode_action(mirror_action(a)), Some((m(f), m(t))));
        }
    }
}
