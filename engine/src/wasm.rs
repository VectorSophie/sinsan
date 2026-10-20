//! WebAssembly bindings (`--features wasm`): one search tree driven from JavaScript, the browser
//! counterpart of `sinsan_engine.SearchBatch`. The caller evaluates each collected batch with the
//! network (onnxruntime-web) and hands back legal-action logits and values, exactly as the Python
//! evaluator does, so the browser runs the same rules, encoding and search as the GPU server.

use crate::board::{parse_uci, Formation};
use crate::game::Game;
use crate::search::{Leaf, SearchParams, Tree};
use wasm_bindgen::prelude::*;

fn bad(msg: String) -> JsValue {
    JsValue::from_str(&msg)
}

fn num(x: f32) -> String {
    if x.is_finite() {
        format!("{x}")
    } else {
        "null".into()
    }
}

/// Game with formations and moves applied, validating each move.
fn game_from(fen: &str, moves: &[String]) -> Result<Game, JsValue> {
    let mut game = Game::from_fen(fen).map_err(bad)?;
    for m in moves {
        let mv = parse_uci(m).ok_or_else(|| bad(format!("bad move {m:?}")))?;
        if !game.legal_moves().contains(&mv) {
            return Err(bad(format!("illegal move {m}")));
        }
        game.push(mv);
    }
    Ok(game)
}

/// FEN of the starting position for the given formations (names as in the web app).
#[wasm_bindgen(js_name = startFen)]
pub fn start_fen(cho: &str, han: &str) -> Result<String, JsValue> {
    let f = |n: &str| Formation::from_name(n).ok_or_else(|| bad(format!("unknown formation {n:?}")));
    Ok(Game::startpos(f(cho)?, f(han)?).root_fen())
}

#[wasm_bindgen]
pub struct WasmSearch {
    params: SearchParams,
    tree: Option<Tree>,
    root: Option<(String, Vec<String>)>,
    pending: Vec<Leaf>,
}

#[wasm_bindgen]
impl WasmSearch {
    #[wasm_bindgen(constructor)]
    pub fn new(cpuct: f32, fpu: f32, policy_temp: f32) -> WasmSearch {
        let params = SearchParams {
            cpuct,
            cpuct_base: 38739.0,
            cpuct_factor: 3.894,
            fpu_reduction: fpu,
            fpu_reduction_root: fpu,
            policy_temp,
        };
        WasmSearch { params, tree: None, root: None, pending: Vec::new() }
    }

    /// Sets the position, keeping the tree when it extends the previous one by up to two moves.
    #[wasm_bindgen(js_name = setPosition)]
    pub fn set_position(&mut self, fen: &str, moves: Vec<String>) -> Result<(), JsValue> {
        self.pending.clear();
        if let (Some((rf, rm)), Some(tree)) = (&self.root, self.tree.as_mut()) {
            if rf == fen && moves.len() >= rm.len() && moves.len() - rm.len() <= 2 && moves[..rm.len()] == rm[..] {
                for m in &moves[rm.len()..] {
                    let mv = parse_uci(m).ok_or_else(|| bad(format!("bad move {m:?}")))?;
                    if !tree.game.legal_moves().contains(&mv) {
                        return Err(bad(format!("illegal move {m}")));
                    }
                    tree.advance(mv);
                }
                self.root = Some((fen.to_string(), moves));
                return Ok(());
            }
        }
        self.tree = Some(Tree::new(game_from(fen, &moves)?, self.params));
        self.root = Some((fen.to_string(), moves));
        Ok(())
    }

    /// Drops the tree (e.g. after the evaluation settings changed).
    pub fn reset(&mut self) {
        self.tree = None;
        self.root = None;
        self.pending.clear();
    }

    /// Collects up to `leaves` leaves to evaluate; returns how many. Read them with boards(),
    /// scalars(), actions() and offsets(), then call apply().
    pub fn collect(&mut self, leaves: usize) -> usize {
        self.pending = match self.tree.as_mut() {
            Some(t) => t.collect(leaves, leaves * 2 + 2),
            None => Vec::new(),
        };
        self.pending.len()
    }

    /// u8 [n, 4, 90] piece codes.
    pub fn boards(&self) -> Vec<u8> {
        self.pending.iter().flat_map(|l| l.encoded.boards.iter().flatten().copied()).collect()
    }

    /// f32 [n, 10].
    pub fn scalars(&self) -> Vec<f32> {
        self.pending.iter().flat_map(|l| l.encoded.scalars).collect()
    }

    /// Canonical legal action ids of all leaves, concatenated.
    pub fn actions(&self) -> Vec<u32> {
        self.pending.iter().flat_map(|l| l.actions.iter().copied()).collect()
    }

    /// [n + 1] start of each leaf's actions.
    pub fn offsets(&self) -> Vec<u32> {
        let mut out = vec![0u32];
        for l in &self.pending {
            out.push(out.last().unwrap() + l.actions.len() as u32);
        }
        out
    }

    /// Legal-action logits (in actions() order) and one side-to-move value per leaf.
    pub fn apply(&mut self, logits: &[f32], values: &[f32]) -> Result<(), JsValue> {
        let tree = self.tree.as_mut().ok_or_else(|| bad("no position".into()))?;
        let pending = std::mem::take(&mut self.pending);
        if values.len() != pending.len() {
            return Err(bad(format!("expected {} values, got {}", pending.len(), values.len())));
        }
        let mut offset = 0;
        for (leaf, &v) in pending.iter().zip(values) {
            let k = leaf.actions.len();
            if offset + k > logits.len() {
                return Err(bad("not enough logits".into()));
            }
            tree.apply(leaf, &logits[offset..offset + k], v);
            offset += k;
        }
        Ok(())
    }

    #[wasm_bindgen(js_name = rootExpanded)]
    pub fn root_expanded(&self) -> bool {
        self.tree.as_ref().is_some_and(|t| t.root_expanded())
    }

    #[wasm_bindgen(js_name = rootVisits)]
    pub fn root_visits(&self) -> u32 {
        self.tree.as_ref().map_or(0, |t| t.root_visits())
    }

    #[wasm_bindgen(js_name = rootValue)]
    pub fn root_value(&self) -> f32 {
        self.tree.as_ref().map_or(0.0, |t| t.root_value())
    }

    /// Network value of the root before search, or NaN if not evaluated.
    #[wasm_bindgen(js_name = rootNetValue)]
    pub fn root_net_value(&self) -> f32 {
        self.tree.as_ref().and_then(|t| t.root_net_value()).unwrap_or(f32::NAN)
    }

    #[wasm_bindgen(js_name = bestMove)]
    pub fn best_move(&self) -> Option<String> {
        self.tree.as_ref().and_then(|t| t.best_move()).map(|m| m.uci())
    }

    /// JSON [[uci, visits, q|null, prior], ...] for the root's children.
    #[wasm_bindgen(js_name = rootChildren)]
    pub fn root_children(&self) -> String {
        let rows: Vec<String> = self.tree.as_ref().map_or(Vec::new(), |t| {
            t.root_children().into_iter().map(|(m, n, q, p)| format!("[\"{}\",{n},{},{}]", m.uci(), num(q), num(p))).collect()
        });
        format!("[{}]", rows.join(","))
    }

    #[wasm_bindgen(js_name = principalVariation)]
    pub fn principal_variation(&self, max_len: usize) -> Vec<String> {
        self.tree.as_ref().map_or(Vec::new(), |t| t.principal_variation(max_len).into_iter().map(|m| m.uci()).collect())
    }

    /// [max depth, visit-weighted mean depth].
    #[wasm_bindgen(js_name = depthStats)]
    pub fn depth_stats(&self) -> Vec<f32> {
        let (mx, mean) = self.tree.as_ref().map_or((0, 0.0), |t| t.depth_stats());
        vec![mx as f32, mean]
    }

    /// JSON [[parent row, depth, uci, visits, q|null, prior, terminal], ...] - see Tree::dump.
    #[wasm_bindgen(js_name = dumpTree)]
    pub fn dump_tree(&self, max_depth: usize, top_k: usize, min_visits: u32) -> String {
        let rows: Vec<String> = self.tree.as_ref().map_or(Vec::new(), |t| {
            t.dump(max_depth, top_k, min_visits)
                .into_iter()
                .map(|(p, d, m, n, q, pr, term)| format!("[{p},{d},\"{}\",{n},{},{},{term}]", m.uci(), num(q), num(pr)))
                .collect()
        });
        format!("[{}]", rows.join(","))
    }

    /// Legal moves in the current root position.
    #[wasm_bindgen(js_name = legalMoves)]
    pub fn legal_moves(&self) -> Vec<String> {
        self.tree.as_ref().map_or(Vec::new(), |t| t.game.legal_moves().into_iter().map(|m| m.uci()).collect())
    }
}
