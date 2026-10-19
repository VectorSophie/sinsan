/** The non-game pages: how Sinsan works, training progress, and the rules it plays by. */

export function renderAbout(container: HTMLElement): void {
  container.innerHTML = `
  <article class="page">
    <h2>How Sinsan works</h2>
    <p class="lead">Sinsan is a Janggi engine that pairs a neural network with tree search, the approach behind
      AlphaZero, Leela Chess Zero and KataGo. The network supplies judgement - which moves look promising and who
      is winning - and the search checks that judgement by reading ahead.</p>

    <div class="flow" aria-label="How a move is chosen">
      <div class="flow-step"><b>Position</b><span>pieces now and 3 moves ago, repetitions, points, pass and bikjang state</span></div>
      <div class="flow-arrow">→</div>
      <div class="flow-step"><b>Network</b><span>60M-parameter transformer: move probabilities, win/loss, expected point margin</span></div>
      <div class="flow-arrow">→</div>
      <div class="flow-step"><b>Search</b><span>hundreds to thousands of visits, each one reading a line deeper</span></div>
      <div class="flow-arrow">→</div>
      <div class="flow-step"><b>Move</b><span>the most-visited candidate</span></div>
    </div>

    <h3>The network</h3>
    <p>Each position goes to a 12-layer transformer that treats the 90 board points as tokens, so every point can
      relate to every other in one step - useful in Janggi, where chariots and cannons act across the whole board.
      It predicts several things at once, and they all help it learn:</p>
    <ul>
      <li><b>Policy</b> - a probability for every legal move: its first instinct. Shown as <i>Prior</i> in the panel.</li>
      <li><b>Win / loss</b> - who is winning: the <i>first instinct</i> marker on the win bar.</li>
      <li><b>Point margin</b> - who would win on points and by how much, since many Janggi games end on points
        (bikjang, passing, the move limit).</li>
      <li>Training-only extras: the opponent's likely reply, how many moves are left, and a value for every candidate move.</li>
    </ul>

    <h3>The search</h3>
    <p>Search grows a tree of possible continuations (Monte Carlo tree search with the PUCT formula). Each
      <i>visit</i> walks down the tree, at every step choosing the move that best balances what it has learned so
      far (its win rate in the search, <i>Q</i>) against the network's prior, then asks the network about the new
      position and passes the result back up. Promising moves get more visits; a move whose deeper lines turn out
      badly loses them. The move played is the most-visited one, and the tree is kept between moves.</p>
    <p>Unlike Stockfish, the search has no fixed depth: it reads deeply where the play is forcing and shallowly
      elsewhere. The panel shows the average depth of visited lines and the deepest line reached.</p>

    <h3>How it was trained</h3>
    <ol>
      <li><b>Exact rules.</b> A new Rust rules engine reproduces Fairy-Stockfish's <code>janggi</code> rules (the ones
        pychess.org uses), checked on 845,000 positions with zero disagreements, including every way a game can end.</li>
      <li><b>Learning from a strong teacher.</b> Fairy-Stockfish with its Janggi NNUE played 284,000 complete games at
        100,000 nodes per move, scoring its eight best moves in every position: 50 million labelled positions.
        The network learned to predict those scores and the games' results.</li>
      <li><b>Self-play.</b> The network then keeps playing itself with search (Gumbel MCTS) and learns from those
        games. A new version replaces the old one only after beating it in a 400-game match.</li>
    </ol>

    <h3>What changed from the original Sinsan</h3>
    <table class="page-table">
      <thead><tr><th>Original</th><th>Now</th><th>Why it matters</th></tr></thead>
      <tbody>
        <tr><td>Move targets spread almost evenly over 8 moves (scores ÷ 800)</td><td>Sharp targets (÷ 30), fitted to the teacher's real score gaps</td><td>The old network couldn't tell good moves from merely legal ones</td></tr>
        <tr><td>Teacher: Fairy-Stockfish classical, depth 5-6</td><td>Fairy-Stockfish NNUE, 100K nodes</td><td>Classical evaluation is worth about 10× fewer nodes</td></tr>
        <tr><td>Only the first 60 plies of 1,800 games, half of them random</td><td>284,000 full games, opening to ending</td><td>Most of the game was never seen</td></tr>
        <tr><td>Endings disagreed with Fairy-Stockfish (bikjang, repetition, move limit)</td><td>Rules engine verified against it</td><td>The network was learning the wrong outcomes</td></tr>
        <tr><td>Tiny 56×7 CNN in the browser</td><td>60M transformer on a GPU, plus search</td><td>Strength first; a small browser model comes later</td></tr>
        <tr><td>No strength measurement</td><td>Rated against a ladder of Fairy-Stockfish settings</td><td>Every change is measured in Elo</td></tr>
      </tbody>
    </table>
    <p>Result: the original V3 rated about −1,420 on the ladder; the current network rates about +2,400,
      roughly 3,800 Elo stronger (see <a href="#progress">Progress</a>).</p>

    <h3>What you can tune</h3>
    <dl class="page-dl">
      <dt>Network</dt><dd>The latest self-play network, the network before self-play, or a smaller early one.</dd>
      <dt>Visits / time</dt><dd>How much it searches per move. More visits read deeper and play stronger; 800 is the tested setting.</dd>
      <dt>Randomness</dt><dd>Plays a searched move in proportion to its visits instead of always the best: weaker and more varied.
        Can be limited to the opening.</dd>
      <dt>Exploration (cpuct)</dt><dd>Higher spreads the search over more moves; lower digs into the favourites. Default 1.745 (Leela Chess Zero's).</dd>
      <dt>Unvisited-move penalty (FPU)</dt><dd>How pessimistic it is about moves it hasn't tried yet. Higher keeps the search narrow.</dd>
      <dt>Policy softness</dt><dd>Flattens the network's move probabilities before searching, so less-favoured moves get a look.</dd>
      <dt>Point greed</dt><dd>Adds the expected point margin to the win estimate, so it prefers winning by more.</dd>
    </dl>

    <h3>Reading the analysis panel</h3>
    <dl class="page-dl">
      <dt>Win estimate</dt><dd>Sinsan's chance of winning after its search. The thin marker is the network's first instinct before searching.</dd>
      <dt>Visits · Win · Prior</dt><dd>How much search each candidate got, how it scored, and how much the network liked it beforehand.
        A move with a low prior but many visits is one the search discovered.</dd>
      <dt>Decision tree</dt><dd>The lines it read, left to right. Node size is visits, colour is Sinsan's win chance; your moves are in italics.
        Hover a node to see the line on the board.</dd>
      <dt>Move ratings</dt><dd>How much win chance your move gave away, by Sinsan's searches before and after it: Best (its expected move,
        or under 0.5%), Excellent (&lt;2%), Good (&lt;5%), Inaccuracy (&lt;10%), Mistake (&lt;20%), Blunder (20%+). Accuracy uses Lichess's formula.</dd>
      <dt>Threat</dt><dd>What Sinsan would play if you passed. Passing is legal in Janggi, so this is the real threat.</dd>
    </dl>

    <h3>Origins and the name</h3>
    <p>Sinsan started from Moka, a project that distilled a strong Go engine into a tiny model that runs in the browser,
      and set out to do the same for Janggi. This rebuild first builds a strong engine; the small browser model is distilled from it later.</p>
    <p><b>神算 (신산, Sinsan)</b> means "calculation that has reached a divine level". It is a nickname of the Go (baduk)
      professional Lee Chang-ho (이창호), known for his endgame precision. It is a Go term, not a Janggi title; the project
      borrows its plain meaning.</p>

    <h3>Honest limits</h3>
    <ul>
      <li>Ratings here are against Fairy-Stockfish, not people. The goal is amateur 1단 (around 1,700-1,800 on pychess.org);
        that needs games against humans, which haven't been played yet.</li>
      <li>The engine runs on a GPU server. The browser-only version is a later phase.</li>
    </ul>
  </article>`;
}

export function renderRules(container: HTMLElement): void {
  container.innerHTML = `
  <article class="page">
    <h2>Rules</h2>
    <p class="lead">Sinsan plays Fairy-Stockfish's <code>janggi</code> rules, the default on pychess.org.
      Games are never drawn: anything that doesn't end in checkmate is decided on points.</p>

    <h3>Pieces</h3>
    <table class="page-table">
      <thead><tr><th>Piece</th><th>Moves</th><th>Points</th></tr></thead>
      <tbody>
        <tr><td>車 Chariot</td><td>Any distance orthogonally; also along the palace diagonals.</td><td>13</td></tr>
        <tr><td>包 Cannon</td><td>Moves and captures by jumping exactly one piece (never a cannon); cannot capture a cannon. Also along palace diagonals.</td><td>7</td></tr>
        <tr><td>馬 Horse</td><td>One step orthogonally, then one diagonally outward; blocked if the first square is occupied.</td><td>5</td></tr>
        <tr><td>象 Elephant</td><td>One step orthogonally, then two diagonally outward; blocked if either square on the way is occupied.</td><td>3</td></tr>
        <tr><td>士 Guard</td><td>One step along the palace lines, staying in the palace.</td><td>3</td></tr>
        <tr><td>卒 兵 Soldier</td><td>One step forward or sideways; forward along the diagonals inside the enemy palace.</td><td>2</td></tr>
        <tr><td>楚 漢 General</td><td>One step along the palace lines, staying in the palace.</td><td>—</td></tr>
      </tbody>
    </table>
    <p>Cho (blue) moves first. Before the game each side chooses where its horses and elephants start (마상/상마 on each wing).</p>

    <h3>Passing and bikjang</h3>
    <ul>
      <li>You may pass instead of moving, except when in check.</li>
      <li><b>Bikjang</b>: the two generals face each other on an open file. The side to move must break it or pass.
        If it is still standing after the next move (the opponent passed rather than breaking it), the game ends on points.</li>
    </ul>

    <h3>How games end</h3>
    <table class="page-table">
      <tbody>
        <tr><td>Checkmate, or no legal moves</td><td>The side to move loses.</td></tr>
        <tr><td>Bikjang left standing</td><td>Decided on points.</td></tr>
        <tr><td>Both sides pass in a row</td><td>Decided on points.</td></tr>
        <tr><td>100 plies without a capture</td><td>Decided on points.</td></tr>
        <tr><td>The same position three times</td><td>If one side gave check on every move of the cycle, that side loses (perpetual check);
          otherwise decided on points.</td></tr>
      </tbody>
    </table>
    <p><b>Points:</b> add up your remaining pieces (values above). Han gets 1.5 extra for moving second,
      so Cho needs to be at least 2 points ahead to win on points.</p>
  </article>`;
}

interface Progress {
  ladder: { id: string; label: string; elo: number; kind: string }[];
  nets: { id: string; label: string; elo: number; visits: number; matches: { opponent: string; games: number; score: number; elo: number }[] }[];
  rl: { iteration: number; promoted: boolean; score: number | null; elo: number | null; low: number | null; high: number | null }[];
  selfplay: string[];
}

const SVG = 'http://www.w3.org/2000/svg';
function svg<K extends keyof SVGElementTagNameMap>(tag: K, attrs: Record<string, string | number>, text?: string): SVGElementTagNameMap[K] {
  const e = document.createElementNS(SVG, tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, String(v));
  if (text !== undefined) e.textContent = text;
  return e;
}

function ladderChart(data: Progress): SVGSVGElement {
  const rows = [
    ...data.ladder.filter((l) => l.kind !== 'fsf' || /nnue/.test(l.id)).map((l) => ({ ...l, sinsan: l.id.startsWith('oldnet') })),
    ...data.nets.map((n) => ({ id: n.id, label: `Sinsan: ${n.label}`, elo: n.elo, kind: 'net', sinsan: true })),
  ].sort((a, b) => b.elo - a.elo);
  const rowH = 22;
  const labelW = 300;
  const w = 760;
  const h = rows.length * rowH + 30;
  const lo = -2400;
  const hi = 2600;
  const x = (elo: number) => labelW + ((elo - lo) / (hi - lo)) * (w - labelW - 50);
  const s = svg('svg', { viewBox: `0 0 ${w} ${h}`, class: 'ladder-svg', role: 'img', 'aria-label': 'Strength ladder' });
  for (const tick of [-2000, -1000, 0, 1000, 2000]) {
    s.append(svg('line', { x1: x(tick), x2: x(tick), y1: 0, y2: h - 22, class: 'ladder-grid' }));
    s.append(svg('text', { x: x(tick), y: h - 6, class: 'ladder-tick', 'text-anchor': 'middle' }, tick > 0 ? `+${tick}` : String(tick)));
  }
  rows.forEach((r, i) => {
    const y = i * rowH + 4;
    s.append(svg('text', { x: labelW - 8, y: y + 14, class: `ladder-label${r.kind === 'net' ? ' ladder-new' : ''}`, 'text-anchor': 'end' }, r.label));
    s.append(svg('rect', {
      x: Math.min(x(0), x(r.elo)), y: y + 4, width: Math.max(2, Math.abs(x(r.elo) - x(0))), height: rowH - 8, rx: 2,
      class: r.kind === 'net' ? 'ladder-bar-new' : r.sinsan ? 'ladder-bar-old' : 'ladder-bar',
    }));
    s.append(svg('text', { x: x(Math.max(r.elo, 0)) + 6, y: y + 14, class: 'ladder-val' }, `${r.elo > 0 ? '+' : ''}${r.elo}`));
  });
  return s;
}

function rlChart(data: Progress): SVGSVGElement {
  const w = 760;
  const h = 200;
  const pad = 36;
  const its = data.rl.filter((r) => r.elo !== null);
  const maxIt = Math.max(5, ...its.map((r) => r.iteration + 1));
  const lo = -250;
  const hi = 120;
  const x = (i: number) => pad + (i / maxIt) * (w - pad - 10);
  const y = (e: number) => 10 + ((hi - Math.max(lo, Math.min(hi, e))) / (hi - lo)) * (h - 40);
  const s = svg('svg', { viewBox: `0 0 ${w} ${h}`, class: 'rl-svg', role: 'img', 'aria-label': 'Self-play rounds' });
  for (const tick of [-200, -100, 0, 100]) {
    s.append(svg('line', { x1: pad, x2: w - 10, y1: y(tick), y2: y(tick), class: tick === 0 ? 'ladder-zero' : 'ladder-grid' }));
    s.append(svg('text', { x: pad - 6, y: y(tick) + 4, class: 'ladder-tick', 'text-anchor': 'end' }, tick > 0 ? `+${tick}` : String(tick)));
  }
  for (const r of its) {
    const cx = x(r.iteration + 0.5);
    s.append(svg('line', { x1: cx, x2: cx, y1: y(r.low!), y2: y(r.high!), class: 'rl-ci' }));
    const c = svg('circle', { cx, cy: y(r.elo!), r: 5, class: r.promoted ? 'rl-pt rl-promoted' : 'rl-pt' });
    c.append(svg('title', {}, `Round ${r.iteration}: ${r.score}% (${r.elo! > 0 ? '+' : ''}${r.elo} Elo, ${r.low} to ${r.high}) - ${r.promoted ? 'promoted' : 'rejected'}`));
    s.append(c);
    s.append(svg('text', { x: cx, y: h - 8, class: 'ladder-tick', 'text-anchor': 'middle' }, String(r.iteration)));
  }
  return s;
}

export async function renderProgress(container: HTMLElement): Promise<void> {
  container.innerHTML = `<article class="page"><h2>Progress</h2><p class="lead">Loading…</p></article>`;
  let data: Progress;
  try {
    const r = await fetch('/api/progress');
    data = (await r.json()) as Progress;
    if (!r.ok) throw new Error(JSON.stringify(data));
  } catch (error) {
    container.innerHTML = `<article class="page"><h2>Progress</h2><p class="lead">Couldn't reach the server (${String(error)}).</p></article>`;
    return;
  }
  const promoted = data.rl.filter((r) => r.promoted).length;
  const best = [...data.nets].sort((a, b) => b.elo - a.elo)[0];
  container.innerHTML = `
  <article class="page">
    <h2>Progress</h2>
    <p class="lead">Live from the training server. Strength is measured in Elo against Fairy-Stockfish with its Janggi NNUE at
      different node budgets; 0 is Fairy-Stockfish at 1,000 nodes per move. +200 Elo means winning about 3 games in 4.</p>
    <div class="stat-row">
      <div class="stat"><b>${best ? `+${best.elo}` : '—'}</b><span>best rating (800 visits)</span></div>
      <div class="stat"><b>${data.rl.length}</b><span>self-play rounds</span></div>
      <div class="stat"><b>${promoted}</b><span>promotions</span></div>
      <div class="stat"><b>50M</b><span>teacher positions</span></div>
    </div>
    <h3>Strength ladder</h3>
    <div class="chart ladder-chart"></div>
    <p class="page-note">Sinsan nets search 800 visits per move; each rating averages its matches (200 games each) against the
      settings it was tested on. Rounds 2 and 9 were each tested in a single match, so treat their ratings as ±50.</p>
    <h3>Self-play rounds</h3>
    <div class="chart rl-chart"></div>
    <p class="page-note">Each round's new network against the current best (400 games, 128 visits): Elo with its 95% interval.
      Filled points were promoted. Most rounds land within a few dozen Elo, which is why each promotion is also
      re-checked against Fairy-Stockfish.</p>
    <h3>Self-play now</h3>
    <pre class="page-pre">${data.selfplay.join('\n') || 'No self-play running.'}</pre>
  </article>`;
  container.querySelector('.ladder-chart')!.append(ladderChart(data));
  container.querySelector('.rl-chart')!.append(rlChart(data));
}
