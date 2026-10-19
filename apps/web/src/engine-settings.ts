/** Sinsan's tunable search settings for the GPU-server opponent, persisted per browser. */

export interface EngineSettings {
  net: string;
  limit: 'visits' | 'time';
  visits: number;
  timeMs: number;
  /** 0 = always play the most-visited move; higher samples weaker/more varied moves. */
  temperature: number;
  /** Apply the temperature only during the first this-many plies (0 = the whole game). */
  tempPlies: number;
  cpuct: number;
  fpu: number;
  ptemp: number;
  util: number;
}

const DEFAULTS: EngineSettings = {
  net: 'best', limit: 'visits', visits: 800, timeMs: 3000, temperature: 0, tempPlies: 0,
  cpuct: 1.745, fpu: 0.33, ptemp: 1.36, util: 0,
};

/** Rough strength levels. Ratings are on the Fairy-Stockfish ladder (0 = FSF NNUE at 1K nodes),
 * not a human scale: the small net without search is roughly FSF at 1K nodes. */
const PRESETS: { id: string; label: string; settings: Partial<EngineSettings> }[] = [
  { id: 'casual', label: 'Casual - small net, instinct only, varied', settings: { net: 'small', limit: 'visits', visits: 1, temperature: 1, tempPlies: 0 } },
  { id: 'club', label: 'Club - small net, 16 visits', settings: { net: 'small', limit: 'visits', visits: 16, temperature: 0.5, tempPlies: 20 } },
  { id: 'strong', label: 'Strong - latest net, 64 visits', settings: { net: 'best', limit: 'visits', visits: 64, temperature: 0, tempPlies: 0 } },
  { id: 'full', label: 'Full - latest net, 800 visits (~+2300)', settings: { net: 'best', limit: 'visits', visits: 800, temperature: 0, tempPlies: 0 } },
  { id: 'max', label: 'Maximum - latest net, 10 s per move', settings: { net: 'best', limit: 'time', timeMs: 10000, temperature: 0, tempPlies: 0 } },
];

const STORAGE_KEY = 'sinsan.engineSettings.v1';

function load(): EngineSettings {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw) return { ...DEFAULTS, ...JSON.parse(raw) };
  } catch {
    // storage unavailable (private window, blocked): fall back to defaults
  }
  return { ...DEFAULTS };
}

function save(s: EngineSettings): void {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(s));
  } catch {
    // ignore: settings just won't persist
  }
}

/** Visits slider: 0..100 mapped logarithmically onto 1..20000. */
const visitsFromSlider = (v: number): number => Math.max(1, Math.round(Math.pow(20000, v / 100)));
const sliderFromVisits = (n: number): number => (100 * Math.log(Math.max(1, n))) / Math.log(20000);

export interface SettingsControl {
  get(): EngineSettings;
  /** Request body fields for /api/move. */
  requestFields(): Record<string, number | string>;
  summary(): string;
}

export function createEngineSettings(container: HTMLElement, onChange: () => void): SettingsControl {
  let s = load();
  container.classList.add('engine-settings');
  container.innerHTML = `
    <details>
      <summary><span class="es-title">Sinsan settings</span> <span class="es-summary"></span></summary>
      <div class="es-grid">
        <label class="es-wide">Preset
          <select data-k="preset"><option value="">Custom</option>
            ${PRESETS.map((p) => `<option value="${p.id}">${p.label}</option>`).join('')}</select></label>
        <label class="es-wide">Network <select data-k="net"></select></label>
        <label>Limit
          <select data-k="limit"><option value="visits">Visits per move</option><option value="time">Time per move</option></select></label>
        <label class="es-visits">Visits <output data-o="visits"></output>
          <input type="range" min="0" max="100" step="1" data-k="visits"></label>
        <label class="es-time">Seconds <output data-o="timeMs"></output>
          <input type="range" min="0.2" max="30" step="0.2" data-k="timeMs"></label>
        <label>Randomness <output data-o="temperature"></output>
          <input type="range" min="0" max="2" step="0.05" data-k="temperature"></label>
        <label>…for the first plies <output data-o="tempPlies"></output>
          <input type="range" min="0" max="60" step="2" data-k="tempPlies"></label>
        <label>Exploration (cpuct) <output data-o="cpuct"></output>
          <input type="range" min="0.5" max="4" step="0.05" data-k="cpuct"></label>
        <label>Unvisited-move penalty (FPU) <output data-o="fpu"></output>
          <input type="range" min="0" max="1" step="0.01" data-k="fpu"></label>
        <label>Policy softness <output data-o="ptemp"></output>
          <input type="range" min="0.5" max="3" step="0.02" data-k="ptemp"></label>
        <label>Point greed <output data-o="util"></output>
          <input type="range" min="0" max="0.5" step="0.01" data-k="util"></label>
        <button type="button" class="es-reset es-wide">Reset search settings to defaults</button>
        <p class="es-help es-wide">Visits = positions searched per move (MCTS has no fixed depth; the panel shows the depth it reaches).
          Randomness samples among searched moves instead of always playing the best. Exploration spreads the search wider;
          FPU and policy softness change how readily it tries moves the network dislikes. Point greed makes it prefer
          winning by more points. Changes apply from Sinsan's next move.</p>
      </div>
    </details>`;
  const q = <T extends Element>(sel: string) => container.querySelector<T>(sel)!;
  const input = (k: string) => q<HTMLInputElement | HTMLSelectElement>(`[data-k="${k}"]`);
  const out = (k: string) => container.querySelector<HTMLOutputElement>(`[data-o="${k}"]`);
  const summaryEl = q<HTMLSpanElement>('.es-summary');
  const netSelect = input('net') as HTMLSelectElement;

  function summary(): string {
    const net = netSelect.selectedOptions[0]?.textContent ?? s.net;
    const limit = s.limit === 'time' ? `${(s.timeMs / 1000).toFixed(1)}s/move` : `${s.visits} visits`;
    return `${net.split(' (')[0]} · ${limit}${s.temperature > 0 ? ` · random ${s.temperature}` : ''}`;
  }

  function render(): void {
    netSelect.value = s.net;
    input('limit').value = s.limit;
    input('visits').value = String(sliderFromVisits(s.visits));
    for (const k of ['timeMs', 'temperature', 'tempPlies', 'cpuct', 'fpu', 'ptemp', 'util'] as const) {
      input(k).value = String(k === 'timeMs' ? s.timeMs / 1000 : s[k]);
    }
    out('visits')!.textContent = s.visits.toLocaleString();
    out('timeMs')!.textContent = `${(s.timeMs / 1000).toFixed(1)}`;
    out('temperature')!.textContent = s.temperature === 0 ? 'off' : s.temperature.toFixed(2);
    out('tempPlies')!.textContent = s.tempPlies === 0 ? 'whole game' : String(s.tempPlies);
    out('cpuct')!.textContent = s.cpuct.toFixed(2);
    out('fpu')!.textContent = s.fpu.toFixed(2);
    out('ptemp')!.textContent = s.ptemp.toFixed(2);
    out('util')!.textContent = s.util === 0 ? 'off' : s.util.toFixed(2);
    container.classList.toggle('es-time-mode', s.limit === 'time');
    summaryEl.textContent = summary();
    const preset = PRESETS.find((p) => Object.entries(p.settings).every(([k, v]) => s[k as keyof EngineSettings] === v));
    input('preset').value = preset?.id ?? '';
  }

  function update(patch: Partial<EngineSettings>): void {
    s = { ...s, ...patch };
    save(s);
    render();
    onChange();
  }

  container.addEventListener('input', (ev) => {
    const t = ev.target as HTMLInputElement | HTMLSelectElement;
    const k = t.dataset['k'];
    if (!k) return;
    if (k === 'preset') {
      const p = PRESETS.find((x) => x.id === t.value);
      if (p) update(p.settings);
    } else if (k === 'net' || k === 'limit') {
      update({ [k]: t.value } as Partial<EngineSettings>);
    } else if (k === 'visits') {
      update({ visits: visitsFromSlider(Number(t.value)) });
    } else if (k === 'timeMs') {
      update({ timeMs: Math.round(Number(t.value) * 1000) });
    } else {
      update({ [k]: Number(t.value) } as Partial<EngineSettings>);
    }
  });
  q<HTMLButtonElement>('.es-reset').addEventListener('click', () =>
    update({ cpuct: DEFAULTS.cpuct, fpu: DEFAULTS.fpu, ptemp: DEFAULTS.ptemp, util: DEFAULTS.util, temperature: 0, tempPlies: 0 }));

  // Networks come from the server; keep the saved choice if it still exists.
  netSelect.innerHTML = `<option value="best">Latest RL</option>`;
  fetch('/api/nets')
    .then((r) => r.json())
    .then((nets: { id: string; label: string }[]) => {
      netSelect.innerHTML = nets.map((n) => `<option value="${n.id}">${n.label}</option>`).join('');
      if (!nets.some((n) => n.id === s.net)) s.net = 'best';
      render();
    })
    .catch(() => {
      // server not running: the AI menu's message explains it when a move is requested
    });
  render();

  return {
    get: () => s,
    summary,
    requestFields() {
      const f: Record<string, number | string> = {
        net: s.net, cpuct: s.cpuct, fpu: s.fpu, ptemp: s.ptemp, util: s.util,
        temperature: s.temperature, temp_plies: s.tempPlies === 0 ? 100000 : s.tempPlies,
      };
      if (s.limit === 'time') f['time_ms'] = s.timeMs;
      else f['visits'] = s.visits;
      return f;
    },
  };
}
