/**
 * The 60 move templates (Section 9): 36 orthogonal (4 directions x distance 1-9) + 8 palace
 * diagonal (4 directions x distance 1-2) + 8 horse + 8 elephant. Each template is a fixed
 * (row, col) offset from an origin square; a template is only used by a real move if applying it
 * to the piece's origin explains the piece's actual (from, to) delta.
 *
 * Order is fixed and never renumbered once published (dataset labels and the browser runtime
 * both index into this exact array) - see docs/MODEL_DESIGN.md.
 */
export interface Template {
  readonly dr: number;
  readonly dc: number;
}

const ORTHOGONAL_DIRS: ReadonlyArray<readonly [number, number]> = [
  [-1, 0], // north
  [1, 0], // south
  [0, -1], // west
  [0, 1], // east
];

const DIAGONAL_DIRS: ReadonlyArray<readonly [number, number]> = [
  [-1, -1], // northwest
  [-1, 1], // northeast
  [1, -1], // southwest
  [1, 1], // southeast
];

const HORSE_OFFSETS: ReadonlyArray<readonly [number, number]> = [
  [-2, -1],
  [-2, 1],
  [2, -1],
  [2, 1],
  [-1, -2],
  [1, -2],
  [-1, 2],
  [1, 2],
];

const ELEPHANT_OFFSETS: ReadonlyArray<readonly [number, number]> = [
  [-3, -2],
  [-3, 2],
  [3, -2],
  [3, 2],
  [-2, -3],
  [2, -3],
  [-2, 3],
  [2, 3],
];

function buildTemplates(): Template[] {
  const templates: Template[] = [];
  for (const [dr, dc] of ORTHOGONAL_DIRS) {
    for (let dist = 1; dist <= 9; dist++) templates.push({ dr: dr * dist, dc: dc * dist });
  }
  for (const [dr, dc] of DIAGONAL_DIRS) {
    for (let dist = 1; dist <= 2; dist++) templates.push({ dr: dr * dist, dc: dc * dist });
  }
  for (const [dr, dc] of HORSE_OFFSETS) templates.push({ dr, dc });
  for (const [dr, dc] of ELEPHANT_OFFSETS) templates.push({ dr, dc });
  return templates;
}

export const TEMPLATES: readonly Template[] = buildTemplates();
export const TEMPLATE_COUNT = TEMPLATES.length; // 60

if (TEMPLATE_COUNT !== 60) {
  throw new Error(`Expected exactly 60 move templates, built ${TEMPLATE_COUNT}`);
}

/** Template index -> template index after a horizontal board reflection (column mirrored).
 * Orthogonal north/south templates (dc===0) are unaffected; east/west swap; diagonal quadrants
 * mirror left<->right; horse/elephant offsets mirror their dc sign. Built by matching each
 * template's (dr,-dc) against the table, rather than hand-listing 60 index pairs. */
export const REFLECTED_TEMPLATE_INDEX: readonly number[] = TEMPLATES.map((t) => {
  const mirrored = TEMPLATES.findIndex((candidate) => candidate.dr === t.dr && candidate.dc === -t.dc);
  if (mirrored === -1) throw new Error(`No mirrored template found for dr=${t.dr} dc=${t.dc}`);
  return mirrored;
});
