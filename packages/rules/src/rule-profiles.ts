import type { RuleProfile, RuleProfileId } from './types.ts';

/**
 * The four Janggi flavors Fairy-Stockfish ships (src/variant.cpp), matched rule for rule:
 *
 * - `kja` = FSF `janggi`, the default and what pychess.org plays: bikjang, point counting,
 *   threefold repetition, 100-ply no-capture limit. Verified against pyffish, including game
 *   endings, by tests/differential/compare_engine.py (via the Rust engine, which this mirrors).
 * - `traditional` = `janggitraditional`: as janggi, but endings are draws instead of point counts.
 * - `modern` = `janggimodern` (Kakao-style): no bikjang, point counting, fourfold repetition,
 *   200-ply limit. Known gap: FSF also makes some back-and-forth move repetitions illegal in this
 *   variant (moveRepetitionIllegal), which is not implemented here.
 * - `casual` = `janggicasual`: no bikjang, no point counting.
 */
export const RULE_PROFILES: Record<RuleProfileId, RuleProfile> = {
  kja: {
    id: 'kja',
    bikjangEndsGame: true,
    materialCountingAdjudication: true,
    repetitionLimit: 3,
    noCaptureMoveLimit: 100,
    hanCompensationPoints: 1.5,
  },
  traditional: {
    id: 'traditional',
    bikjangEndsGame: true,
    materialCountingAdjudication: false,
    repetitionLimit: 3,
    noCaptureMoveLimit: 100,
    hanCompensationPoints: 1.5,
  },
  modern: {
    id: 'modern',
    bikjangEndsGame: false,
    materialCountingAdjudication: true,
    repetitionLimit: 4,
    noCaptureMoveLimit: 200,
    hanCompensationPoints: 1.5,
  },
  casual: {
    id: 'casual',
    bikjangEndsGame: false,
    materialCountingAdjudication: false,
    repetitionLimit: 3,
    noCaptureMoveLimit: 100,
    hanCompensationPoints: 1.5,
  },
};

export const DEFAULT_RULE_PROFILE_ID: RuleProfileId = 'kja';

export function ruleProfileFor(id: RuleProfileId): RuleProfile {
  return RULE_PROFILES[id];
}
