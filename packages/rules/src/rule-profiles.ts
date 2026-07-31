import type { RuleProfile, RuleProfileId } from './types.ts';

/**
 * Maps onto Fairy-Stockfish's four hardcoded Janggi variant flavors (src/variant.cpp), confirmed
 * during Phase 0 research - see docs/RULES.md. `kja` is the project's default and is named for
 * intent (closest match to competitive play: bikjang + material-count adjudication both active),
 * not verified confirmation of the Korea Janggi Association's exact rules - that verification is
 * currently blocked by a site access issue, documented in docs/RESEARCH.md.
 */
export const RULE_PROFILES: Record<RuleProfileId, RuleProfile> = {
  kja: {
    id: 'kja',
    bikjangEndsGame: true,
    bikjangResult: 'draw',
    materialCountingAdjudication: true,
    repetitionLimit: 3,
    noCaptureMoveLimit: 200,
    hanCompensationPoints: 1.5,
  },
  traditional: {
    id: 'traditional',
    bikjangEndsGame: true,
    bikjangResult: 'draw',
    materialCountingAdjudication: false,
    repetitionLimit: 3,
    noCaptureMoveLimit: 200,
    hanCompensationPoints: 1.5,
  },
  modern: {
    id: 'modern',
    bikjangEndsGame: false,
    bikjangResult: 'draw',
    materialCountingAdjudication: true,
    repetitionLimit: 4,
    noCaptureMoveLimit: 100,
    hanCompensationPoints: 1.5,
  },
  casual: {
    id: 'casual',
    bikjangEndsGame: false,
    bikjangResult: 'draw',
    materialCountingAdjudication: false,
    repetitionLimit: 3,
    noCaptureMoveLimit: 200,
    hanCompensationPoints: 1.5,
  },
};

export const DEFAULT_RULE_PROFILE_ID: RuleProfileId = 'kja';

export function ruleProfileFor(id: RuleProfileId): RuleProfile {
  return RULE_PROFILES[id];
}
