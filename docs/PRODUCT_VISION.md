# Product Vision

## What Sinsan is

Simultaneously: (1) a polished Korean Janggi web application, (2) a learned
browser model distilled from a strong Janggi engine, (3) a reproducible
research project (code, data, models, evaluations, documented decisions),
and (4) an open-source project published on GitHub and Hugging Face. It is
not only a board game site and not only an AI demo — it's a demonstration,
backed by measurement, of how much real Janggi strength survives
compression into an artifact under 500KiB.

Priority order when these pull in different directions (Section 2):
strength-relative-to-size > rule correctness > browser accessibility >
reproducibility > honest evaluation > product polish. Polish never trades
against rule correctness or honest evaluation.

## On the name — verified, not invented

**神算 (Sinsan, 신산)** is not a name invented for this project or borrowed
from an unverified folk association. It's a real, documented nickname in
Korean Go (Baduk — not Janggi) for professional player Lee Chang-ho
(이창호): "신산(神算)은 계산력이 신의 경지에 도달했다는 뜻" — "Sinsan means
one's calculation ability has reached a divine level," describing his
renowned endgame precision and imperturbable playing manner. Confirmed via
multiple independent Korean-language sources ([티타임즈](https://www.ttimes.co.kr/article/2019121911167754823),
[나무위키 이창호 항목](https://namu.wiki/w/%EC%9D%B4%EC%B0%BD%ED%98%B8)).

**This is explicitly disclosed, not hidden:** research found no evidence
connecting 神算/신산 to any Janggi master, historical or contemporary. Using
it here is a deliberate borrowing of a Go term's plain meaning
("extraordinary, foresightful calculation") for a Janggi project — which
happens to mirror the project's own origin story: Moka took a Go-distillation
idea and this project reimagines it for Janggi. The name follows the same
move, one level up. Any future copy (README, about page, model card) that
discusses the name in historical or cultural terms must carry this same
citation and disclaimer, not present it as an ancient or Janggi-specific
title.

## Visual identity

Korean identity should emerge from restraint and materiality, not
decoration (Section 6). Explicitly avoided: dragons, excessive gold, taeguk
symbols, hanok silhouettes, fake brush-calligraphy textures, generic
East-Asian motifs, tourism-poster aesthetics. Decorative Chinese characters
are not scattered through the UI — 神算 is used carefully as the core
wordmark, and piece characters are the only other Chinese characters that
appear, because they are the pieces, not decoration.

Recommended direction (unchanged from spec, no research contradicted it):
warm, moderately light wood board; Cho in blue/blue-green, Han in deep
red/red-brown; warm-white or muted-neutral page background; ink-like dark
text; brass/gold reserved for small highlights only; modern, minimal
interface; physical depth (shadow, lift, easing) reserved primarily for
board and pieces, not chrome around them; research pages stay clean,
numerical, and understated — the numbers are the point there, not
illustration.

## Interaction feel

Human move ~180ms, AI move ~230ms, selected-piece lift ~3px with a
slightly stronger shadow, short physical cubic easing, sub-80ms landing
compression, 100-140ms capture fade/shrink, visible last-move markers,
restrained legal-destination dots/rings, a brief check pulse around the
palace/general, `prefers-reduced-motion` respected throughout. Game end
does not shake or explode the board. This is detail for Phase 2
implementation; recorded here so the vision and the eventual
`packages/ui` implementation don't drift apart.

## What Sinsan explicitly is not trying to be

Not a maximum-strength engine — professional top-level strength is
explicitly deprioritized versus strength-per-byte, rule correctness, and
honest evaluation (Section 2). Not a platform — no accounts, no server-side
inference in normal play, no enterprise scaffolding. Not a mechanical Moka
reskin — every Go-specific assumption (action space, board size, network
shape, protocol glue) is redesigned for Janggi, with the reasoning recorded
in `RESEARCH.md` and `REFERENCES.md` rather than copied silently.
