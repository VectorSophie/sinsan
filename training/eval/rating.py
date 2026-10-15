"""Elo estimates with error bars, multi-player Bradley-Terry ratings, and a pentanomial SPRT.

Games are played in pairs (same opening, colors swapped), so pair outcomes are not independent
game outcomes; error bars use the pentanomial model (per-pair score in {0, .5, 1, 1.5, 2}), as
Fishtest does.
"""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np


def elo_from_score(score: float) -> float:
    score = min(max(score, 1e-6), 1 - 1e-6)
    return -400.0 * math.log10(1.0 / score - 1.0)


def pentanomial_elo(pair_scores: list[float]) -> tuple[float, float, float]:
    """Pair scores (A's points over both games of a pair, 0..2) -> (elo, lo95, hi95)."""
    n = len(pair_scores)
    if n == 0:
        return 0.0, -math.inf, math.inf
    x = np.asarray(pair_scores, dtype=float) / 2.0
    mean = float(x.mean())
    se = float(x.std(ddof=1) / math.sqrt(n)) if n > 1 else 0.5
    return elo_from_score(mean), elo_from_score(mean - 1.96 * se), elo_from_score(mean + 1.96 * se)


def bradley_terry(results: list[tuple[str, str, float]], anchor: str | None = None, anchor_elo: float = 0.0,
                  prior_games: float = 1.0) -> dict[str, tuple[float, float]]:
    """Maximum-likelihood ratings from (a, b, score_of_a) game results, score in {0, .5, 1}.

    A weak prior (prior_games virtual draws against a rating-0 phantom) keeps players with all wins
    or all losses finite. Returns {player: (elo, stderr)} with the anchor fixed at anchor_elo.
    """
    players = sorted({p for a, b, _ in results for p in (a, b)})
    idx = {p: i for i, p in enumerate(players)}
    k = len(players)
    wins = defaultdict(float)
    games = defaultdict(float)
    for a, b, s in results:
        i, j = idx[a], idx[b]
        wins[(i, j)] += s
        wins[(j, i)] += 1 - s
        games[(i, j)] += 1
        games[(j, i)] += 1

    r = np.zeros(k)  # natural-log strength scale
    for _ in range(200):
        grad = np.zeros(k)
        hess = np.zeros((k, k))
        for (i, j), g in games.items():
            p = 1 / (1 + math.exp(r[j] - r[i]))
            grad[i] += wins[(i, j)] - g * p
            hess[i, i] -= g * p * (1 - p)
            hess[i, j] += g * p * (1 - p)
        # prior: draws against a phantom at 0
        p0 = 1 / (1 + np.exp(-r))
        grad += prior_games * (0.5 - p0)
        hess[np.diag_indices(k)] -= prior_games * p0 * (1 - p0)
        step = np.linalg.solve(hess, -grad)
        r += step
        if np.abs(step).max() < 1e-9:
            break
    cov = np.linalg.inv(-hess)
    scale = 400 / math.log(10)
    shift = r[idx[anchor]] if anchor else 0.0
    out = {}
    for p, i in idx.items():
        var = cov[i, i] + (cov[idx[anchor], idx[anchor]] - 2 * cov[i, idx[anchor]] if anchor else 0.0)
        out[p] = ((r[i] - shift) * scale + anchor_elo, math.sqrt(max(var, 0.0)) * scale)
    return out


def sprt_llr(pair_scores: list[float], elo0: float, elo1: float) -> float:
    """Generalized SPRT log-likelihood ratio on pentanomial pair scores for logistic Elo bounds."""
    n = len(pair_scores)
    if n < 2:
        return 0.0
    x = np.asarray(pair_scores, dtype=float) / 2.0
    mean, var = float(x.mean()), float(x.var(ddof=0))
    if var <= 0:
        return 0.0
    s0 = 1 / (1 + 10 ** (-elo0 / 400))
    s1 = 1 / (1 + 10 ** (-elo1 / 400))
    return n * (s1 - s0) * (2 * mean - s0 - s1) / (2 * var)


def sprt_bounds(alpha: float = 0.05, beta: float = 0.05) -> tuple[float, float]:
    return math.log(beta / (1 - alpha)), math.log((1 - beta) / alpha)
