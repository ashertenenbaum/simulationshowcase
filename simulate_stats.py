"""Lightweight match-stat simulation helper."""
import numpy as np

RNG = np.random.default_rng()

def _bounded(val, low, high):
    return int(max(low, min(high, round(val))))

def simulate_match_stats(row):
    elo_diff = row.get("HomeElo", 1500) - row.get("AwayElo", 1500)
    shot_delta = elo_diff / 80
    home_shots = _bounded(RNG.poisson(10 + shot_delta), 3, 25)
    away_shots = _bounded(RNG.poisson(10 - shot_delta), 3, 25)
    home_fouls = _bounded(RNG.normal(12, 3), 5, 30)
    away_fouls = _bounded(RNG.normal(12, 3), 5, 30)
    home_yellows = _bounded(RNG.binomial(home_fouls, 0.15), 0, 6)
    away_yellows = _bounded(RNG.binomial(away_fouls, 0.15), 0, 6)
    home_reds = int(RNG.random() < 0.02 + 0.01 * (home_yellows > 3))
    away_reds = int(RNG.random() < 0.02 + 0.01 * (away_yellows > 3))
    return dict(
        HomeShots=home_shots, AwayShots=away_shots,
        HomeFouls=home_fouls, AwayFouls=away_fouls,
        HomeYellows=home_yellows, AwayYellows=away_yellows,
        HomeReds=home_reds, AwayReds=away_reds,
    )
