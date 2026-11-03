import numpy as np
import pandas as pd
import pytest
from src.training.calibration import SoftmaxCalibrator
from src.policies.score_policy import CalibratedProbPolicy


def test_softmax_calibrator_recovers_temperature():
    """真の温度T=2のレース内softmaxから勝ち馬を生成した合成データで、Tが復元されることを検証する"""
    rng = np.random.default_rng(0)
    n_races, n_horses, true_t = 3000, 12, 2.0
    scores = rng.normal(0, 2, size=(n_races, n_horses))
    probs = np.exp(scores / true_t)
    probs /= probs.sum(axis=1, keepdims=True)
    winners = np.array([rng.choice(n_horses, p=p) for p in probs])
    is_winner = np.zeros_like(scores, dtype=bool)
    is_winner[np.arange(n_races), winners] = True
    race_ids = np.repeat(np.arange(n_races), n_horses)

    calib = SoftmaxCalibrator().fit(scores.ravel(), race_ids, is_winner.ravel())

    assert calib.temperature == pytest.approx(true_t, rel=0.1)


def test_softmax_probs_sum_to_one_per_race_and_are_stable():
    """レース内合計が1になること、および大きなスコアでもオーバーフローしないことを検証する"""
    calib = SoftmaxCalibrator()
    calib.temperature = 1.0
    df_race = pd.DataFrame({"race_id": ["a", "a", "b", "b", "b"]})
    scores = np.array([1000.0, 999.0, -5.0, 0.0, 5.0])

    probs = CalibratedProbPolicy(calib).calculate(scores, df_race)

    assert np.all(np.isfinite(probs))
    sums = pd.Series(probs).groupby(df_race["race_id"]).sum()
    assert sums.values == pytest.approx([1.0, 1.0])
    assert probs[0] == pytest.approx(1 / (1 + np.exp(-1)))


def _simulate_races(rng, n_races, n_horses, alpha, beta, score_informative=True):
    """市場の暗黙勝率 q と、モデルのスコア s を作り、p ∝ exp(alpha*s + beta*log q) に従って1着馬を生成する"""
    market_strength = rng.normal(0, 1, size=(n_races, n_horses))
    q = np.exp(market_strength)
    q /= q.sum(axis=1, keepdims=True)
    s = rng.normal(0, 1, size=(n_races, n_horses))
    logit = (alpha * s if score_informative else 0) + beta * np.log(q)
    p = np.exp(logit - logit.max(axis=1, keepdims=True))
    p /= p.sum(axis=1, keepdims=True)
    winners = np.array([rng.choice(n_horses, p=row) for row in p])
    is_winner = np.zeros((n_races, n_horses), dtype=bool)
    is_winner[np.arange(n_races), winners] = True
    race_ids = np.repeat(np.arange(n_races), n_horses)
    return s.ravel(), q.ravel(), race_ids, is_winner.ravel()


def test_market_combiner_recovers_coefficients():
    from src.training.calibration import MarketCombiner
    rng = np.random.default_rng(1)
    s, q, race_ids, is_winner = _simulate_races(rng, 4000, 12, alpha=0.5, beta=0.8)

    comb = MarketCombiner().fit(s, q, race_ids, is_winner)

    # 真の係数が推定値の ±3SE に入る
    assert abs(comb.alpha - 0.5) < 3 * comb.alpha_se
    assert abs(comb.beta - 0.8) < 3 * comb.beta_se
    probs = comb.predict_proba(s, q, race_ids)
    assert pd.Series(probs).groupby(race_ids).sum().values == pytest.approx(np.ones(4000))


def test_market_combiner_alpha_not_significant_for_noise_score():
    """モデルのスコアが勝敗と無関係なら、alpha は標準誤差に比べて有意にならない(エッジなしを検出できる)"""
    from src.training.calibration import MarketCombiner
    rng = np.random.default_rng(2)
    s, q, race_ids, is_winner = _simulate_races(rng, 4000, 12, alpha=0.0, beta=1.0, score_informative=False)

    comb = MarketCombiner().fit(s, q, race_ids, is_winner)

    assert abs(comb.alpha / comb.alpha_se) < 3
    assert comb.beta == pytest.approx(1.0, abs=3 * comb.beta_se)
