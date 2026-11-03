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
