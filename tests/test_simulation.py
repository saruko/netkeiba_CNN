import numpy as np
import pandas as pd
import pytest
from src.simulation.simulator import (
    evaluate_bets, market_implied_probs, race_prob_metrics, bootstrap_recovery_ci
)


def test_evaluate_bets_known_payoff():
    """既知の払戻データに対して回収率・的中率が正しく計算されることを検証する(仕様書§10)"""
    df_test = pd.DataFrame({
        "race_id": ["202001010101", "202001010101", "202001010102"],
        "horse_num": [1, 2, 1],
        "should_bet": [True, True, True],
    })
    df_returns = pd.DataFrame({
        "race_id": ["202001010101", "202001010101", "202001010102", "202001010102"],
        "ticket_type": ["単勝", "複勝", "単勝", "複勝"],
        "horse_num": [1, 1, 2, 2],
        "payoff": [250, 120, 500, 180],
    })

    result = evaluate_bets(df_test, "should_bet", df_returns, "test")

    assert result["n_bets_test"] == 3
    # 単勝的中: race1-horse1 (250円) のみ。race1-horse2, race2-horse1 は不的中(0円)
    assert result["hit_rate_test"] == pytest.approx(1 / 3 * 100)
    assert result["recovery_rate_test"] == pytest.approx(250 / 300 * 100)
    # 複勝払戻: race1-horse1(120円)のみ実際のベット馬番(1)と一致。race2は的中馬番が2でベットは1なので不的中
    assert result["hit_rate_place_test"] == pytest.approx(1 / 3 * 100)
    assert result["recovery_rate_place_test"] == pytest.approx(120 / 300 * 100)


def test_evaluate_bets_no_bets():
    df_test = pd.DataFrame({
        "race_id": ["202001010101"],
        "horse_num": [1],
        "should_bet": [False],
    })
    df_returns = pd.DataFrame({"race_id": [], "ticket_type": [], "horse_num": [], "payoff": []})

    result = evaluate_bets(df_test, "should_bet", df_returns, "test")

    assert result["n_bets_test"] == 0
    assert result["recovery_rate_test"] == 0.0
    assert result["hit_rate_test"] == 0.0


def test_market_implied_probs_normalized_per_race():
    df = pd.DataFrame({
        "race_id": ["r1", "r1", "r2", "r2"],
        "tansho_odds": [2.0, 2.0, 1.6, None],
    })
    probs = market_implied_probs(df)
    # r1: 1/2, 1/2 -> 0.5, 0.5 / r2: オッズ欠損馬は0、残りで合計1
    assert probs == pytest.approx([0.5, 0.5, 1.0, 0.0])


def test_race_prob_metrics_known_values():
    df = pd.DataFrame({
        "race_id": ["r1", "r1", "r2", "r2", "r3", "r3"],
        "rank_num": [1, 2, 2, 1, None, None],  # r3 は1着馬なし(除外される)
        "p": [0.8, 0.2, 0.6, 0.4, 0.5, 0.5],
    })
    m = race_prob_metrics(df, "p")
    assert m["n_races"] == 2
    assert m["log_loss"] == pytest.approx((-np.log(0.8) - np.log(0.4)) / 2)
    # r1: (0.8-1)^2 + 0.2^2 = 0.08 / r2: 0.6^2 + (0.4-1)^2 = 0.72
    assert m["brier"] == pytest.approx((0.08 + 0.72) / 2)


def test_bootstrap_recovery_ci():
    # 全レース同一の払戻なら信頼区間は点推定(150%)に一致する
    df_const = pd.DataFrame({"race_id": ["r1", "r2", "r3"], "payoff": [150, 150, 150]})
    assert bootstrap_recovery_ci(df_const) == pytest.approx((150.0, 150.0))

    # ばらつきがある場合、区間は点推定を含み、払戻の最小・最大の範囲内に収まる
    df_var = pd.DataFrame({"race_id": [f"r{i}" for i in range(50)],
                           "payoff": [0] * 45 + [1000] * 5})
    lower, upper = bootstrap_recovery_ci(df_var)
    assert 0.0 <= lower < 100.0 < upper <= 1000.0


def test_bootstrap_logloss_diff_ci():
    from src.simulation.simulator import bootstrap_logloss_diff_ci
    # 全レースで a の1着馬確率が 0.5、b が 0.25 -> 差は log(0.25) - log(0.5) = -log 2 で一定
    df = pd.DataFrame({
        "race_id": ["r1", "r1", "r2", "r2"],
        "rank_num": [1, 2, 2, 1],
        "a": [0.5, 0.5, 0.5, 0.5],
        "b": [0.25, 0.75, 0.75, 0.25],
    })
    mean, lower, upper = bootstrap_logloss_diff_ci(df, "a", "b")
    assert mean == pytest.approx(-np.log(2))
    assert (lower, upper) == pytest.approx((-np.log(2), -np.log(2)))
