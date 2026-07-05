import pandas as pd
import pytest
from src.simulation.simulator import evaluate_bets


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
