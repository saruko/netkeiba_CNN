import itertools
import numpy as np
import pytest
from src.policies.bet_policy import kelly_fractions, kelly_stakes


def expected_log_wealth(f, probs, odds):
    """結果 i が起きたときの資金 = 1 - Σf + f_i * o_i。1着馬が必ずどれか1頭である前提"""
    rest = 1 - f.sum()
    return float(np.sum(probs * np.log(rest + f * odds)))


def test_kelly_single_horse_matches_classic_formula():
    # 1頭のみ: f = (b*p - q) / b, b = o - 1 -> (2*0.5 - 0.5) / 2 = 0.25
    f = kelly_fractions(np.array([0.5, 0.5]), np.array([3.0, np.nan]))
    assert f == pytest.approx([0.25, 0.0])


def test_kelly_no_bet_when_no_positive_expectation():
    f = kelly_fractions(np.array([0.3, 0.3, 0.4]), np.array([3.0, 3.0, 2.4]))  # 期待値 0.9, 0.9, 0.96
    assert f == pytest.approx([0.0, 0.0, 0.0])


def test_kelly_multi_horse_is_optimal():
    probs = np.array([0.35, 0.25, 0.2, 0.2])
    odds = np.array([3.5, 5.0, 4.0, 3.0])  # 期待値 1.225, 1.25, 0.8, 0.6
    f = kelly_fractions(probs, odds)
    # 上位2頭購入後の R = 0.4 / (1 - 1/3.5 - 1/5) = 0.778 < 0.8 のため、期待値0.8の馬も保険として購入対象になる
    assert (f[:3] > 0).all() and f[3] == 0

    # 各馬の比率を単独・組で微小に動かしても期待対数資金が増えない(局所最適 = 凹関数なので大域最適)
    base = expected_log_wealth(f, probs, odds)
    eps = 1e-4
    for d in itertools.product([-eps, 0, eps], repeat=4):
        g = f + np.array(d)
        if (g < 0).any() or g.sum() > 1:
            continue
        assert expected_log_wealth(g, probs, odds) <= base + 1e-12


def test_kelly_stakes_applies_margin_threshold_and_rounding():
    probs = np.array([0.5, 0.3, 0.2])
    odds = np.array([3.0, 3.9, np.nan])
    # 実効オッズ(0.9倍): 2.7, 3.51 -> 期待値 1.35, 1.053。閾値1.1で馬1のみが候補
    stakes = kelly_stakes(probs, odds, bankroll=10000, kelly_fraction=0.5, odds_margin=0.9, min_expected_value=1.1)
    # 馬1のケリー比率: R = 0.5 / (1 - 1/2.7), f = 0.5 - R / 2.7 = 0.2059 -> 0.5 * f * 10000 = 1029 -> 1000円
    assert stakes == pytest.approx([1000, 0, 0])
    assert stakes.sum() <= 10000
