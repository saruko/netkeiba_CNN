from abc import ABC, abstractmethod
import pandas as pd
import numpy as np

class BetPolicy(ABC):
    @abstractmethod
    def decide_bets(self, df_race: pd.DataFrame, scores: np.ndarray) -> np.ndarray:
        """各馬の購入判定(True/False)を決定する"""
        pass

class ThresholdBet(BetPolicy):
    """スコア(標準化スコアなど)が指定した閾値を超えた馬を一律購入する"""
    def __init__(self, threshold: float = 2.0):
        self.threshold = threshold

    def decide_bets(self, df_race: pd.DataFrame, scores: np.ndarray) -> np.ndarray:
        return scores >= self.threshold

class ExpectedValueBet(BetPolicy):
    """期待値(校正勝率 * 実効オッズ)が指定した閾値を超えた馬を購入する"""
    def __init__(self, min_expected_value: float = 1.1, odds_margin: float = 0.9):
        self.min_expected_value = min_expected_value
        self.odds_margin = odds_margin

    def decide_bets(self, df_race: pd.DataFrame, scores: np.ndarray) -> np.ndarray:
        if "tansho_odds" not in df_race.columns:
            raise KeyError("ExpectedValueBet requires 'tansho_odds' column in df_race")
            
        odds = df_race["tansho_odds"].values
        # 期待値の計算 (オッズマージンを適用)
        expected_values = scores * (odds * self.odds_margin)
        
        # 期待値が閾値以上で、オッズが正常値である馬を判定
        decision = (expected_values >= self.min_expected_value) & (odds > 0) & (~np.isnan(odds))
        return decision


def kelly_fractions(probs: np.ndarray, odds: np.ndarray) -> np.ndarray:
    """1レースの単勝(排他的な結果)に同時に賭けるときのケリー基準の賭け比率。
    期待対数資金を最大化する解析解(Kelly 1956 / Smoczynski & Tomkins 2010):
    期待値 p*o の大きい順に、p*o が「残す資金の比率 R」を上回る間だけ購入対象に加え、
    R = (1 - Σp) / (1 - Σ1/o)、比率 f_i = p_i - R / o_i とする。
    odds が NaN または 1 以下の馬は購入対象にしない。返り値は資金に対する比率(合計 <= 1)。"""
    probs = np.asarray(probs, dtype=float)
    odds = np.asarray(odds, dtype=float)
    fractions = np.zeros(len(probs))
    valid = np.where((odds > 1) & (probs > 0))[0]
    order = valid[np.argsort(-(probs[valid] * odds[valid]))]
    sum_p = sum_inv = 0.0
    reserve = 1.0
    chosen = []
    for i in order:
        if probs[i] * odds[i] <= reserve or sum_inv + 1 / odds[i] >= 1:
            break
        chosen.append(i)
        sum_p += probs[i]
        sum_inv += 1 / odds[i]
        reserve = (1 - sum_p) / (1 - sum_inv)
    chosen = np.array(chosen, dtype=int)
    fractions[chosen] = probs[chosen] - reserve / odds[chosen]
    return fractions


def kelly_stakes(probs: np.ndarray, tansho_odds: np.ndarray, bankroll: float, kelly_fraction: float,
                 odds_margin: float, min_expected_value: float, unit: int = 100) -> np.ndarray:
    """1レースの単勝の購入金額(円, unit 単位で切り捨て)。
    実効オッズ(オッズ * odds_margin)での期待値が min_expected_value 以上の馬だけを候補とし、
    候補内でケリー比率を求めて kelly_fraction 倍(分数ケリー)する。購入合計は bankroll を超えない。"""
    eff_odds = np.asarray(tansho_odds, dtype=float) * odds_margin
    probs = np.asarray(probs, dtype=float)
    candidate = probs * eff_odds >= min_expected_value  # NaN は False
    f = kelly_fractions(probs, np.where(candidate, eff_odds, np.nan))
    return np.floor(kelly_fraction * f * bankroll / unit) * unit
