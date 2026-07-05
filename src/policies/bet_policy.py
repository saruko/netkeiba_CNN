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
