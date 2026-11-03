from abc import ABC, abstractmethod
import pandas as pd
import numpy as np

def market_implied_probs(df: pd.DataFrame) -> np.ndarray:
    """単勝オッズから市場の暗黙勝率を求める: (1/オッズ) をレース内で合計1に正規化(控除率分を除去)"""
    inv = 1.0 / df["tansho_odds"].where(df["tansho_odds"] > 0)
    return (inv / inv.groupby(df["race_id"]).transform("sum")).fillna(0).values


class ScorePolicy(ABC):
    @abstractmethod
    def calculate(self, raw_scores: np.ndarray, df_race: pd.DataFrame) -> np.ndarray:
        """予測スコアを各ポリシーに基づいて変換する"""
        pass

class RawScorePolicy(ScorePolicy):
    """モデル出力をそのままスコアとする"""
    def calculate(self, raw_scores: np.ndarray, df_race: pd.DataFrame) -> np.ndarray:
        return raw_scores

class StdScorePolicy(ScorePolicy):
    """レース内でスコアを標準化する(平均0, 標準偏差1)"""
    def calculate(self, raw_scores: np.ndarray, df_race: pd.DataFrame) -> np.ndarray:
        df = df_race.copy()
        df["raw_score"] = raw_scores
        
        grouped = df.groupby("race_id")["raw_score"]
        means = grouped.transform("mean")
        stds = grouped.transform("std")
        
        # 標準偏差が0または欠損の場合は標準化スコアを0とする
        std_zero_mask = (stds == 0) | stds.isna()
        std_scores = np.zeros_like(raw_scores)
        std_scores[~std_zero_mask] = (raw_scores[~std_zero_mask] - means[~std_zero_mask]) / stds[~std_zero_mask]
        
        return std_scores

class CalibratedProbPolicy(ScorePolicy):
    """校正済み勝率(期待値計算用)。校正モデル(SoftmaxCalibrator)がレース内合計1の勝率を返す"""
    def __init__(self, calibration_model):
        self.calibration_model = calibration_model

    def calculate(self, raw_scores: np.ndarray, df_race: pd.DataFrame) -> np.ndarray:
        return self.calibration_model.predict_proba(raw_scores, df_race["race_id"].values)


class MarketCombinedProbPolicy(ScorePolicy):
    """モデルのスコアと市場の暗黙勝率(df_race の tansho_odds から算出)を MarketCombiner で結合した勝率"""
    def __init__(self, combiner):
        self.combiner = combiner

    def calculate(self, raw_scores: np.ndarray, df_race: pd.DataFrame) -> np.ndarray:
        return self.combiner.predict_proba(raw_scores, market_implied_probs(df_race), df_race["race_id"].values)
