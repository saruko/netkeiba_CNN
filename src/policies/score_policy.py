from abc import ABC, abstractmethod
import pandas as pd
import numpy as np

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
    """校正済み勝率(期待値計算用)。各レース内の合計が1になるよう正規化する"""
    def __init__(self, calibration_model):
        self.calibration_model = calibration_model

    def calculate(self, raw_scores: np.ndarray, df_race: pd.DataFrame) -> np.ndarray:
        probs = self.calibration_model.predict(raw_scores)
        
        df = df_race.copy()
        df["prob"] = probs
        prob_sums = df.groupby("race_id")["prob"].transform("sum")
        
        normalized_probs = np.zeros_like(probs)
        valid_mask = (prob_sums > 0)
        
        normalized_probs[valid_mask] = probs[valid_mask] / prob_sums[valid_mask]
        
        # 合計が0の場合は、頭数で均等に割り振る
        zero_mask = ~valid_mask
        if zero_mask.any():
            n_horses = df.groupby("race_id")["prob"].transform("count")
            normalized_probs[zero_mask] = 1.0 / n_horses[zero_mask]
            
        return normalized_probs
