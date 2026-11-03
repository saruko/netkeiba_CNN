import os
import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
import yaml
from src.training.train import Trainer


def _race_log_softmax(z: np.ndarray, race_codes: np.ndarray) -> np.ndarray:
    """各行について log(exp(z_i) / Σ_{j∈同一レース} exp(z_j)) を返す(レースごとに最大値を引いて数値安定化)"""
    n_races = race_codes.max() + 1
    race_max = np.full(n_races, -np.inf)
    np.maximum.at(race_max, race_codes, z)
    shifted = z - race_max[race_codes]
    log_sum = np.log(np.bincount(race_codes, weights=np.exp(shifted), minlength=n_races))
    return shifted - log_sum[race_codes]


class SoftmaxCalibrator:
    """レース内softmax p_i = exp(s_i/T) / Σ_j exp(s_j/T) で「スコア→勝率」を校正する(条件付きロジット)。
    温度Tは校正データ上の1着馬の対数尤度を最大化して推定する。
    LambdaRankのスコアはレース内の相対値にしか意味がないため、全レースを混ぜて
    スコア→勝率を回帰する方式(Isotonic等)ではなく、レース内で確率を定義する。"""

    def fit(self, scores, race_ids, is_winner):
        s = np.asarray(scores, dtype=float)
        race_codes = pd.factorize(np.asarray(race_ids))[0]
        winner = np.asarray(is_winner).astype(bool)

        def neg_log_likelihood(log_t):
            return -_race_log_softmax(s / np.exp(log_t), race_codes)[winner].sum()

        res = minimize_scalar(neg_log_likelihood, bounds=(-5.0, 5.0), method="bounded")
        self.temperature = float(np.exp(res.x))
        return self

    def predict_proba(self, scores, race_ids) -> np.ndarray:
        race_codes = pd.factorize(np.asarray(race_ids))[0]
        z = np.asarray(scores, dtype=float) / self.temperature
        return np.exp(_race_log_softmax(z, race_codes))


class Calibrator:
    def __init__(self, config_path="config/config.yaml"):
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        base_dir = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        self.models_dir = os.path.join(base_dir, self.config["paths"]["models_dir"])
        self.trainer = Trainer(config_path)

    def run(self):
        # 検証データのロード
        # early stopping に使用したデータと同一データでキャリブレーションを学習すると
        # 校正が楽観的になるため、検証データの後半(early stoppingに使われていない部分)のみを使う。
        print("Loading validation dataset for calibration...")
        df_val_all = self.trainer.load_dataset([self.trainer.val_year])
        _, df_val = self.trainer.split_validation(df_val_all)

        # 学習時と同一のカテゴリ水準を適用する(水準がズレるとスコアが黙って狂う)
        df_val = self.trainer.apply_categories(df_val)

        # モデルのロード
        model_path = os.path.join(self.models_dir, "lgb_lambdarank.model")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}. Train model first.")
        model = joblib.load(model_path)

        # 予測スコア (Raw Score)
        raw_scores = model.predict(df_val[self.trainer.features])

        # キャリブレーションの学習 (ターゲット: 1着なら1, その他は0)
        print("Fitting race-wise softmax temperature for calibration...")
        calib = SoftmaxCalibrator().fit(raw_scores, df_val["race_id"], df_val["rank_num"] == 1)
        print(f"Fitted temperature: {calib.temperature:.4f}")

        # 保存
        calib_path = os.path.join(self.models_dir, "softmax_calibration.model")
        joblib.dump(calib, calib_path)
        print(f"Calibration model saved to {calib_path}")

if __name__ == "__main__":
    calibrator = Calibrator()
    calibrator.run()
