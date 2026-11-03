import os
import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
import yaml
from src.training.train import Trainer
from src.policies.score_policy import market_implied_probs


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


def _fit_conditional_logit(X: np.ndarray, race_codes: np.ndarray, winner: np.ndarray, x0: np.ndarray):
    """条件付きロジット p_i = exp(x_i·w) / Σ_{j∈同一レース} exp(x_j·w) の係数wを、1着馬の対数尤度最大化で推定する。
    解析的な勾配・ヘッセ行列を用い、係数と標準誤差(観測フィッシャー情報量の逆行列の対角)を返す。"""
    n_races = race_codes.max() + 1
    k = X.shape[1]
    n_winners = np.bincount(race_codes[winner], minlength=n_races)
    x_winner_sum = X[winner].sum(axis=0)

    def race_moments(w):
        p = np.exp(_race_log_softmax(X @ w, race_codes))
        px = np.column_stack([np.bincount(race_codes, weights=p * X[:, a], minlength=n_races) for a in range(k)])
        return p, px

    def nll(w):
        return -_race_log_softmax(X @ w, race_codes)[winner].sum()

    def grad(w):
        _, px = race_moments(w)
        return -(x_winner_sum - (n_winners[:, None] * px).sum(axis=0))

    def hess(w):
        p, px = race_moments(w)
        h = np.empty((k, k))
        for a in range(k):
            for b in range(k):
                pxx = np.bincount(race_codes, weights=p * X[:, a] * X[:, b], minlength=n_races)
                h[a, b] = (n_winners * (pxx - px[:, a] * px[:, b])).sum()
        return h

    res = minimize(nll, x0, jac=grad, hess=hess, method="trust-exact")
    se = np.sqrt(np.diag(np.linalg.inv(hess(res.x))))
    return res.x, se


class MarketCombiner:
    """モデルのスコア s と市場の暗黙勝率 q を、レース内softmax p ∝ exp(α·s + β·log q) で結合する
    (2段階条件付きロジット)。オッズは特徴量として学習には使わず、この結合段階でのみ使う。
    α が標準誤差に比べて十分に正でなければ、モデルは市場に織り込まれていない情報を持っていない。"""

    MIN_MARKET_PROB = 1e-6  # オッズ欠損馬(q=0)の log q を有限にするための下限

    def _design(self, scores, market_probs):
        q = np.clip(np.asarray(market_probs, dtype=float), self.MIN_MARKET_PROB, None)
        return np.column_stack([np.asarray(scores, dtype=float), np.log(q)])

    def fit(self, scores, market_probs, race_ids, is_winner):
        race_codes = pd.factorize(np.asarray(race_ids))[0]
        winner = np.asarray(is_winner).astype(bool)
        # 初期値は「市場そのまま」(α=0, β=1)
        coef, se = _fit_conditional_logit(self._design(scores, market_probs), race_codes, winner, np.array([0.0, 1.0]))
        self.alpha, self.beta = float(coef[0]), float(coef[1])
        self.alpha_se, self.beta_se = float(se[0]), float(se[1])
        return self

    def predict_proba(self, scores, market_probs, race_ids) -> np.ndarray:
        race_codes = pd.factorize(np.asarray(race_ids))[0]
        z = self._design(scores, market_probs) @ np.array([self.alpha, self.beta])
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

        # 市場との結合モデル(同じ校正データの確定オッズで推定する)
        print("Fitting market combiner (model score + market implied probability)...")
        combiner = MarketCombiner().fit(raw_scores, market_implied_probs(df_val), df_val["race_id"], df_val["rank_num"] == 1)
        print(f"alpha (model): {combiner.alpha:.4f} (SE {combiner.alpha_se:.4f}), "
              f"beta (market): {combiner.beta:.4f} (SE {combiner.beta_se:.4f})")
        combiner_path = os.path.join(self.models_dir, "market_combiner.model")
        joblib.dump(combiner, combiner_path)
        print(f"Market combiner saved to {combiner_path}")

if __name__ == "__main__":
    calibrator = Calibrator()
    calibrator.run()
