import os
import pandas as pd
import numpy as np
import joblib
import yaml
import sqlite3
from src.scraping.netkeiba_source import NetkeibaSource
from src.preprocessing.parser import NetkeibaParser
from src.preprocessing.features import FeatureBuilder
from src.policies.score_policy import CalibratedProbPolicy, MarketCombinedProbPolicy
from src.policies.bet_policy import ExpectedValueBet, kelly_stakes
from src.training.train import Trainer

class Predictor:
    def __init__(self, config_path="config/config.yaml"):
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)
            
        base_dir = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        self.models_dir = os.path.join(base_dir, self.config["paths"]["models_dir"])
        self.trainer = Trainer(config_path)
        self.source = NetkeibaSource(config_path)
        self.parser = NetkeibaParser(config_path)
        self.builder = FeatureBuilder(config_path)
        self.odds_margin = self.config["betting"]["odds_margin"]
        self.min_expected_value = self.config["betting"]["min_expected_value"]
        self.kelly_fraction = self.config["betting"]["kelly_fraction"]

    def predict_race(self, race_id: str, budget: float = None):
        # 1. 出馬表のHTML取得とパース
        print(f"Fetching shutuba HTML for {race_id}...")
        self.source.get_shutuba_html(race_id)
        self.parser.parse_shutuba(race_id)
        
        # 2. 出馬表の馬情報を取得
        # SQLiteからhorse_idを読み出す
        conn = sqlite3.connect(self.parser.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT horse_id FROM results WHERE race_id = ?", (race_id,))
        horse_ids = [row[0] for row in cursor.fetchall()]
        conn.close()
        
        # 出走馬の成績と血統の取得・パース
        for h_id in horse_ids:
            print(f"Updating data for horse: {h_id}...")
            self.source.get_horse_html(h_id, force=True)
            self.source.get_ped_html(h_id)
            self.parser.parse_horse(h_id)
            self.parser.parse_ped(h_id)
            
        # 3. 特徴量生成
        print("Building features...")
        df_race = self.builder.build_features(race_ids=[race_id])
        if df_race.empty:
            print(f"No data found for race_id: {race_id}")
            return
            
        # 4. モデルロード
        model_path = os.path.join(self.models_dir, "lgb_lambdarank.model")
        calib_path = os.path.join(self.models_dir, "softmax_calibration.model")
        
        if not os.path.exists(model_path) or not os.path.exists(calib_path):
            raise FileNotFoundError("Model files not found. Run training first.")
            
        model = joblib.load(model_path)
        calib_model = joblib.load(calib_path)
        
        # 5. 直前オッズの取得 (出馬表の静的HTMLにはオッズが含まれないため、オッズAPIから取得)
        odds_map = self.source.get_tansho_odds(race_id)
        if odds_map:
            df_race["tansho_odds"] = df_race["horse_num"].map(odds_map)
            print(f"Live odds fetched for {len(odds_map)} horses.")
        else:
            print("Warning: live odds unavailable. Expected value cannot be computed reliably.")

        # 6. 予測
        # 学習時に保存したカテゴリ水準を適用する。
        # 単純な astype("category") では水準集合が学習時と異なり、
        # LightGBMがコード値ズレで黙って誤った予測を返すため必須。
        df_race = self.trainer.apply_categories(df_race)

        X = df_race[self.trainer.features]
        raw_scores = model.predict(X)
        
        # 校正済み勝率。ライブオッズが取れた場合は市場との結合モデルの勝率を期待値計算に使う
        # (結合モデルは確定オッズで推定しているため、発走前のオッズに対しては推定誤差が大きくなりうる)
        win_probs = CalibratedProbPolicy(calib_model).calculate(raw_scores, df_race)
        df_race["model_prob"] = win_probs
        combiner_path = os.path.join(self.models_dir, "market_combiner.model")
        if odds_map and os.path.exists(combiner_path):
            win_probs = MarketCombinedProbPolicy(joblib.load(combiner_path)).calculate(raw_scores, df_race)
            print("Using market-combined win probabilities.")

        df_race["raw_score"] = raw_scores
        df_race["win_prob"] = win_probs
        
        # 期待値計算とベット判定
        bet_policy = ExpectedValueBet(
            min_expected_value=self.min_expected_value,
            odds_margin=self.odds_margin
        )
        
        decisions = bet_policy.decide_bets(df_race, win_probs)
        df_race["should_bet"] = decisions
        df_race["expected_value"] = win_probs * (df_race["tansho_odds"] * self.odds_margin)
        if budget is not None:
            # 予算を資金とみなした分数ケリーの単勝購入金額(100円単位, 合計は予算以下)
            df_race["stake"] = kelly_stakes(win_probs, df_race["tansho_odds"].values, budget,
                                            self.kelly_fraction, self.odds_margin, self.min_expected_value)
        
        # 結果の出力
        output_cols = [
            "race_id", "race_name", "bracket_num", "horse_num", 
            "horse_id", "tansho_odds", "model_prob", "win_prob", "expected_value", "should_bet"
        ] + (["stake"] if budget is not None else [])
        
        output_dir = os.path.dirname(self.builder.features_dir)
        output_path = os.path.join(output_dir, f"prediction_{race_id}.csv")
        
        df_output = df_race[output_cols].sort_values("horse_num")
        df_output.to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"Prediction result saved to {output_path}")
        
        # 購入対象の表示
        bet_targets = df_output[df_output["should_bet"]]
        if not bet_targets.empty:
            print("\n--- BET TARGETS ---")
            print(bet_targets[["horse_num", "tansho_odds", "win_prob", "expected_value"]])
        else:
            print("\nNo bet targets found (no horses with expected value > threshold).")

        if budget is not None:
            df_stake = df_output[df_output["stake"] > 0]
            print(f"\n--- 単勝購入案 (予算 {budget:,.0f}円, {self.kelly_fraction}ケリー) ---")
            if df_stake.empty:
                print("購入なし (100円未満に切り捨てられた、または期待値が閾値未満)")
            else:
                print(df_stake[["horse_num", "tansho_odds", "win_prob", "expected_value", "stake"]])
                print(f"合計: {df_stake['stake'].sum():,.0f}円")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        pred = Predictor()
        pred.predict_race(sys.argv[1])
    else:
        print("Please provide race_id as argument.")
