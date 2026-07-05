import os
import pandas as pd
import numpy as np
import yaml
import sqlite3
import joblib
import lightgbm as lgb
from sklearn.isotonic import IsotonicRegression
from src.preprocessing.features import FeatureBuilder
from src.policies.score_policy import CalibratedProbPolicy
from src.policies.bet_policy import ExpectedValueBet
from src.training.train import Trainer


def evaluate_bets(df_test: pd.DataFrame, bet_col: str, df_returns: pd.DataFrame, label: str) -> dict:
    """購入判定結果(bet_col列がTrueの行)について、単勝・複勝それぞれの
    的中率・回収率を払戻テーブル(df_returns)と突合して計算する。
    (仕様書§10: シミュレーション/払戻計算の単体テスト対象)"""
    df_bets = df_test[df_test[bet_col]].copy()
    n_bets = len(df_bets)
    if n_bets == 0:
        return {
            f"n_bets_{label}": 0,
            f"hit_rate_{label}": 0.0,
            f"recovery_rate_{label}": 0.0,
            f"hit_rate_place_{label}": 0.0,
            f"recovery_rate_place_{label}": 0.0,
        }

    # 単勝の払戻金を取得してマージ
    df_tansho = df_bets.merge(
        df_returns[df_returns["ticket_type"] == "単勝"][["race_id", "horse_num", "payoff"]],
        on=["race_id", "horse_num"],
        how="left"
    )
    df_tansho["payoff"] = df_tansho["payoff"].fillna(0)

    total_bet = n_bets * 100
    total_return = df_tansho["payoff"].sum()
    recovery_rate = (total_return / total_bet) * 100 if total_bet > 0 else 0.0
    hits = (df_tansho["payoff"] > 0).sum()
    hit_rate = (hits / n_bets) * 100 if n_bets > 0 else 0.0

    # 複勝の払戻金(同じ購入判定馬について、複勝で購入した場合の回収率)
    df_fukusho = df_bets.merge(
        df_returns[df_returns["ticket_type"] == "複勝"][["race_id", "horse_num", "payoff"]],
        on=["race_id", "horse_num"],
        how="left"
    )
    df_fukusho["payoff"] = df_fukusho["payoff"].fillna(0)
    total_return_place = df_fukusho["payoff"].sum()
    recovery_rate_place = (total_return_place / total_bet) * 100 if total_bet > 0 else 0.0
    hits_place = (df_fukusho["payoff"] > 0).sum()
    hit_rate_place = (hits_place / n_bets) * 100 if n_bets > 0 else 0.0

    print(f"[{label}] Bets: {n_bets}, Hits: {hits}, Hit Rate: {hit_rate:.2f}%, "
          f"Recovery Rate(単勝): {recovery_rate:.2f}%, Recovery Rate(複勝): {recovery_rate_place:.2f}%")

    return {
        f"n_bets_{label}": n_bets,
        f"hit_rate_{label}": hit_rate,
        f"recovery_rate_{label}": recovery_rate,
        f"hit_rate_place_{label}": hit_rate_place,
        f"recovery_rate_place_{label}": recovery_rate_place,
    }


class Simulator:
    def __init__(self, config_path="config/config.yaml"):
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)
            
        base_dir = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        self.db_path = os.path.join(base_dir, self.config["paths"]["db_path"])
        self.builder = FeatureBuilder(config_path)
        self.trainer = Trainer(config_path)
        self.odds_margin = self.config["betting"]["odds_margin"]
        self.min_expected_value = self.config["betting"]["min_expected_value"]

    def load_returns(self):
        conn = sqlite3.connect(self.db_path)
        df_returns = pd.read_sql_query("SELECT * FROM returns", conn)
        conn.close()
        return df_returns

    def run_walk_forward(self):
        # 特徴量のビルドとロード
        df_all = self.builder.build_features()

        # 結果未確定行(出馬表由来)を除外する。バックテストは確定済みレースのみを対象とする。
        if "is_unresolved" in df_all.columns:
            df_all = df_all[~df_all["is_unresolved"]].reset_index(drop=True)

        # 払戻情報のロード
        df_returns = self.load_returns()

        df_all["date"] = pd.to_datetime(df_all["date"])
        df_all = df_all.sort_values("date").reset_index(drop=True)

        min_date = df_all["date"].min()
        max_date = df_all["date"].max()
        
        if pd.isna(min_date) or pd.isna(max_date):
            print("No data available for simulation.")
            return
            
        print(f"Data range: {min_date.date()} to {max_date.date()}")
        
        # 2021-01-01 以降から半年ごとにテストウィンドウを設定
        test_start_dates = pd.date_range(start="2021-01-01", end=max_date, freq="6MS")
        
        results = []
        
        for i, start_date in enumerate(test_start_dates):
            end_date = start_date + pd.DateOffset(months=6)
            if start_date >= max_date:
                break
                
            # テストデータ抽出
            df_test = df_all[(df_all["date"] >= start_date) & (df_all["date"] < end_date)].copy()
            if df_test.empty:
                continue
                
            # 学習データ・検証データ抽出 (テスト開始前1年を検証用とする)
            val_start = start_date - pd.DateOffset(years=1)
            df_train = df_all[df_all["date"] < val_start].copy()
            df_val = df_all[(df_all["date"] >= val_start) & (df_all["date"] < start_date)].copy()
            
            if df_train.empty or df_val.empty:
                continue
                
            print(f"\n--- Window {i+1}: Test period {start_date.date()} to {end_date.date()} ---")
            print(f"Train size: {len(df_train)}, Val size: {len(df_val)}, Test size: {len(df_test)}")

            # early stoppingとキャリブレーションを同一データで行うと校正が楽観的になるため、
            # 検証データを時系列で前半(early stopping用)/後半(キャリブレーション用)に分割する。
            df_val_stop, df_val_calib = self.trainer.split_validation(df_val)
            if df_val_stop.empty or df_val_calib.empty:
                continue

            # データのソートと型変換 (LambdaRank要件)
            df_train_sorted = df_train.sort_values("race_id").reset_index(drop=True)
            df_val_stop_sorted = df_val_stop.sort_values("race_id").reset_index(drop=True)

            for col in self.trainer.cat_cols:
                df_train_sorted[col] = df_train_sorted[col].astype("category")
                df_val_stop_sorted[col] = df_val_stop_sorted[col].astype("category")
                df_val_calib[col] = df_val_calib[col].astype("category")
                df_test[col] = df_test[col].astype("category")

            group_train = df_train_sorted.groupby("race_id").size().values
            group_val_stop = df_val_stop_sorted.groupby("race_id").size().values

            X_train = df_train_sorted[self.trainer.features]
            y_train_rank = df_train_sorted["rank_label"]
            y_train_top3 = df_train_sorted["is_top3"]
            X_val_stop = df_val_stop_sorted[self.trainer.features]
            y_val_stop_rank = df_val_stop_sorted["rank_label"]
            y_val_stop_top3 = df_val_stop_sorted["is_top3"]

            train_data_rank = lgb.Dataset(X_train, label=y_train_rank, group=group_train)
            val_data_rank = lgb.Dataset(X_val_stop, label=y_val_stop_rank, group=group_val_stop, reference=train_data_rank)

            # LambdaRank(主モデル)学習
            model = lgb.train(
                self.config["model"]["lgb_params"],
                train_data_rank,
                valid_sets=[val_data_rank],
                num_boost_round=500,
                callbacks=[lgb.early_stopping(30, verbose=False)]
            )

            # 二値分類ベースラインモデル学習 (仕様書§6.2: 主モデルとの比較用)
            train_data_binary = lgb.Dataset(X_train, label=y_train_top3)
            val_data_binary = lgb.Dataset(X_val_stop, label=y_val_stop_top3, reference=train_data_binary)
            binary_params = {
                "objective": "binary",
                "metric": "binary_logloss",
                "boosting_type": "gbdt",
                "learning_rate": 0.05,
                "num_leaves": 31,
                "min_data_in_leaf": 20,
                "random_state": 42,
                "verbose": -1
            }
            binary_model = lgb.train(
                binary_params,
                train_data_binary,
                valid_sets=[val_data_binary],
                num_boost_round=500,
                callbacks=[lgb.early_stopping(30, verbose=False)]
            )

            # キャリブレーション学習 (early stoppingに使っていない後半データのみを使用)
            raw_scores_calib = model.predict(df_val_calib[self.trainer.features])
            is_winner_calib = (df_val_calib["rank_num"] == 1).astype(int)
            ir = IsotonicRegression(out_of_bounds="clip")
            ir.fit(raw_scores_calib, is_winner_calib)

            raw_scores_calib_binary = binary_model.predict(df_val_calib[self.trainer.features])
            ir_binary = IsotonicRegression(out_of_bounds="clip")
            ir_binary.fit(raw_scores_calib_binary, is_winner_calib)

            # 予測と期待値ベット判定 (主モデル: LambdaRank)
            raw_scores_test = model.predict(df_test[self.trainer.features])
            prob_policy = CalibratedProbPolicy(ir)
            win_probs = prob_policy.calculate(raw_scores_test, df_test)

            df_test["win_prob"] = win_probs

            bet_policy = ExpectedValueBet(
                min_expected_value=self.min_expected_value,
                odds_margin=self.odds_margin
            )
            decisions = bet_policy.decide_bets(df_test, win_probs)
            df_test["should_bet"] = decisions

            # ベースラインモデル(二値分類)の予測と期待値ベット判定
            raw_scores_test_binary = binary_model.predict(df_test[self.trainer.features])
            prob_policy_binary = CalibratedProbPolicy(ir_binary)
            win_probs_binary = prob_policy_binary.calculate(raw_scores_test_binary, df_test)
            decisions_binary = bet_policy.decide_bets(df_test, win_probs_binary)
            df_test["should_bet_binary"] = decisions_binary

            window_result = {
                "window": i + 1,
                "period": f"{start_date.date()}~{end_date.date()}",
            }
            window_result.update(evaluate_bets(df_test, "should_bet", df_returns, "lambdarank"))
            window_result.update(evaluate_bets(df_test, "should_bet_binary", df_returns, "binary"))
            results.append(window_result)

        if results:
            df_res = pd.DataFrame(results)
            print("\n========================================")
            print("WALK-FORWARD VALIDATION SUMMARY")
            print("========================================")
            print(df_res.to_string(index=False))
            print("----------------------------------------")
            for label in ["lambdarank", "binary"]:
                print(f"[{label}] Mean Recovery Rate(単勝): {df_res[f'recovery_rate_{label}'].mean():.2f}% "
                      f"(Std: {df_res[f'recovery_rate_{label}'].std():.2f}%)")
                print(f"[{label}] Mean Recovery Rate(複勝): {df_res[f'recovery_rate_place_{label}'].mean():.2f}% "
                      f"(Std: {df_res[f'recovery_rate_place_{label}'].std():.2f}%)")
                print(f"[{label}] Mean Hit Rate: {df_res[f'hit_rate_{label}'].mean():.2f}%")
                print(f"[{label}] Total Bets: {df_res[f'n_bets_{label}'].sum()}")
            print("========================================")
            
            report_path = os.path.join(self.trainer.models_dir, "simulation_report.json")
            df_res.to_json(report_path, orient="records", indent=2)
            print(f"Report saved to {report_path}")
        else:
            print("No validation windows could be run due to insufficient data.")

if __name__ == "__main__":
    sim = Simulator()
    sim.run_walk_forward()
