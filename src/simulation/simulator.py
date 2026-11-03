import os
import glob
import json
import pandas as pd
import numpy as np
import yaml
import sqlite3
import joblib
import lightgbm as lgb
from src.training.calibration import SoftmaxCalibrator, MarketCombiner
from src.policies.score_policy import CalibratedProbPolicy, MarketCombinedProbPolicy, market_implied_probs
from src.policies.bet_policy import ExpectedValueBet, kelly_stakes
from src.training.train import Trainer


def bet_payoffs(df_bets: pd.DataFrame, df_returns: pd.DataFrame, ticket_type: str) -> pd.DataFrame:
    """購入馬(1行=100円1点)に指定券種の払戻金を突合する。不的中は payoff=0"""
    df = df_bets[["race_id", "horse_num"]].merge(
        df_returns[df_returns["ticket_type"] == ticket_type][["race_id", "horse_num", "payoff"]],
        on=["race_id", "horse_num"],
        how="left"
    )
    df["payoff"] = df["payoff"].fillna(0)
    return df


def bootstrap_recovery_ci(df_payoff: pd.DataFrame, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0) -> tuple:
    """回収率(%)の信頼区間をレース単位の復元抽出(ブートストラップ)で求める。
    同一レース内の購入は結果が相関するため、馬単位ではなくレース単位で抽出する。"""
    per_race = df_payoff.groupby("race_id")["payoff"].agg(["sum", "count"])
    sums = per_race["sum"].values
    counts = per_race["count"].values
    rng = np.random.default_rng(seed)
    rates = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, len(per_race), size=len(per_race))
        rates[b] = sums[idx].sum() / (counts[idx].sum() * 100) * 100
    lower, upper = np.percentile(rates, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lower), float(upper)


def simulate_bankroll(df: pd.DataFrame, prob_col: str, df_returns: pd.DataFrame, initial_bankroll: float,
                      kelly_fraction: float, odds_margin: float, min_expected_value: float) -> tuple:
    """初期資金から、各レースの単勝を分数ケリーで購入し続けたときの資金推移を求める。
    df は race_id, horse_num, date, tansho_odds, prob_col 列を持つこと。
    レースの発走時刻を持たないため、同じ日のレースはすべてその日の開始時点の資金で同時に購入したとみなし、
    日の終わりに精算する。同日の購入合計が資金を超える場合は全購入を比例縮小する。
    返り値: (集計dict, 日次の資金推移DataFrame)"""
    tansho = df_returns[df_returns["ticket_type"] == "単勝"][["race_id", "horse_num", "payoff"]]
    d = df[["date", "race_id", "horse_num", "tansho_odds", prob_col]].merge(
        tansho, on=["race_id", "horse_num"], how="left")
    d["payoff"] = d["payoff"].fillna(0)

    bankroll = peak = float(initial_bankroll)
    max_drawdown = 0.0
    n_bets = 0
    total_stake = total_return = 0.0
    rows = []
    for date, day in d.groupby("date", sort=True):
        stakes = pd.Series(0.0, index=day.index)
        for _, race in day.groupby("race_id"):
            stakes[race.index] = kelly_stakes(race[prob_col].values, race["tansho_odds"].values, bankroll,
                                              kelly_fraction, odds_margin, min_expected_value)
        if stakes.sum() > bankroll:
            stakes = np.floor(stakes * bankroll / stakes.sum() / 100) * 100
        day_stake = stakes.sum()
        day_return = (stakes / 100 * day["payoff"]).sum()
        bankroll += day_return - day_stake
        peak = max(peak, bankroll)
        max_drawdown = max(max_drawdown, 1 - bankroll / peak)
        n_bets += int((stakes > 0).sum())
        total_stake += day_stake
        total_return += day_return
        rows.append({"date": date, "stake": day_stake, "return": day_return, "bankroll": bankroll})

    summary = {
        "initial_bankroll": float(initial_bankroll),
        "final_bankroll": bankroll,
        "max_drawdown": max_drawdown,
        "n_bets": n_bets,
        "total_stake": total_stake,
        "total_return": total_return,
        "recovery_rate": total_return / total_stake * 100 if total_stake > 0 else 0.0,
    }
    return summary, pd.DataFrame(rows)


def race_prob_metrics(df: pd.DataFrame, prob_col: str) -> dict:
    """レース単位の勝率予測の精度。
    log_loss: 各レースの1着馬に与えた確率の -log の平均(多項対数損失)
    brier: 各レースの Σ_i (p_i - 1着フラグ_i)^2 の平均(多クラスBrier)
    どちらも小さいほど良い。1着馬のいないレースは除外する。"""
    d = df[["race_id", prob_col]].assign(y=(df["rank_num"] == 1).astype(float))
    d = d[d.groupby("race_id")["y"].transform("sum") > 0]
    winner_prob = d[d["y"] == 1].groupby("race_id")[prob_col].sum()
    sq_err = ((d[prob_col] - d["y"]) ** 2).groupby(d["race_id"]).sum()
    return {
        "log_loss": float(-np.log(winner_prob.clip(lower=1e-15)).mean()),
        "brier": float(sq_err.mean()),
        "n_races": int(len(winner_prob)),
    }


def bootstrap_logloss_diff_ci(df: pd.DataFrame, prob_col_a: str, prob_col_b: str,
                              n_boot: int = 2000, alpha: float = 0.05, seed: int = 0) -> tuple:
    """レース単位LogLossの差 (a - b) の平均と95%信頼区間を、レース単位の対応のあるブートストラップで求める。
    区間の上限が0未満なら a は b より有意に良い。1着馬のいないレースは除外する。"""
    winners = df[df["rank_num"] == 1]
    per_race = winners.groupby("race_id")[[prob_col_a, prob_col_b]].sum().clip(lower=1e-15)
    diff = (-np.log(per_race[prob_col_a]) + np.log(per_race[prob_col_b])).values
    rng = np.random.default_rng(seed)
    means = np.array([diff[rng.integers(0, len(diff), size=len(diff))].mean() for _ in range(n_boot)])
    lower, upper = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(diff.mean()), float(lower), float(upper)


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

    df_tansho = bet_payoffs(df_bets, df_returns, "単勝")

    total_bet = n_bets * 100
    total_return = df_tansho["payoff"].sum()
    recovery_rate = (total_return / total_bet) * 100 if total_bet > 0 else 0.0
    hits = (df_tansho["payoff"] > 0).sum()
    hit_rate = (hits / n_bets) * 100 if n_bets > 0 else 0.0

    # 複勝の払戻金(同じ購入判定馬について、複勝で購入した場合の回収率)
    df_fukusho = bet_payoffs(df_bets, df_returns, "複勝")
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


# 勝率予測精度を比較する列とラベル / 期待値ベットの評価ラベル
PROB_LABELS = [("win_prob", "lambdarank"), ("win_prob_binary", "binary"),
               ("win_prob_combined", "combined"), ("market_prob", "market"),
               ("market_recal_prob", "market_recal")]
BET_LABELS = ["lambdarank", "binary", "combined"]


class Simulator:
    def __init__(self, config_path="config/config.yaml"):
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)
            
        base_dir = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        self.db_path = os.path.join(base_dir, self.config["paths"]["db_path"])
        self.trainer = Trainer(config_path)
        self.odds_margin = self.config["betting"]["odds_margin"]
        self.min_expected_value = self.config["betting"]["min_expected_value"]
        self.test_start = self.config["simulation"]["test_start"]
        self.initial_bankroll = self.config["simulation"]["initial_bankroll"]
        self.kelly_fraction = self.config["betting"]["kelly_fraction"]

    def load_returns(self):
        conn = sqlite3.connect(self.db_path)
        df_returns = pd.read_sql_query("SELECT * FROM returns", conn)
        conn.close()
        return df_returns

    def run_walk_forward(self):
        # build-features で生成済みの特徴量Parquetを全年分ロードする。
        # (load_dataset は結果未確定行(出馬表由来)を除外するため、確定済みレースのみが対象になる)
        year_dirs = glob.glob(os.path.join(self.trainer.features_dir, "year=*"))
        years = sorted(int(os.path.basename(d).split("=")[1]) for d in year_dirs)
        df_all = self.trainer.load_dataset(years)

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
        
        # config の simulation.test_start 以降、半年ごとにテストウィンドウを設定
        test_start_dates = pd.date_range(start=self.test_start, end=max_date, freq="6MS")
        
        results = []
        test_frames = []
        
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
            # 二値分類モデルは確率ではなく対数オッズ(raw_score=True)をスコアとして校正する
            raw_scores_calib = model.predict(df_val_calib[self.trainer.features])
            is_winner_calib = df_val_calib["rank_num"] == 1
            calib = SoftmaxCalibrator().fit(raw_scores_calib, df_val_calib["race_id"], is_winner_calib)

            raw_scores_calib_binary = binary_model.predict(df_val_calib[self.trainer.features], raw_score=True)
            calib_binary = SoftmaxCalibrator().fit(raw_scores_calib_binary, df_val_calib["race_id"], is_winner_calib)

            # 市場との結合モデル: p ∝ exp(α·s + β·log q)。校正データの確定オッズで推定する
            combiner = MarketCombiner().fit(raw_scores_calib, market_implied_probs(df_val_calib),
                                            df_val_calib["race_id"], is_winner_calib)
            # 比較用: 市場確率だけを再校正したもの(α=0, p ∝ q^β)。本命・大穴バイアスの補正分を
            # モデルの寄与と取り違えないよう、結合モデルはこれと比較する
            log_q_calib = np.log(np.clip(market_implied_probs(df_val_calib), MarketCombiner.MIN_MARKET_PROB, None))
            market_recal = SoftmaxCalibrator().fit(log_q_calib, df_val_calib["race_id"], is_winner_calib)

            # 予測と期待値ベット判定 (主モデル: LambdaRank)
            raw_scores_test = model.predict(df_test[self.trainer.features])
            prob_policy = CalibratedProbPolicy(calib)
            win_probs = prob_policy.calculate(raw_scores_test, df_test)

            df_test["win_prob"] = win_probs

            bet_policy = ExpectedValueBet(
                min_expected_value=self.min_expected_value,
                odds_margin=self.odds_margin
            )
            decisions = bet_policy.decide_bets(df_test, win_probs)
            df_test["should_bet"] = decisions

            # ベースラインモデル(二値分類)の予測と期待値ベット判定
            raw_scores_test_binary = binary_model.predict(df_test[self.trainer.features], raw_score=True)
            prob_policy_binary = CalibratedProbPolicy(calib_binary)
            win_probs_binary = prob_policy_binary.calculate(raw_scores_test_binary, df_test)
            df_test["win_prob_binary"] = win_probs_binary
            decisions_binary = bet_policy.decide_bets(df_test, win_probs_binary)
            df_test["should_bet_binary"] = decisions_binary

            # 市場(単勝オッズ)の暗黙勝率。モデルの勝率がこれより高精度でなければエッジの根拠はない
            df_test["market_prob"] = market_implied_probs(df_test)
            log_q_test = np.log(np.clip(df_test["market_prob"].values, MarketCombiner.MIN_MARKET_PROB, None))
            df_test["market_recal_prob"] = market_recal.predict_proba(log_q_test, df_test["race_id"].values)

            # 結合モデル(LambdaRankスコア + 市場)の勝率と期待値ベット判定
            win_probs_combined = MarketCombinedProbPolicy(combiner).calculate(raw_scores_test, df_test)
            df_test["win_prob_combined"] = win_probs_combined
            df_test["should_bet_combined"] = bet_policy.decide_bets(df_test, win_probs_combined)

            window_result = {
                "window": i + 1,
                "period": f"{start_date.date()}~{end_date.date()}",
                "combiner_alpha": combiner.alpha,
                "combiner_alpha_se": combiner.alpha_se,
                "combiner_beta": combiner.beta,
            }
            for prob_col, label in PROB_LABELS:
                m = race_prob_metrics(df_test, prob_col)
                window_result[f"log_loss_{label}"] = m["log_loss"]
                window_result[f"brier_{label}"] = m["brier"]
            window_result.update(evaluate_bets(df_test, "should_bet", df_returns, "lambdarank"))
            window_result.update(evaluate_bets(df_test, "should_bet_binary", df_returns, "binary"))
            window_result.update(evaluate_bets(df_test, "should_bet_combined", df_returns, "combined"))
            results.append(window_result)
            test_frames.append(df_test[["race_id", "horse_num", "date", "tansho_odds", "rank_num",
                                        "win_prob", "win_prob_binary",
                                        "win_prob_combined", "market_prob", "market_recal_prob",
                                        "should_bet", "should_bet_binary", "should_bet_combined"]])

        if results:
            df_res = pd.DataFrame(results)
            print("\n========================================")
            print("WALK-FORWARD VALIDATION SUMMARY")
            print("========================================")
            print(df_res.to_string(index=False))
            print("----------------------------------------")
            for label in BET_LABELS:
                print(f"[{label}] Mean Recovery Rate(単勝): {df_res[f'recovery_rate_{label}'].mean():.2f}% "
                      f"(Std: {df_res[f'recovery_rate_{label}'].std():.2f}%)")
                print(f"[{label}] Mean Recovery Rate(複勝): {df_res[f'recovery_rate_place_{label}'].mean():.2f}% "
                      f"(Std: {df_res[f'recovery_rate_place_{label}'].std():.2f}%)")
                print(f"[{label}] Mean Hit Rate: {df_res[f'hit_rate_{label}'].mean():.2f}%")
                print(f"[{label}] Total Bets: {df_res[f'n_bets_{label}'].sum()}")
            print("----------------------------------------")

            # 全ウィンドウを通算した評価。回収率はレース単位ブートストラップの95%信頼区間を付ける
            df_pooled = pd.concat(test_frames, ignore_index=True)
            pooled = {}
            print("POOLED (all test windows)")
            for prob_col, label in PROB_LABELS:
                m = race_prob_metrics(df_pooled, prob_col)
                pooled[f"log_loss_{label}"] = m["log_loss"]
                pooled[f"brier_{label}"] = m["brier"]
                print(f"[{label}] Race LogLoss: {m['log_loss']:.4f}, Brier: {m['brier']:.4f} (races: {m['n_races']})")
            print("  * LogLoss/Brier は小さいほど良い。モデルが market を下回らない限り、市場より正確という根拠はない")
            print("  * market_recal は市場確率のみを再校正したもの(本命・大穴バイアスの補正)。"
                  "combined と market_recal の差がモデルの寄与")
            for col_a, col_b, name in [("win_prob_combined", "market_recal_prob", "combined - market_recal"),
                                       ("win_prob_combined", "market_prob", "combined - market"),
                                       ("win_prob", "market_prob", "lambdarank - market")]:
                mean, lower, upper = bootstrap_logloss_diff_ci(df_pooled, col_a, col_b)
                pooled[f"log_loss_diff[{name}]"] = [mean, lower, upper]
                print(f"LogLoss差 {name}: {mean:+.4f} (95% CI: {lower:+.4f} - {upper:+.4f})")
            print("  * 差の信頼区間の上限が0未満のときだけ、左側が右側より正確と判断できる")
            alphas = df_res["combiner_alpha"]
            print(f"[combined] alpha(モデルの重み) 各期間: {', '.join(f'{a:.3f}' for a in alphas)} "
                  f"/ alpha/SE: {', '.join(f'{z:.1f}' for z in alphas / df_res['combiner_alpha_se'])}")
            print("  * alpha/SE がおおむね2未満なら、モデルが市場に無い情報を持つという根拠は弱い")
            for bet_col, label in [("should_bet", "lambdarank"), ("should_bet_binary", "binary"),
                                   ("should_bet_combined", "combined")]:
                df_bets = df_pooled[df_pooled[bet_col]]
                pooled[f"n_bets_{label}"] = int(len(df_bets))
                for ticket_type, suffix in [("単勝", ""), ("複勝", "_place")]:
                    if df_bets.empty:
                        continue
                    df_pay = bet_payoffs(df_bets, df_returns, ticket_type)
                    rate = df_pay["payoff"].sum() / (len(df_pay) * 100) * 100
                    lower, upper = bootstrap_recovery_ci(df_pay)
                    pooled[f"recovery_rate{suffix}_{label}"] = rate
                    pooled[f"recovery_rate{suffix}_{label}_ci95"] = [lower, upper]
                    print(f"[{label}] Recovery Rate({ticket_type}): {rate:.2f}% "
                          f"(95% CI: {lower:.2f}% - {upper:.2f}%, bets: {len(df_pay)})")
            print("----------------------------------------")
            print(f"BANKROLL (初期資金 {self.initial_bankroll:,.0f}円, 単勝, {self.kelly_fraction}ケリー, "
                  f"期待値 >= {self.min_expected_value} の馬のみ)")
            for prob_col, label in [("win_prob", "lambdarank"), ("win_prob_binary", "binary"),
                                    ("win_prob_combined", "combined")]:
                summary, curve = simulate_bankroll(df_pooled, prob_col, df_returns, self.initial_bankroll,
                                                   self.kelly_fraction, self.odds_margin, self.min_expected_value)
                pooled[f"bankroll_{label}"] = summary
                curve.to_csv(os.path.join(self.trainer.models_dir, f"bankroll_{label}.csv"),
                             index=False, encoding="utf-8-sig")
                print(f"[{label}] 最終資金: {summary['final_bankroll']:,.0f}円, "
                      f"最大ドローダウン: {summary['max_drawdown'] * 100:.1f}%, 購入: {summary['n_bets']}件, "
                      f"購入総額: {summary['total_stake']:,.0f}円, 回収率: {summary['recovery_rate']:.2f}%")
            print("  * 判定に使う tansho_odds は確定オッズであり、購入時点のオッズより有利な情報を含む")
            print(f"  * 日次の資金推移は {self.trainer.models_dir}/bankroll_<label>.csv に保存")
            print("========================================")
            
            report_path = os.path.join(self.trainer.models_dir, "simulation_report.json")
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump({"windows": results, "pooled": pooled}, f, ensure_ascii=False, indent=2, default=float)
            print(f"Report saved to {report_path}")
        else:
            print("No validation windows could be run due to insufficient data.")

if __name__ == "__main__":
    sim = Simulator()
    sim.run_walk_forward()
