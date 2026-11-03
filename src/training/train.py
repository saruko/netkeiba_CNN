import os
import glob
import json
from datetime import datetime, timezone
import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
import yaml
import joblib

class Trainer:
    def __init__(self, config_path="config/config.yaml"):
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)
            
        base_dir = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        self.features_dir = os.path.join(base_dir, self.config["paths"]["features_dir"])
        self.models_dir = os.path.join(base_dir, self.config["paths"]["models_dir"])
        os.makedirs(self.models_dir, exist_ok=True)
        
        self.train_years = self.config["model"]["target_years_train"]
        self.val_year = self.config["model"]["target_year_val"]
        self.test_year = self.config["model"]["target_year_test"]
        
        # 特徴量の定義
        self.cat_cols = [
            "gender", "course_type", "direction", "weather",
            "surface_state", "race_class", "place", "run_style",
            "run_style_place", "distance_band",
            # 血統は仕様書§5.1の通り父・母父のみ。母・祖母などのIDはほぼ馬ごとに固有で、過学習の原因になるため使わない
            "father_id", "father_of_mother_id"
        ]

        self.num_cols = [
            "bracket_num", "horse_num", "age", "weight_carrier",
            "horse_weight", "weight_diff", "n_horses", "distance",
            "jockey_top3_rate", "trainer_top3_rate",
            "sire_top3_rate_course", "sire_top3_rate_distance",
            "rank_mean_3", "rank_mean_5", "rank_mean_all",
            "rank_ratio_mean_3", "rank_ratio_mean_5", "rank_ratio_mean_all",
            "rank_ratio_mean_by_distance_band", "rank_ratio_mean_by_course_type", "rank_ratio_mean_by_place",
            "class_level_mean_5", "class_diff",
            "prize_sum_3", "prize_sum_5", "prize_sum_all",
            "margin_mean_3", "margin_mean_5",
            "last_3f_mean_3", "last_3f_mean_5",
            "time_diff_mean_3", "time_diff_mean_5",
            "speed_fig_mean_3", "speed_fig_mean_5", "speed_fig_max_5",
            "interval_days"
        ]
        
        self.features = self.cat_cols + self.num_cols

    def load_dataset(self, years):
        dfs = []
        for year in years:
            pattern = os.path.join(self.features_dir, f"year={year}", "*.parquet")
            files = glob.glob(pattern)
            for file in files:
                dfs.append(pd.read_parquet(file))
        if not dfs:
            raise FileNotFoundError(f"No parquet files found in features_dir for years: {years}")
        df = pd.concat(dfs, ignore_index=True)

        # 結果未確定行(出馬表由来。is_unresolved=True)を学習・評価対象から除外する。
        # これらは「全馬凡走」の負例(is_top3=0, rank_label=0)ではなく、単にまだ
        # レースが行われていないだけであり、混入すると学習データを汚染する。
        if "is_unresolved" in df.columns:
            df = df[~df["is_unresolved"]].reset_index(drop=True)

        # カテゴリ変数の明示的型変換
        for col in self.cat_cols:
            if col in df.columns:
                df[col] = df[col].astype("category")

        return df

    def split_validation(self, df_val):
        """検証データ(val_year)を開催日単位で時系列に前半/後半に分割する。
        前半をearly stopping用、後半をキャリブレーション用として分離することで、
        「early stoppingとキャリブレーションを同一データで行うと校正が楽観的になる」
        問題(仕様書§6.3)を避ける。ランダム分割ではなく時系列分割とする(仕様書§7.1準拠)。
        開催日が2日以上あれば必ず両側が非空になる(行数の中央値ベースだと
        日数が少ない場合に後半が空になりうるため、日付単位で分割する)。"""
        unique_dates = sorted(pd.to_datetime(df_val["date"]).unique())
        if len(unique_dates) < 2:
            raise ValueError(
                "Validation data must span at least 2 distinct race dates to separate "
                f"early-stopping data from calibration data (found {len(unique_dates)})."
            )
        n_stop_dates = (len(unique_dates) + 1) // 2
        cutoff = unique_dates[n_stop_dates - 1]
        dates = pd.to_datetime(df_val["date"])
        df_val_stop = df_val[dates <= cutoff]
        df_val_calib = df_val[dates > cutoff]
        return df_val_stop.reset_index(drop=True), df_val_calib.reset_index(drop=True)

    def train_binary(self, df_train, df_val):
        """二値分類モデルの学習（3着以内に入るか）"""
        print("Training Binary Classification Model...")
        
        X_train = df_train[self.features]
        y_train = df_train["is_top3"]
        X_val = df_val[self.features]
        y_val = df_val["is_top3"]
        
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "boosting_type": "gbdt",
            "learning_rate": 0.05,
            "num_leaves": 31,
            "min_data_in_leaf": 20,
            "random_state": 42,
            "verbose": -1
        }
        
        model = lgb.train(
            params,
            train_data,
            valid_sets=[train_data, val_data],
            num_boost_round=1000,
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(50)]
        )
        
        model_path = os.path.join(self.models_dir, "lgb_binary.model")
        joblib.dump(model, model_path)
        print(f"Binary model saved to {model_path}")
        return model

    def train_lambdarank(self, df_train, df_val, tune_hparam=False):
        """LambdaRankモデルの学習"""
        print("Training LambdaRank Model...")
        
        # LambdaRankでは同じレース内でグループ化し、かつソートされている必要がある
        df_train_sorted = df_train.sort_values(by="race_id").reset_index(drop=True)
        df_val_sorted = df_val.sort_values(by="race_id").reset_index(drop=True)
        
        # カテゴリ変数の明示的型変換（ソート等で型が崩れるのを防ぐ）
        for col in self.cat_cols:
            df_train_sorted[col] = df_train_sorted[col].astype("category")
            df_val_sorted[col] = df_val_sorted[col].astype("category")
            
        group_train = df_train_sorted.groupby("race_id").size().values
        group_val = df_val_sorted.groupby("race_id").size().values
        
        X_train = df_train_sorted[self.features]
        y_train = df_train_sorted["rank_label"]
        X_val = df_val_sorted[self.features]
        y_val = df_val_sorted["rank_label"]
        
        train_data = lgb.Dataset(X_train, label=y_train, group=group_train)
        val_data = lgb.Dataset(X_val, label=y_val, group=group_val, reference=train_data)
        
        params = self.config["model"]["lgb_params"]
        
        if tune_hparam:
            # Optunaを使用したハイパーパラメータ探索
            def objective(trial):
                trial_params = {
                    "objective": "lambdarank",
                    "metric": "ndcg",
                    "ndcg_eval_at": [1, 3],
                    "boosting_type": "gbdt",
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1),
                    "num_leaves": trial.suggest_int("num_leaves", 15, 63),
                    "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 10, 50),
                    "random_state": 42,
                    "verbose": -1
                }
                
                trial_model = lgb.train(
                    trial_params,
                    train_data,
                    valid_sets=[val_data],
                    num_boost_round=500,
                    callbacks=[lgb.early_stopping(30, verbose=False)]
                )
                
                # 検証データ上でのNDCGスコアを取得
                val_pred = trial_model.predict(X_val)
                # Optunaは検証メトリクスを返す
                # ここでは最もシンプルな最適化指標として検証データの平均NDCGを使用
                # validation_sets の評価結果は trial_model.best_score['valid_0']['ndcg@1'] などで取得可能
                ndcg_score = trial_model.best_score['valid_0']['ndcg@1']
                return ndcg_score

            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=10)
            print(f"Best trial params: {study.best_params}")
            params.update(study.best_params)
            
        model = lgb.train(
            params,
            train_data,
            valid_sets=[train_data, val_data],
            num_boost_round=1000,
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(50)]
        )
        
        model_path = os.path.join(self.models_dir, "lgb_lambdarank.model")
        joblib.dump(model, model_path)
        print(f"LambdaRank model saved to {model_path}")
        return model

    def save_categories(self, df_train):
        """学習時のカテゴリ変数の水準(カテゴリ→コード対応)を保存する。
        LightGBMはpandasカテゴリのコード値で学習するため、予測時に水準集合が
        異なると黙って誤った予測になる。予測側はこのファイルの水準を適用すること。"""
        categories = {}
        for col in self.cat_cols:
            if col in df_train.columns:
                categories[col] = [str(c) for c in df_train[col].astype("category").cat.categories]
        path = os.path.join(self.models_dir, "categories.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(categories, f, ensure_ascii=False, indent=2)
        return path

    def apply_categories(self, df):
        """保存済みのカテゴリ水準をdfに適用する(予測時用)。未知の水準はNaN(欠損)になる。"""
        path = os.path.join(self.models_dir, "categories.json")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"categories.json not found at {path}. Train the model first so that "
                "category levels are saved for consistent prediction."
            )
        with open(path, "r", encoding="utf-8") as f:
            categories = json.load(f)
        for col, cats in categories.items():
            if col in df.columns:
                df[col] = pd.Categorical(df[col].astype(str), categories=cats)
        return df

    def run(self, tune_hparam=False):
        df_train = self.load_dataset(self.train_years)
        df_val = self.load_dataset([self.val_year])

        # 学習データのカテゴリ水準を確定・保存し、検証データにも同じ水準を適用する。
        # (別々にロードしたデータ間で水準集合が異なると、LightGBMのコード値がズレて
        #  検証メトリクスやearly stoppingが黙って狂う)
        self.save_categories(df_train)
        df_val = self.apply_categories(df_val)

        df_val_stop, _ = self.split_validation(df_val)

        binary_model = self.train_binary(df_train, df_val_stop)
        lambdarank_model = self.train_lambdarank(df_train, df_val_stop, tune_hparam=tune_hparam)

        self._save_experiment_log(binary_model, lambdarank_model)

    def _save_experiment_log(self, binary_model, lambdarank_model):
        """仕様書§11: 実験ごとにconfig + 学習期間 + メトリクスをJSONで保存し再現可能にする"""
        log = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "config": self.config,
            "train_years": self.train_years,
            "val_year": self.val_year,
            "test_year": self.test_year,
            "features": self.features,
            "metrics": {
                "binary": dict(binary_model.best_score) if binary_model.best_score else {},
                "lambdarank": dict(lambdarank_model.best_score) if lambdarank_model.best_score else {},
            },
        }
        # defaultdictのネストをJSON化可能なplain dictに変換
        for model_key in log["metrics"]:
            log["metrics"][model_key] = {
                valid_name: dict(metric_dict) for valid_name, metric_dict in log["metrics"][model_key].items()
            }

        timestamp_slug = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        log_path = os.path.join(self.models_dir, f"experiment_{timestamp_slug}.json")
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(log, f, ensure_ascii=False, indent=2, default=str)
        print(f"Experiment log saved to {log_path}")

if __name__ == "__main__":
    trainer = Trainer()
    trainer.run()
