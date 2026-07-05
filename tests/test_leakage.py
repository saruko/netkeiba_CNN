import os
import glob
import pandas as pd
import pytest
import sqlite3
import yaml

def test_feature_leakage():
    # 設定ファイルとパスのロード
    config_path = "config/config.yaml"
    if not os.path.exists(config_path):
        pytest.skip("config.yaml not found.")
        
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    features_dir = config["paths"]["features_dir"]
    db_path = config["paths"]["db_path"]
    
    if not os.path.exists(db_path):
        pytest.skip("db.sqlite not found.")
        
    parquet_files = glob.glob(os.path.join(features_dir, "year=*", "*.parquet"))
    if not parquet_files:
        pytest.skip("No parquet files found. Build features first.")
        
    df = pd.concat([pd.read_parquet(f) for f in parquet_files], ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    
    # データベースの過去馬成績をロード
    conn = sqlite3.connect(db_path)
    df_hr = pd.read_sql_query("SELECT * FROM horse_results", conn)
    conn.close()
    df_hr["date"] = pd.to_datetime(df_hr["date"])
    
    # サンプル行を抽出して検証
    sample_size = min(100, len(df))
    if sample_size == 0:
        pytest.skip("No data available for testing.")
        
    sample_df = df.sample(sample_size, random_state=42)
    for _, row in sample_df.iterrows():
        horse_id = row["horse_id"]
        race_date = row["date"]
        
        # 過去の馬成績(当日を含まない未来情報を排除した過去走)
        past_races = df_hr[(df_hr["horse_id"] == horse_id) & (df_hr["date"] < race_date)].sort_values("date")
        
        # 過去3走の平均着順の計算検証
        if not past_races.empty:
            recent_3 = past_races.tail(3).copy()
            recent_3["rank_num"] = pd.to_numeric(recent_3["rank"], errors="coerce")
            expected_mean_3 = recent_3["rank_num"].mean()
            
            feat_val = row["rank_mean_3"]
            if not pd.isna(expected_mean_3) and not pd.isna(feat_val):
                assert pytest.approx(feat_val) == expected_mean_3, f"Leakage or calculation error detected for horse {horse_id} on {race_date.date()}"
                
        # 騎手・調教師の複勝率のリーク検証
        # 該当レース日より前の成績のみで計算されていること。
        for col_name, feat_name in [("jockey_id", "jockey_top3_rate"), ("trainer_id", "trainer_top3_rate")]:
            entity_id = row[col_name]
            past_rows = df[(df[col_name] == entity_id) & (df["date"] < race_date) & (~df["rank"].isna())]
            recent_window = past_rows[past_rows["date"] >= race_date - pd.Timedelta(days=365)]

            n_past = len(recent_window)
            feat_val = row[feat_name]
            if n_past >= 5:
                expected_rate = recent_window["is_top3"].sum() / n_past
                assert pytest.approx(feat_val, abs=1e-6) == expected_rate, (
                    f"Leakage or calculation error detected for {col_name}={entity_id} on {race_date.date()}"
                )
            else:
                # 直近365日の出走数が5未満の場合はデフォルト値(0.3)が使われる
                assert pytest.approx(feat_val, abs=1e-6) == 0.3, (
                    f"Default value expected for {col_name}={entity_id} on {race_date.date()} "
                    f"(n_past={n_past})"
                )


def test_split_boundary():
    """仕様書§10: 学習データの最大日付 < テストデータの最小日付 を検証する"""
    config_path = "config/config.yaml"
    if not os.path.exists(config_path):
        pytest.skip("config.yaml not found.")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    features_dir = config["paths"]["features_dir"]
    parquet_files = glob.glob(os.path.join(features_dir, "year=*", "*.parquet"))
    if not parquet_files:
        pytest.skip("No parquet files found. Build features first.")

    df = pd.concat([pd.read_parquet(f) for f in parquet_files], ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])

    train_years = config["model"]["target_years_train"]
    test_year = config["model"]["target_year_test"]

    df_train = df[df["year"].isin(train_years)]
    df_test = df[df["year"] == test_year]

    if df_train.empty or df_test.empty:
        pytest.skip("Train or test data not available for the configured years.")

    assert df_train["date"].max() < df_test["date"].min(), (
        "Train/test split leaks future information: "
        f"train max date {df_train['date'].max()} >= test min date {df_test['date'].min()}"
    )
