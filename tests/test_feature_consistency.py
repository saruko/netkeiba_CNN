import os
import sqlite3
import yaml
import pytest
from src.preprocessing.parser import NetkeibaParser
from src.preprocessing.features import FeatureBuilder

TEST_CONFIG = """
paths:
  data_dir: "{base}"
  html_dir: "{base}/html"
  db_path: "{base}/db.sqlite"
  features_dir: "{base}/features"
  models_dir: "{base}/models"

scraping:
  interval: 0.1
  retry_count: 1
  user_agent: "TestAgent/1.0"
"""


@pytest.fixture
def test_env(tmp_path):
    base = str(tmp_path).replace("\\", "/")
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    config_file = config_dir / "config.yaml"
    config_file.write_text(TEST_CONFIG.format(base=base), encoding="utf-8")
    yield str(config_file)


def _insert_race(conn, race_id, date_str, place="中山", distance=1600, course_type="芝"):
    conn.execute("""
        INSERT OR REPLACE INTO races (race_id, race_name, date, course_type, distance, direction, weather, surface_state, race_class, place)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (race_id, "テストレース", date_str, course_type, distance, "右", "晴", "良", "1勝クラス", place))


def _insert_result(conn, race_id, horse_id, rank, jockey_id="j1", trainer_id="t1"):
    conn.execute("""
        INSERT OR REPLACE INTO results (
            race_id, horse_id, rank, bracket_num, horse_num, gender, age, weight_carrier,
            horse_weight, weight_diff, jockey_id, trainer_id, time_sec, margin, last_3f,
            passing_order, tansho_odds, popularity, prize
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (race_id, horse_id, rank, 1, 1, "牡", 4, 55.0, 480.0, 0.0, jockey_id, trainer_id,
          90.0 if rank is not None else None, "1", 35.0, "3-3", 3.0, 1, 100.0))


def _insert_horse_result(conn, horse_id, date_str, race_id, rank, place="中山", course_type="芝", distance=1600):
    conn.execute("""
        INSERT OR REPLACE INTO horse_results (
            horse_id, date, race_id, rank, distance, course_type, place,
            time_sec, margin, last_3f, passing_order, prize
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (horse_id, date_str, race_id, rank, distance, course_type, place, 90.0, 0.1, 35.0, "3-3", 100.0))


def test_train_and_predict_time_features_match(test_env):
    """race C の特徴量が、(a)学習時(horse_resultsにレースC自身の行が存在する)でも
    (b)予測時(horse_resultsにレースC自身の行がまだ存在しない=出馬表段階)でも、
    どちらも「レースA・Bの2走分の履歴」に基づく同一の値になることを検証する。
    (このテストが無い状態だと、旧shift(1)実装では予測時に直近1走(レースB)の情報が
    欠落し、学習時とは異なる値になるバグが再発しうる)"""
    with open(test_env, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    parser = NetkeibaParser(config_path=test_env)
    conn = parser.conn

    horse_id = "2020100001"

    # 過去2走 (A, B) を races/results/horse_results 全てに登録(確定済みレース)
    _insert_race(conn, "202001010101", "2024-01-01")
    _insert_result(conn, "202001010101", horse_id, rank="3")
    _insert_horse_result(conn, horse_id, "2024-01-01", "202001010101", rank="3")

    _insert_race(conn, "202001010201", "2024-02-01")
    _insert_result(conn, "202001010201", horse_id, rank="1")
    _insert_horse_result(conn, horse_id, "2024-02-01", "202001010201", rank="1")

    # レースC: races/resultsには登録するが、horse_resultsにはまだ登録しない
    # (= 出馬表段階の「未実施レース」を模擬。実際の運用でも、当該レース自身の
    #  horse_results行は、そのレースが終わってhorse詳細ページが更新されるまで存在しない)
    _insert_race(conn, "202001010301", "2024-03-01")
    _insert_result(conn, "202001010301", horse_id, rank=None)  # parse_shutuba相当(結果未確定)

    conn.commit()

    builder = FeatureBuilder(config_path=test_env)
    df_predict_time = builder.build_features()

    row_c_predict = df_predict_time[df_predict_time["race_id"] == "202001010301"].iloc[0]

    # 続けて、レースCが実際に終わった状態を模擬(horse_resultsにレースC自身の行が追加される)
    _insert_result(conn, "202001010301", horse_id, rank="2")
    _insert_horse_result(conn, horse_id, "2024-03-01", "202001010301", rank="2")
    conn.commit()

    df_train_time = builder.build_features()
    row_c_train = df_train_time[df_train_time["race_id"] == "202001010301"].iloc[0]

    # レースA(3着)・B(1着)の2走の平均 = (3+1)/2 = 2.0 がどちらの時点でも一致するはず
    assert row_c_predict["rank_mean_all"] == pytest.approx(2.0)
    assert row_c_train["rank_mean_all"] == pytest.approx(2.0)
    assert row_c_predict["rank_mean_3"] == pytest.approx(row_c_train["rank_mean_3"])
    assert row_c_predict["interval_days"] == row_c_train["interval_days"]

    parser.close()


def test_out_of_scope_rows_excluded(test_env):
    """仕様書§2: 障害レース・JRA以外(場コード01〜10以外)のレース、および出走取消・競走除外の馬が
    特徴量テーブル(=学習・評価対象)から除外されることを検証する"""
    parser = NetkeibaParser(config_path=test_env)
    conn = parser.conn

    # JRA平地(場コード06=中山): 1着馬と取消馬
    _insert_race(conn, "202406010101", "2024-01-06")
    _insert_result(conn, "202406010101", "h_flat", rank="1")
    _insert_result(conn, "202406010101", "h_scratched", rank="取消")
    # JRA障害
    _insert_race(conn, "202406010102", "2024-01-06", course_type="障")
    _insert_result(conn, "202406010102", "h_jump", rank="1")
    # 地方(場コード44=大井)
    _insert_race(conn, "202444010101", "2024-01-06", place="大井")
    _insert_result(conn, "202444010101", "h_nar", rank="1")
    conn.commit()

    df = FeatureBuilder(config_path=test_env).build_features()

    assert set(df["horse_id"]) == {"h_flat"}
    assert df["n_horses"].tolist() == [1]

    parser.close()
