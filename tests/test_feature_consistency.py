import os
import sqlite3
import pandas as pd
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


def test_parse_rank_handles_demotion():
    """降着・再騎乗は先頭の数字を着順とし、中止・失格などはNaNとする"""
    ranks = pd.Series(["1", "4(降)", "9(再)", "中止", "失格", None])
    parsed = FeatureBuilder._parse_rank(ranks)
    assert parsed.iloc[:3].tolist() == [1, 4, 9]
    assert parsed.iloc[3:].isna().all()


def test_race_restricted_build_matches_full_build(test_env):
    """build_features(race_ids=...) で得た対象レースの特徴量が、全件生成時の同レースの行と一致することを検証する
    (当日予測では対象レースのみ計算するため、学習時と値がずれてはならない)"""
    from src.training.train import Trainer

    parser = NetkeibaParser(config_path=test_env)
    conn = parser.conn
    dates = ["2024-01-06", "2024-01-13", "2024-01-20", "2024-01-27", "2024-02-03", "2024-02-10", "2024-02-17"]
    for d_idx, d in enumerate(dates):
        race_id = f"2024060101{d_idx + 1:02d}"
        _insert_race(conn, race_id, d, distance=1200 + 200 * (d_idx % 3), course_type="芝" if d_idx % 2 else "ダ")
        for h in range(4):
            rank = str((h + d_idx) % 4 + 1)
            _insert_result(conn, race_id, f"h{h}", rank=rank, jockey_id=f"j{h % 2}", trainer_id=f"t{h % 3}")
            _insert_horse_result(conn, f"h{h}", d, race_id, rank=rank)
    conn.commit()

    builder = FeatureBuilder(config_path=test_env)
    target = "202406010107"
    df_full = builder.build_features()
    df_one = builder.build_features(race_ids=[target])

    full_rows = df_full[df_full["race_id"] == target].sort_values("horse_id").reset_index(drop=True)
    one_rows = df_one.sort_values("horse_id").reset_index(drop=True)
    assert len(one_rows) == 4
    # カテゴリ型は水準集合が、数値型はNaNの有無でint/floatが異なりうるため、値のみを比較する
    def normalize(df):
        df = df.copy()
        for col in df.select_dtypes("category").columns:
            df[col] = df[col].astype(str)
        return df
    pd.testing.assert_frame_equal(normalize(full_rows[one_rows.columns]), normalize(one_rows), check_dtype=False)

    parser.close()


def _insert_horse_result_full(conn, horse_id, date_str, race_id, rank, n_horses, passing_order,
                              race_class="1勝クラス", course_type="芝", place="中山"):
    conn.execute("""
        INSERT OR REPLACE INTO horse_results (
            horse_id, date, race_id, rank, distance, course_type, place,
            time_sec, margin, last_3f, passing_order, prize, n_horses, race_class
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (horse_id, date_str, race_id, rank, 1600, course_type, place, 90.0, 0.1, 35.0,
          passing_order, 100.0, n_horses, race_class))


def test_entity_top3_rate_shrinkage_and_cutoff(test_env):
    """騎手複勝率が「当日を含まない直近365日」の成績を事前分布で縮小推定した値になることを検証する"""
    from src.preprocessing.features import PRIOR_TOP3_RATE as P, PRIOR_WEIGHT as W
    parser = NetkeibaParser(config_path=test_env)
    conn = parser.conn
    # 2023-01-01: 2着(複勝圏) / 2024-01-06: 1着, 5着 / 2024-01-06 同日にもう1鞍 / 2024-01-13: 対象
    rides = [("202306010101", "2023-01-01", "2"), ("202406010101", "2024-01-06", "1"),
             ("202406010102", "2024-01-06", "5"), ("202406010201", "2024-01-13", "4")]
    for i, (race_id, d, rank) in enumerate(rides):
        _insert_race(conn, race_id, d)
        _insert_result(conn, race_id, f"h{i}", rank=rank, jockey_id="jx")
    conn.commit()

    df = FeatureBuilder(config_path=test_env).build_features().set_index("race_id")

    # 初騎乗: 過去なし -> 事前分布の値
    assert df.loc["202306010101", "jockey_top3_rate"] == pytest.approx(P)
    # 2024-01-06: 2023-01-01 は365日超前のため窓外。同日の他レースも含まない -> 過去0走
    assert df.loc["202406010102", "jockey_top3_rate"] == pytest.approx(P)
    # 2024-01-13: 2024-01-06 の2走(複勝1) のみ
    assert df.loc["202406010201", "jockey_top3_rate"] == pytest.approx((1 + W * P) / (2 + W))
    parser.close()


def test_relative_rank_run_style_class_and_jra_only(test_env):
    """相対着順・1角相対位置による脚質・クラス変化が計算され、地方の過去走は成績集計から除外される一方、
    前走間隔には含まれることを検証する"""
    parser = NetkeibaParser(config_path=test_env)
    conn = parser.conn
    horse_id = "hz"
    # JRA 2走: 16頭立て1着(1角1番手, 未勝利), 8頭立て4着(1角1番手, 1勝クラス)
    _insert_horse_result_full(conn, horse_id, "2024-01-06", "202406010101", "1", 16, "1-1", race_class="未勝利")
    _insert_horse_result_full(conn, horse_id, "2024-02-03", "202406010201", "4", 8, "1-2", race_class="1勝クラス")
    # 地方 1走 (場コード44): 成績集計からは除外、前走間隔には使う
    _insert_horse_result_full(conn, horse_id, "2024-03-02", "202444030201", "10", 12, "12-12",
                              race_class="その他", place="大井")
    # 対象レース (2勝クラス)
    _insert_race(conn, "202406040101", "2024-03-16")
    conn.execute("UPDATE races SET race_class='2勝クラス' WHERE race_id='202406040101'")
    _insert_result(conn, "202406040101", horse_id, rank="3")
    conn.commit()

    row = FeatureBuilder(config_path=test_env).build_features().set_index("race_id").loc["202406040101"]

    assert row["rank_ratio_mean_all"] == pytest.approx((1 / 16 + 4 / 8) / 2)
    assert row["rank_mean_all"] == pytest.approx(2.5)   # 地方の10着は含まない
    assert row["interval_days"] == 14                   # 地方の 2024-03-02 からの間隔
    assert row["run_style"] == "逃げ"                   # 1角相対位置 0, 0
    assert row["class_level_mean_5"] == pytest.approx(0.5)
    assert row["class_diff"] == pytest.approx(1)        # 1勝クラス -> 2勝クラス
    parser.close()


def test_sire_top3_rate_uses_only_past_offspring(test_env):
    """種牡馬×コース種別の複勝率が、対象日より前の産駒成績のみから計算されることを検証する"""
    from src.preprocessing.features import PRIOR_TOP3_RATE as P, PRIOR_WEIGHT as W
    parser = NetkeibaParser(config_path=test_env)
    conn = parser.conn
    for h in ["k1", "k2", "k3"]:
        conn.execute("INSERT INTO peds (horse_id, father_id) VALUES (?, ?)", (h, "sire1"))
    _insert_race(conn, "202406010101", "2024-01-06", course_type="芝")
    _insert_result(conn, "202406010101", "k1", rank="1")
    _insert_race(conn, "202406010102", "2024-01-06", course_type="ダ")
    _insert_result(conn, "202406010102", "k2", rank="1")
    _insert_race(conn, "202406010201", "2024-01-13", course_type="芝")
    _insert_result(conn, "202406010201", "k3", rank="8")
    conn.commit()

    df = FeatureBuilder(config_path=test_env).build_features().set_index("race_id")

    # 芝の過去産駒成績は k1 の1走(複勝1)のみ。ダートの k2 は含まない
    assert df.loc["202406010201", "sire_top3_rate_course"] == pytest.approx((1 + W * P) / (1 + W))
    assert df.loc["202406010101", "sire_top3_rate_course"] == pytest.approx(P)
    parser.close()


def _insert_timed_run(conn, race_id, date_str, horse_id, rank, time_sec):
    """結果(results)と馬の過去走(horse_results)の両方に、指定タイムの1走を登録する"""
    conn.execute("""
        INSERT OR REPLACE INTO results (race_id, horse_id, rank, horse_num, jockey_id, trainer_id, time_sec)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (race_id, horse_id, rank, int(rank), "j", "t", time_sec))
    conn.execute("""
        INSERT OR REPLACE INTO horse_results (horse_id, date, race_id, rank, distance, course_type, place,
                                              time_sec, n_horses, race_class)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (horse_id, date_str, race_id, rank, 1600, "芝", "中山", time_sec, 10, "1勝クラス"))


def test_speed_figure_par_and_track_variant(test_env):
    """スピード指数 = -(1000mあたりの基準タイムとの差 - 当日の馬場差) を手計算と照合する。
    - 1〜5日目: 中山芝1600の勝ち馬タイム 96.0 秒(各日1レース) -> 6日目の基準タイム 96.0 (過去5頭)
    - 6日目: 3レースの勝ち馬が全て 96.8 秒 -> 基準タイムとの差 0.8/1.6 = 0.5 秒/1000m が馬場差
      馬Xは 97.6 秒で2着 -> 差 1.0 秒/1000m、指数 = -(1.0 - 0.5) = -0.5
    - 7日目: 馬Xの出走。過去走の指数平均は -0.5"""
    parser = NetkeibaParser(config_path=test_env)
    conn = parser.conn
    for day in range(1, 6):
        race_id = f"2024060101{day:02d}"
        date_str = f"2024-01-{day:02d}"
        _insert_race(conn, race_id, date_str)
        _insert_timed_run(conn, race_id, date_str, f"w{day}", "1", 96.0)
    for r in range(1, 4):
        race_id = f"20240601060{r}"
        _insert_race(conn, race_id, "2024-01-06")
        _insert_timed_run(conn, race_id, "2024-01-06", f"v{r}", "1", 96.8)
    _insert_timed_run(conn, "202406010601", "2024-01-06", "X", "2", 97.6)
    _insert_race(conn, "202406010701", "2024-01-07")
    _insert_result(conn, "202406010701", "X", rank=None)
    conn.commit()

    df = FeatureBuilder(config_path=test_env).build_features()
    row = df[(df["race_id"] == "202406010701") & (df["horse_id"] == "X")].iloc[0]

    assert row["speed_fig_mean_3"] == pytest.approx(-0.5)
    assert row["speed_fig_max_5"] == pytest.approx(-0.5)
    parser.close()
