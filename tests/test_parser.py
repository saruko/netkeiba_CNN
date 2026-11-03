import os
import gzip
import sqlite3
import yaml
import pytest
from src.preprocessing.parser import NetkeibaParser

TEST_CONFIG = """
paths:
  data_dir: "tests/test_data"
  html_dir: "tests/test_data/html"
  db_path: "tests/test_data/db.sqlite"
  features_dir: "tests/test_data/features"
  models_dir: "tests/test_data/models"

scraping:
  interval: 0.1
  retry_count: 1
  user_agent: "TestAgent/1.0"
"""

@pytest.fixture
def test_env(tmp_path):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    config_file = config_dir / "config.yaml"
    
    config_content = TEST_CONFIG.replace("tests/test_data", str(tmp_path).replace("\\", "/"))
    config_file.write_text(config_content, encoding="utf-8")
    
    html_dir = tmp_path / "html"
    (html_dir / "race").mkdir(parents=True)
    (html_dir / "horse").mkdir(parents=True)
    (html_dir / "ped").mkdir(parents=True)
    (html_dir / "shutuba").mkdir(parents=True)
    
    yield str(config_file)

def test_parse_race(test_env):
    dummy_race_html = """
    <html>
    <body>
        <p class="smalltxt">2020年12月20日 5回中山6日目</p>
        <div class="data_intro">
            <h1>有馬記念(G1)</h1>
            <span>芝右2500m / 天候 : 晴 / 芝 : 良 / 発走 : 15:25</span>
        </div>
        <table class="race_table_01">
            <tr>
                <th>着順</th><th>枠</th><th>馬番</th><th>馬名</th><th>性齢</th><th>斤量</th><th>騎手</th><th>タイム</th><th>着差</th><th>ﾀｲﾑ指数</th><th>通過</th><th>上り</th><th>単勝</th><th>人気</th><th>馬体重</th><th>調教ﾀｲﾑ</th><th>厩舎ｺﾒﾝﾄ</th><th>備考</th><th>調教師</th><th>馬主</th><th>賞金(万円)</th>
            </tr>
            <tr>
                <td>1</td><td>5</td><td>9</td><td><a href="/horse/2016104889">クロノジェネシス</a></td><td>牝4</td><td>55</td><td><a href="/jockey/01015">北村友一</a></td><td>2:35.0</td><td></td><td></td><td>5-5-5-4</td><td>36.2</td><td>2.5</td><td>1</td><td>474(-2)</td><td></td><td></td><td></td><td><a href="/trainer/01111">斉藤崇史</a></td><td></td><td></td>
            </tr>
        </table>
        <table class="pay_table_01">
            <tr>
                <th>単勝</th><td>9</td><td>250</td>
            </tr>
        </table>
    </body>
    </html>
    """
    
    with open(test_env, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    html_file = os.path.join(config["paths"]["html_dir"], "race", "202006050611.html.gz")
    with gzip.open(html_file, "wt", encoding="euc-jp", errors="replace") as f:
        f.write(dummy_race_html)
        
    parser = NetkeibaParser(config_path=test_env)
    success = parser.parse_race("202006050611")
    assert success
    
    conn = sqlite3.connect(parser.db_path)
    cursor = conn.cursor()
    
    cursor.execute("SELECT race_name, date, distance, course_type, place FROM races WHERE race_id='202006050611'")
    race = cursor.fetchone()
    assert race == ("有馬記念(G1)", "2020-12-20", 2500, "芝", "中山")
    
    cursor.execute("""
        SELECT horse_id, rank, bracket_num, horse_num, gender, age, weight_carrier, 
               horse_weight, weight_diff, jockey_id, trainer_id, time_sec, last_3f, 
               passing_order, tansho_odds, popularity 
        FROM results WHERE race_id='202006050611'
    """)
    result = cursor.fetchone()
    assert result == ("2016104889", "1", 5, 9, "牝", 4, 55.0, 474.0, -2.0, "01015", "01111", 155.0, 36.2, "5-5-5-4", 2.5, 1)
    
    cursor.execute("SELECT ticket_type, horse_num, payoff FROM returns WHERE race_id='202006050611'")
    ret = cursor.fetchone()
    assert ret == ("単勝", 9, 250)

    conn.close()
    parser.close()


def test_parse_horse_header_based(test_env):
    """馬成績ページ(/horse/result/)のパースがヘッダー名ベースで列位置を解決することを検証する。
    netkeibaの2026年時点の33列レイアウト(水分量・指数列が挿入された形)を模したフィクスチャを使用。"""
    headers = ["日付", "開催", "天気", "R", "レース名", "映像", "頭数", "枠番", "馬番",
               "オッズ", "人気", "着順", "騎手", "斤量", "距離", "水分量", "馬場", "馬場指数",
               "タイム", "着差", "ﾀｲﾑ指数", "ﾀｲﾑ指数M", "ｽﾀｰﾄ指数", "追走指数", "上がり指数",
               "通過", "ペース", "上り", "馬体重", "厩舎ｺﾒﾝﾄ", "備考", "勝ち馬(2着馬)", "賞金"]
    values = ["2024/06/09", "3中山6", "晴", "10", '<a href="/race/202406030610/">テストレース</a>', "",
              "16", "3", "5", "4.2", "2", "1", "テスト騎手", "55", "芝1600", "", "良", "",
              "1:33.5", "-0.2", "", "", "", "", "", "3-3-2-1", "35.1-34.8", "34.5",
              "480(+2)", "", "", "2着馬名", "1,100.0"]

    header_html = "".join(f"<th>{h}</th>" for h in headers)
    value_html = "".join(f"<td>{v}</td>" for v in values)
    dummy_horse_html = f"""
    <html><body>
    <table class="db_h_race_results nk_tb_common">
        <tr>{header_html}</tr>
        <tr>{value_html}</tr>
    </table>
    </body></html>
    """

    with open(test_env, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    horse_id = "2020104889"
    html_file = os.path.join(config["paths"]["html_dir"], "horse", f"{horse_id}.html.gz")
    with gzip.open(html_file, "wt", encoding="euc-jp", errors="replace") as f:
        f.write(dummy_horse_html)

    parser = NetkeibaParser(config_path=test_env)
    success = parser.parse_horse(horse_id)
    assert success

    conn = sqlite3.connect(parser.db_path)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT date, race_id, rank, distance, course_type, place, time_sec, margin, last_3f, passing_order, prize,
               n_horses, race_class
        FROM horse_results WHERE horse_id=?
    """, (horse_id,))
    row = cursor.fetchone()
    assert row == ("2024-06-09", "202406030610", "1", 1600, "芝", "中山",
                   93.5, -0.2, 34.5, "3-3-2-1", 1100.0, 16, "その他")

    conn.close()
    parser.close()


@pytest.mark.parametrize("race_name,expected", [
    ("有馬記念(G1)", "G1"),
    ("ジュニアカップ(L)", "オープン"),
    ("4歳以上1勝クラス", "1勝クラス"),
    ("4歳以上1000万下", "2勝クラス"),
    ("3歳未勝利", "未勝利"),
    ("2歳新馬", "新馬"),
    ("テストレース", "その他"),
])
def test_parse_race_class(race_name, expected):
    assert NetkeibaParser.parse_race_class(race_name) == expected
