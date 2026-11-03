import pytest
from src.preprocessing.parser import NetkeibaParser
from src.preprocessing.audit import audit_db

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
def conn(tmp_path):
    base = str(tmp_path).replace("\\", "/")
    config_file = tmp_path / "config.yaml"
    config_file.write_text(TEST_CONFIG.format(base=base), encoding="utf-8")
    parser = NetkeibaParser(config_path=str(config_file))
    yield parser.conn
    parser.close()


def insert_clean_race(conn, race_id, place="中山", resolved=True):
    """問題の無い3頭立てのレース。馬IDは {race_id}_1..3、1着は馬番1(単勝2.5倍 -> 払戻250円)"""
    conn.execute("INSERT INTO races (race_id, date, course_type, distance, place) VALUES (?, '2024-01-06', '芝', 1600, ?)",
                 (race_id, place))
    for num, (rank, time_sec, odds) in enumerate([("1", 90.0, 2.5), ("2", 90.2, 4.0), ("3", 90.5, 6.0)], start=1):
        horse_id = f"{race_id}_{num}"
        conn.execute("INSERT INTO results (race_id, horse_id, rank, horse_num, time_sec, tansho_odds) VALUES (?, ?, ?, ?, ?, ?)",
                     (race_id, horse_id, rank if resolved else None, num, time_sec, odds))
        conn.execute("INSERT INTO horse_results (horse_id, date, race_id, n_horses) VALUES (?, '2024-01-06', ?, 3)",
                     (horse_id, race_id))
        conn.execute("INSERT INTO peds (horse_id, father_id) VALUES (?, 'sire')", (horse_id,))
    conn.execute("INSERT INTO returns VALUES (?, '単勝', 1, 250)", (race_id,))


def race(i):
    return f"2024060101{i:02d}"  # 場コード06(中山)


def test_audit_detects_each_injected_problem(conn, tmp_path):
    for i in range(12):
        insert_clean_race(conn, race(i))
    # 対象外: 地方(場コード44)と、出馬表のみの未確定レース。問題があっても報告しない
    insert_clean_race(conn, "202444010101", place="大井")
    conn.execute("DELETE FROM returns WHERE race_id = '202444010101'")
    insert_clean_race(conn, race(20), resolved=False)
    conn.execute("DELETE FROM returns WHERE race_id = ?", (race(20),))

    conn.execute("UPDATE races SET place = '' WHERE race_id = ?", (race(1),))
    conn.execute("UPDATE horse_results SET n_horses = 5 WHERE race_id = ?", (race(2),))
    conn.execute("UPDATE results SET rank = '中止' WHERE race_id = ? AND rank = '1'", (race(3),))
    conn.execute("DELETE FROM returns WHERE race_id = ?", (race(4),))
    conn.execute("UPDATE returns SET payoff = 990 WHERE race_id = ?", (race(5),))
    conn.execute("UPDATE results SET tansho_odds = NULL WHERE race_id = ? AND rank = '3'", (race(6),))
    conn.execute("UPDATE results SET time_sec = 89.0 WHERE race_id = ? AND rank = '3'", (race(7),))
    conn.execute("DELETE FROM horse_results WHERE horse_id = ?", (f"{race(8)}_3",))
    conn.execute("UPDATE horse_results SET race_id = 'older' WHERE horse_id = ?", (f"{race(9)}_3",))
    conn.execute("DELETE FROM peds WHERE horse_id = ?", (f"{race(10)}_3",))
    # 降着による着順とタイムの逆転は正常
    conn.execute("UPDATE results SET rank = '2(降)', time_sec = 89.9 WHERE race_id = ? AND rank = '2'", (race(11),))
    conn.commit()

    failures = tmp_path / "scrape_failures.tsv"
    failures.write_text(
        f"2024-01-07T10:00:00\trace\t202406019999\t503 Server Error\n"   # 未解消
        f"2024-01-07T10:00:01\trace\t{race(0)}\t503 Server Error\n"      # 取り直し済み
        f"2024-01-07T10:00:02\thorse\t{race(0)}_1\ttimeout\n",           # 取り直し済み
        encoding="utf-8")

    issues, n_races, n_starts = audit_db(conn, str(failures))

    found = {(r.check, r.race_id if isinstance(r.race_id, str) else r.horse_id) for r in issues.itertuples()}
    assert found == {
        ("race_missing_fields", race(1)),
        ("n_horses_mismatch", race(2)),
        ("no_winner", race(3)),
        ("missing_tansho_return", race(4)),
        ("tansho_payoff_mismatch", race(5)),
        ("missing_odds", race(6)),
        ("rank_time_inversion", race(7)),
        ("horse_page_missing", race(8)),
        ("horse_page_stale", race(9)),
        ("ped_missing", race(10)),
        ("scrape_failure_unresolved", "202406019999"),
    }
    assert (n_races, n_starts) == (12, 36)


def test_audit_clean_db_has_no_issues(conn):
    insert_clean_race(conn, race(0))
    conn.execute("INSERT INTO results (race_id, horse_id, rank, horse_num) VALUES (?, 'scr', '取消', 4)", (race(0),))
    conn.commit()
    issues, n_races, n_starts = audit_db(conn)
    assert issues.empty
    assert (n_races, n_starts) == (1, 3)
