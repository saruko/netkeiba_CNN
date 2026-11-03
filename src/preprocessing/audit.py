import os
import sqlite3
import pandas as pd
import yaml
from src.preprocessing.features import JRA_PLACE_CODES

SCRATCHED = ["取消", "除外"]

# 検査名 -> 説明 (表示順)
CHECKS = {
    "race_missing_fields": "レース情報(日付・距離・コース種別・開催場)の欠損",
    "n_horses_mismatch": "結果表の行数が、成績ページの頭数と一致しない",
    "no_winner": "1着馬がいない(結果表のパース失敗の疑い)",
    "missing_tansho_return": "単勝の払戻がない",
    "tansho_payoff_mismatch": "1着馬の単勝払戻が 単勝オッズ×100 と一致しない(同着の可能性あり)",
    "missing_odds": "完走馬の単勝オッズが欠損",
    "rank_time_inversion": "上位の着順の馬ほどタイムが遅い(降着を除く)",
    "horse_page_missing": "出走馬の成績ページが無い",
    "horse_page_stale": "出走馬の成績ページにこのレースが載っていない(ページが古い)",
    "ped_missing": "出走馬の血統が無い",
    "scrape_failure_unresolved": "取得に失敗し、まだ取り直されていないページ",
}


def _issues(check: str, df: pd.DataFrame, detail=None) -> pd.DataFrame:
    out = pd.DataFrame({
        "check": check,
        "race_id": df["race_id"].values if "race_id" in df else None,
        "horse_id": df["horse_id"].values if "horse_id" in df else None,
    })
    out["detail"] = detail.values if isinstance(detail, pd.Series) else detail
    return out


def audit_db(conn, failures_path: str = None) -> tuple:
    """DBの整合性を検査する。対象はJRA平地で、結果が確定したレース(出馬表のみの未確定レースは除く)。
    返り値: (問題の一覧DataFrame[check, race_id, horse_id, detail], 対象のレース数, 対象の出走数)"""
    races = pd.read_sql_query("SELECT * FROM races", conn)
    results = pd.read_sql_query("SELECT race_id, horse_id, rank, horse_num, time_sec, tansho_odds FROM results", conn)
    returns = pd.read_sql_query("SELECT * FROM returns WHERE ticket_type = '単勝'", conn)
    hr = pd.read_sql_query("SELECT horse_id, race_id, n_horses FROM horse_results", conn)
    peds = pd.read_sql_query("SELECT horse_id, father_id FROM peds", conn)
    # pandas 3 の文字列型(Arrow)は大きな集合に対する isin が非常に遅いため、ID列は object 型で扱う
    for df in (races, results, returns, hr, peds):
        for col in ("race_id", "horse_id"):
            if col in df:
                df[col] = df[col].astype(object)

    is_target = races["race_id"].astype(str).str[4:6].isin(JRA_PLACE_CODES) & (races["course_type"] != "障")
    races = races[is_target]
    results = results[results["race_id"].isin(races["race_id"])]
    resolved_ids = results.loc[results["rank"].notna(), "race_id"].unique()
    races = races[races["race_id"].isin(resolved_ids)]
    results = results[results["race_id"].isin(resolved_ids)].copy()
    results["rank_num"] = pd.to_numeric(results["rank"].astype(str).str.extract(r"^(\d+)")[0], errors="coerce")
    started = results[~results["rank"].isin(SCRATCHED)]

    found = []

    bad = races[(races["date"].fillna("") == "") | (races["distance"].fillna(0) <= 0)
                | (races["course_type"].fillna("") == "") | (races["place"].fillna("") == "")]
    found.append(_issues("race_missing_fields", bad))

    # 成績ページの頭数は出走取消を含むかが不確かなため、全行数・取消除外を除いた行数のどちらとも一致しない場合のみ問題とする
    counts = (results.assign(started=~results["rank"].isin(SCRATCHED))
              .groupby("race_id").agg(n_all=("horse_id", "size"), n_started=("started", "sum")))
    # 成績ページの頭数はレースごとに最頻値を使う
    hr_target = hr[hr["race_id"].isin(races["race_id"])].dropna(subset=["n_horses"])
    n_page = (hr_target.groupby(["race_id", "n_horses"]).size().reset_index(name="n")
              .sort_values("n", ascending=False).drop_duplicates("race_id").set_index("race_id")["n_horses"])
    c = counts.join(n_page, how="inner").reset_index()
    bad = c[(c["n_all"] != c["n_horses"]) & (c["n_started"] != c["n_horses"])]
    found.append(_issues("n_horses_mismatch", bad,
                         "rows=" + bad["n_all"].astype(str) + ", page=" + bad["n_horses"].astype(int).astype(str)))

    winners = results[results["rank_num"] == 1]
    no_winner = races[~races["race_id"].isin(winners["race_id"])]
    found.append(_issues("no_winner", no_winner))

    no_return = races[~races["race_id"].isin(returns["race_id"])]
    found.append(_issues("missing_tansho_return", no_return))

    w = winners.merge(returns[["race_id", "horse_num", "payoff"]], on=["race_id", "horse_num"], how="inner")
    bad = w[(w["payoff"] - w["tansho_odds"] * 100).abs() > 5]
    found.append(_issues("tansho_payoff_mismatch", bad,
                         "payoff=" + bad["payoff"].astype(str) + ", odds=" + bad["tansho_odds"].astype(str)))

    bad = results[results["rank_num"].notna() & results["tansho_odds"].isna()]
    found.append(_issues("missing_odds", bad))

    t = results[results["rank_num"].notna() & (results["time_sec"] > 0)
                & ~results["rank"].astype(str).str.contains("降")]
    t = t.sort_values(["race_id", "rank_num"])
    faster_than_above = t["time_sec"] < t.groupby("race_id")["time_sec"].shift(1)
    bad = t[faster_than_above].drop_duplicates("race_id")
    found.append(_issues("rank_time_inversion", bad, "rank=" + bad["rank"].astype(str)))

    has_page = started["horse_id"].isin(hr["horse_id"])
    found.append(_issues("horse_page_missing", started[~has_page]))
    listed = started[has_page].merge(hr[["horse_id", "race_id"]].drop_duplicates(),
                                     on=["horse_id", "race_id"], how="left", indicator=True)
    found.append(_issues("horse_page_stale", listed[listed["_merge"] == "left_only"]))

    with_ped = peds.loc[peds["father_id"].fillna("") != "", "horse_id"]
    found.append(_issues("ped_missing", started[~started["horse_id"].isin(with_ped)].drop_duplicates("horse_id")))

    if failures_path and os.path.exists(failures_path):
        fails = pd.read_csv(failures_path, sep="\t", header=None, names=["time", "kind", "id", "error"],
                            dtype=str, quoting=3).drop_duplicates(["kind", "id"], keep="last")
        all_races = pd.read_sql_query("SELECT race_id FROM races", conn)["race_id"]
        unresolved = fails[((fails["kind"] == "race") & ~fails["id"].isin(all_races))
                           | ((fails["kind"] == "horse") & ~fails["id"].isin(hr["horse_id"]))]
        found.append(pd.DataFrame({
            "check": "scrape_failure_unresolved",
            "race_id": unresolved["id"].where(unresolved["kind"] == "race").values,
            "horse_id": unresolved["id"].where(unresolved["kind"] == "horse").values,
            "detail": unresolved["error"].values,
        }))

    issues = pd.concat(found, ignore_index=True)
    return issues, len(races), len(started)


def run_audit(config_path: str = "config/config.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    base_dir = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    db_path = os.path.join(base_dir, config["paths"]["db_path"])
    failures_path = os.path.join(os.path.dirname(db_path), "scrape_failures.tsv")

    conn = sqlite3.connect(db_path)
    issues, n_races, n_starts = audit_db(conn, failures_path)
    conn.close()

    print(f"対象: JRA平地の確定レース {n_races} 件 / 出走 {n_starts} 件")
    counts = issues["check"].value_counts()
    for check, desc in CHECKS.items():
        n = int(counts.get(check, 0))
        examples = issues.loc[issues["check"] == check, ["race_id", "horse_id"]].head(3)
        ids = ", ".join(str(r.race_id if pd.notna(r.race_id) else r.horse_id) for r in examples.itertuples())
        print(f"[{'OK' if n == 0 else 'NG'}] {check}: {n} 件 - {desc}" + (f" (例: {ids})" if ids else ""))

    report_path = os.path.join(os.path.dirname(db_path), "audit_report.csv")
    issues.to_csv(report_path, index=False, encoding="utf-8-sig")
    print(f"問題の一覧を {report_path} に保存しました")
    if counts.get("horse_page_stale", 0) > 0:
        print("  * horse_page_stale は、該当期間を再度 scrape すると古い成績ページが再取得されて解消します")
    return issues
