import os
import glob
import argparse
import calendar
import requests
import pandas as pd
from datetime import datetime, timedelta
import yaml

from src.scraping.netkeiba_source import NetkeibaSource
from src.preprocessing.parser import NetkeibaParser
from src.preprocessing.features import FeatureBuilder
from src.training.train import Trainer
from src.training.calibration import Calibrator
from src.simulation.simulator import Simulator
from src.predict.predictor import Predictor
from src.preprocessing.audit import run_audit

def daterange(start_date, end_date):
    for n in range(int((end_date - start_date).days) + 1):
        yield start_date + timedelta(n)

def _parse_scrape_date(date_str: str, is_end: bool) -> datetime:
    """'YYYY-MM-DD' または 'YYYY-MM' を日付に変換する。
    'YYYY-MM' の場合、is_end=Falseなら月初、is_end=Trueなら月末に展開する(仕様書9.1準拠)。"""
    try:
        return datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        pass
    year, month = datetime.strptime(date_str, "%Y-%m").timetuple()[:2]
    if is_end:
        last_day = calendar.monthrange(year, month)[1]
        return datetime(year, month, last_day)
    return datetime(year, month, 1)

def scrape_failures_path(parser) -> str:
    """取得失敗の記録先 (DBと同じフォルダ)。audit コマンドが未解消の失敗を報告するために使う"""
    return os.path.join(os.path.dirname(parser.db_path), "scrape_failures.tsv")

def log_scrape_failure(path: str, kind: str, item_id: str, error: Exception):
    with open(path, "a", encoding="utf-8") as f:
        message = " ".join(str(error).split())  # タブ・改行を含めない
        f.write(f"{datetime.now().isoformat(timespec='seconds')}\t{kind}\t{item_id}\t{message}\n")

def _horse_page_has_race(conn, horse_id: str, race_id: str) -> bool:
    cursor = conn.execute("SELECT 1 FROM horse_results WHERE horse_id = ? AND race_id = ? LIMIT 1",
                          (horse_id, race_id))
    return cursor.fetchone() is not None

def update_horse_pages(source, parser, race_id: str, failures_path: str):
    """出走馬の成績・血統ページを取得してパースする。
    保存済みの成績ページにこのレースが載っていなければ、レースより前に保存された古いページなので再取得する。
    (再取得しないと、ページ保存後に走ったレースが過去走から欠け、前走・間隔などの特徴量が誤る)
    取消・除外の馬は成績ページに載らない可能性があるため、この判定から除く。"""
    cursor = parser.conn.execute("SELECT horse_id, rank FROM results WHERE race_id = ?", (race_id,))
    for h_id, rank in cursor.fetchall():
        print(f"  Horse: {h_id}")
        try:
            source.get_horse_html(h_id)
            source.get_ped_html(h_id)
            parser.parse_horse(h_id)
            if rank not in (None, "取消", "除外") and not _horse_page_has_race(parser.conn, h_id, race_id):
                print(f"  Saved page of horse {h_id} predates race {race_id}. Re-fetching.")
                source.get_horse_html(h_id, force=True)
                parser.parse_horse(h_id)
                if not _horse_page_has_race(parser.conn, h_id, race_id):
                    print(f"  Warning: race {race_id} is still missing from the page of horse {h_id}.")
        except requests.exceptions.RequestException as e:
            print(f"Failed to fetch data for horse {h_id}: {e}. Skipping.")
            log_scrape_failure(failures_path, "horse", h_id, e)
            continue
        parser.parse_ped(h_id)

def handle_scrape(args):
    config_path = args.config
    source = NetkeibaSource(config_path)
    parser = NetkeibaParser(config_path)

    # 日付範囲の解決
    # "2024-01-01" または "2024-01"(月初〜月末に自動展開) の形式に対応
    try:
        start_dt = _parse_scrape_date(args.start, is_end=False)
        end_dt = _parse_scrape_date(args.to, is_end=True)
    except ValueError:
        print("Error: Date format must be YYYY-MM-DD or YYYY-MM")
        return
        
    print(f"Scraping from {start_dt.date()} to {end_dt.date()}...")
    failures_path = scrape_failures_path(parser)
    
    race_ids = []
    for single_date in daterange(start_dt, end_dt):
        date_str = single_date.strftime("%Y-%m-%d")
        print(f"Checking date: {date_str}")
        ids = source.get_race_ids_by_date(date_str)
        race_ids.extend(ids)
        
    print(f"Found {len(race_ids)} races total.")
    
    if args.limit_races:
        race_ids = race_ids[:args.limit_races]
        print(f"Limiting to first {args.limit_races} races.")
        
    # 各レースの取得とパース
    for r_idx, r_id in enumerate(race_ids):
        print(f"\n[{r_idx+1}/{len(race_ids)}] Processing race: {r_id}")
        try:
            source.get_race_html(r_id)
        except requests.exceptions.RequestException as e:
            # 個別レースの取得失敗でバッチ全体を止めない(異常なHTMLはディスクに保存されないため安全)
            print(f"Failed to fetch race {r_id}: {e}. Skipping.")
            log_scrape_failure(failures_path, "race", r_id, e)
            continue

        parsed = parser.parse_race(r_id)
        if not parsed:
            print(f"Skipped parsing for race {r_id} (HTML might not exist).")
            continue

        update_horse_pages(source, parser, r_id, failures_path)

    parser.close()
    print("Scraping and parsing complete.")

def handle_reparse(args):
    """保存済みHTML(data/html/race, horse, ped)からSQLiteを再生成する(仕様書§4.3: パース処理の変更時はHTMLから再生成)。
    出馬表(shutuba)は結果確定済みのレース行を未確定で上書きしうるため対象外とする。"""
    parser = NetkeibaParser(args.config)
    for category, parse in [("race", parser.parse_race), ("horse", parser.parse_horse), ("ped", parser.parse_ped)]:
        files = sorted(glob.glob(os.path.join(parser.html_dir, category, "*.html.gz")))
        print(f"Re-parsing {len(files)} {category} pages...")
        for path in files:
            item_id = os.path.basename(path)[:-len(".html.gz")]
            try:
                parse(item_id)
            except ValueError as e:
                # スキーマ検証エラー(HTML構造変更)は該当ページのみスキップして報告する
                print(f"Failed to parse {category}/{item_id}: {e}")
    parser.close()
    print("Re-parse complete.")

def handle_audit(args):
    run_audit(args.config)

def handle_build_features(args):
    print("Building features...")
    builder = FeatureBuilder(args.config)
    builder.build_features()

def handle_train(args):
    print("Training models...")
    trainer = Trainer(args.config)
    trainer.run(tune_hparam=args.tune)
    
    print("Calibrating model probabilities...")
    calibrator = Calibrator(args.config)
    calibrator.run()

def handle_simulate(args):
    print("Running walk-forward simulation...")
    sim = Simulator(args.config)
    sim.run_walk_forward()

def handle_predict(args):
    if not args.race_id:
        print("Error: --race-id is required for predict command.")
        return
    print(f"Predicting for race: {args.race_id}")
    pred = Predictor(args.config)
    pred.predict_race(args.race_id, budget=args.budget)

def main():
    parser = argparse.ArgumentParser(description="Keiba AI Prediction System CLI")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to config.yaml")
    
    subparsers = parser.add_subparsers(dest="command", required=True, help="Subcommand to run")
    
    # scrape
    parser_scrape = subparsers.add_parser("scrape", help="Scrape data from netkeiba")
    parser_scrape.add_argument("--start", type=str, required=True, help="Start date (YYYY-MM-DD or YYYY-MM)")
    parser_scrape.add_argument("--to", type=str, required=True, help="End date (YYYY-MM-DD or YYYY-MM)")
    parser_scrape.add_argument("--limit-races", type=int, default=None, help="Limit number of races to process (for testing)")
    
    # reparse
    subparsers.add_parser("reparse", help="Re-parse saved HTML into SQLite (after parser changes)")

    # audit
    subparsers.add_parser("audit", help="Check the consistency and coverage of scraped data in SQLite")

    # build-features
    parser_build = subparsers.add_parser("build-features", help="Process SQLite data and build Parquet features")
    
    # train
    parser_train = subparsers.add_parser("train", help="Train LightGBM models and calibration model")
    parser_train.add_argument("--tune", action="store_true", help="Tune hyperparameters using Optuna")
    
    # simulate
    parser_sim = subparsers.add_parser("simulate", help="Run walk-forward simulation")
    
    # predict
    parser_pred = subparsers.add_parser("predict", help="Predict race results and output bet recommendations")
    parser_pred.add_argument("--race-id", type=str, required=True, help="Race ID to predict (12 digits)")
    parser_pred.add_argument("--budget", type=float, default=None,
                             help="Budget in yen. Outputs fractional-Kelly win-bet stakes within this amount")
    
    args = parser.parse_args()
    
    if args.command == "scrape":
        handle_scrape(args)
    elif args.command == "reparse":
        handle_reparse(args)
    elif args.command == "audit":
        handle_audit(args)
    elif args.command == "build-features":
        handle_build_features(args)
    elif args.command == "train":
        handle_train(args)
    elif args.command == "simulate":
        handle_simulate(args)
    elif args.command == "predict":
        handle_predict(args)

if __name__ == "__main__":
    main()
