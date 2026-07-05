import os
import time
import gzip
import requests
import yaml
from urllib3.util import Retry
from requests.adapters import HTTPAdapter
from src.scraping.datasource import DataSource

class NetkeibaSource(DataSource):
    def __init__(self, config_path="config/config.yaml"):
        # configの読み込み
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)
        
        # パスの解決
        base_dir = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        html_dir_rel = self.config["paths"]["html_dir"]
        self.html_dir = os.path.join(base_dir, html_dir_rel)
        
        self.interval = self.config["scraping"]["interval"]
        self.retry_count = self.config["scraping"]["retry_count"]
        self.user_agent = self.config["scraping"]["user_agent"]
        
        # フォルダの作成
        for category in ["race", "horse", "ped", "shutuba"]:
            os.makedirs(os.path.join(self.html_dir, category), exist_ok=True)
            
        # requestsセッションの設定
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": self.user_agent})
        
        # 指数バックオフ的リトライの設定
        retries = Retry(
            total=self.retry_count,
            backoff_factor=1,  # 1s, 2s, 4s...
            status_forcelist=[500, 502, 503, 504],
            raise_on_status=True
        )
        self.session.mount("http://", HTTPAdapter(max_retries=retries))
        self.session.mount("https://", HTTPAdapter(max_retries=retries))

    def _get_and_save(self, url: str, category: str, item_id: str, force: bool = False,
                      encoding: str = "euc-jp") -> str:
        """URLを取得してgzip保存する。encodingはページ種別ごとに異なる:
        db.netkeiba.com はEUC-JP、race.netkeiba.com (出馬表) はUTF-8。"""
        file_path = os.path.join(self.html_dir, category, f"{item_id}.html.gz")

        if os.path.exists(file_path) and not force:
            # 既に存在する場合は読み込んで返す
            with gzip.open(file_path, "rt", encoding=encoding, errors="replace") as f:
                return f.read()

        # 最低1秒以上のインターバル
        time.sleep(self.interval)

        response = self.session.get(url)
        # エラーレスポンス(4xx/5xx)をディスクに保存すると、以後「取得済み」として
        # 恒久的にスキップされ、パースが黙って失敗し続けるため、保存前に必ず検査する。
        response.raise_for_status()
        response.encoding = encoding
        html_content = response.text

        # 保存
        with gzip.open(file_path, "wt", encoding=encoding, errors="replace") as f:
            f.write(html_content)

        return html_content

    def get_race_html(self, race_id: str) -> str:
        url = f"https://db.netkeiba.com/race/{race_id}"
        return self._get_and_save(url, "race", race_id)

    def get_horse_html(self, horse_id: str, force: bool = False) -> str:
        # 注意: /horse/{id} (プロフィールページ)には競走成績テーブルが含まれない。
        # 全成績テーブル(db_h_race_results)は /horse/result/{id}/ にある。
        url = f"https://db.netkeiba.com/horse/result/{horse_id}/"
        return self._get_and_save(url, "horse", horse_id, force=force)

    def get_ped_html(self, horse_id: str) -> str:
        url = f"https://db.netkeiba.com/horse/ped/{horse_id}"
        return self._get_and_save(url, "ped", horse_id)

    def get_shutuba_html(self, race_id: str) -> str:
        # race.netkeiba.com はUTF-8 (db.netkeiba.comのEUC-JPと異なる)
        url = f"https://race.netkeiba.com/race/shutuba.html?race_id={race_id}"
        return self._get_and_save(url, "shutuba", race_id, force=True, encoding="utf-8")

    def get_tansho_odds(self, race_id: str) -> dict:
        """netkeibaのオッズAPIから単勝オッズを取得する(直前オッズ用)。
        返り値: {馬番(int): 単勝オッズ(float)}。取得失敗時は空dict。
        出馬表の静的HTMLにはオッズが含まれない(JS読み込み)ため、このAPIを使用する。"""
        url = (f"https://race.netkeiba.com/api/api_get_jra_odds.html"
               f"?race_id={race_id}&type=1&action=init")
        time.sleep(self.interval)
        try:
            response = self.session.get(url)
            response.raise_for_status()
            payload = response.json()
        except (requests.exceptions.RequestException, ValueError) as e:
            print(f"Failed to fetch odds for race {race_id}: {e}")
            return {}

        # 監査用に生JSONを保存しておく
        odds_dir = os.path.join(self.html_dir, "odds")
        os.makedirs(odds_dir, exist_ok=True)
        with gzip.open(os.path.join(odds_dir, f"{race_id}.json.gz"), "wt", encoding="utf-8") as f:
            f.write(response.text)

        result = {}
        odds_map = payload.get("data", {}).get("odds", {}).get("1", {})
        for horse_num_str, values in odds_map.items():
            try:
                odds = float(values[0])
                if odds > 0:
                    result[int(horse_num_str)] = odds
            except (ValueError, IndexError, TypeError):
                continue
        return result

    def get_race_ids_by_date(self, date_str: str) -> list:
        """YYYYMMDD 形式の日付を指定し、その日に開催されたレースID一覧を取得する"""
        ymd = date_str.replace("-", "").replace("/", "")
        url = f"https://db.netkeiba.com/race/list/{ymd}/"
        
        time.sleep(self.interval)
        try:
            response = self.session.get(url)
            response.encoding = "euc-jp"
            html = response.text
            
            import re
            race_ids = re.findall(r"/race/(\d{12})", html)
            return sorted(list(set(race_ids)))
        except Exception as e:
            print(f"Error fetching race list for date {date_str}: {e}")
            return []
