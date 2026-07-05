import os
import re
import gzip
import sqlite3
import yaml
from bs4 import BeautifulSoup

class NetkeibaParser:
    def __init__(self, config_path="config/config.yaml"):
        # 設定ファイルの読み込み
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)
            
        base_dir = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        db_path_rel = self.config["paths"]["db_path"]
        self.db_path = os.path.join(base_dir, db_path_rel)
        self.html_dir = os.path.join(base_dir, self.config["paths"]["html_dir"])
        
        # DB接続とテーブル初期化
        db_parent = os.path.dirname(self.db_path)
        if db_parent:
            os.makedirs(db_parent, exist_ok=True)
            
        self.conn = sqlite3.connect(self.db_path)
        self._create_tables()

    def _create_tables(self):
        cursor = self.conn.cursor()
        
        # races テーブル
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS races (
            race_id TEXT PRIMARY KEY,
            race_name TEXT,
            date TEXT,
            course_type TEXT,
            distance INTEGER,
            direction TEXT,
            weather TEXT,
            surface_state TEXT,
            race_class TEXT,
            place TEXT
        )
        """)
        
        # results テーブル
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS results (
            race_id TEXT,
            horse_id TEXT,
            rank TEXT,
            bracket_num INTEGER,
            horse_num INTEGER,
            gender TEXT,
            age INTEGER,
            weight_carrier REAL,
            horse_weight REAL,
            weight_diff REAL,
            jockey_id TEXT,
            trainer_id TEXT,
            time_sec REAL,
            margin TEXT,
            last_3f REAL,
            passing_order TEXT,
            tansho_odds REAL,
            popularity INTEGER,
            prize REAL,
            PRIMARY KEY (race_id, horse_id)
        )
        """)
        self._ensure_column("results", "prize", "REAL")
        
        # returns テーブル
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS returns (
            race_id TEXT,
            ticket_type TEXT,
            horse_num INTEGER,
            payoff INTEGER,
            PRIMARY KEY (race_id, ticket_type, horse_num)
        )
        """)
        
        # horse_results テーブル
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS horse_results (
            horse_id TEXT,
            date TEXT,
            race_id TEXT,
            rank TEXT,
            distance INTEGER,
            course_type TEXT,
            place TEXT,
            time_sec REAL,
            margin REAL,
            last_3f REAL,
            passing_order TEXT,
            prize REAL,
            PRIMARY KEY (horse_id, date)
        )
        """)
        self._ensure_column("horse_results", "course_type", "TEXT")
        self._ensure_column("horse_results", "place", "TEXT")
        self._ensure_column("horse_results", "margin", "REAL")
        self._ensure_column("horse_results", "prize", "REAL")
        
        # peds テーブル
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS peds (
            horse_id TEXT PRIMARY KEY,
            father_id TEXT,
            mother_id TEXT,
            father_of_father_id TEXT,
            mother_of_father_id TEXT,
            father_of_mother_id TEXT,
            mother_of_mother_id TEXT
        )
        """)
        
        self.conn.commit()

    def _ensure_column(self, table: str, column: str, coltype: str):
        """既存DBに新しいカラムが無い場合にALTER TABLEで追加する(スキーマ移行)"""
        cursor = self.conn.cursor()
        cursor.execute(f"PRAGMA table_info({table})")
        existing_cols = {row[1] for row in cursor.fetchall()}
        if column not in existing_cols:
            cursor.execute(f"ALTER TABLE {table} ADD COLUMN {column} {coltype}")
            self.conn.commit()

    def close(self):
        self.conn.close()

    def parse_margin(self, margin_str: str) -> float:
        """着差文字列を馬身換算の数値に変換する(例: 'ハナ'->0.05, 'クビ'->0.1, '1/2'->0.5, '1.1/4'->1.25)。
        変換不能な表記(同着、取消等)はNoneを返す。"""
        if not margin_str:
            return None
        s = margin_str.strip()
        if s in ("", "-"):
            return None
        special = {"ハナ": 0.05, "アタマ": 0.1, "クビ": 0.2, "同着": 0.0}
        if s in special:
            return special[s]
        # 分数混じり表記 (例: "1.1/4" -> 1 + 1/4, "3/4" -> 0.75)
        match = re.match(r"^(\d+)?\.?(\d+)/(\d+)$", s)
        if match:
            whole = int(match.group(1)) if match.group(1) else 0
            numerator = int(match.group(2))
            denominator = int(match.group(3))
            if denominator == 0:
                return None
            return whole + numerator / denominator
        try:
            return float(s)
        except ValueError:
            return None

    @staticmethod
    def _to_float(s: str):
        """数値文字列を安全にfloat変換する。'**'(計測不能)や'---'、空文字はNoneを返す。"""
        if not s:
            return None
        s = s.strip().replace(",", "")
        try:
            return float(s)
        except ValueError:
            return None

    @staticmethod
    def _to_int(s: str):
        """数値文字列を安全にint変換する。変換不能ならNoneを返す。"""
        if not s:
            return None
        s = s.strip()
        try:
            return int(s)
        except ValueError:
            return None

    def parse_time(self, time_str: str) -> float:
        """'1:33.1' のようなタイム文字列を秒数に変換する。変換不能な表記はNoneを返す。"""
        if not time_str or time_str == "0.0" or time_str == "0" or time_str == "-":
            return None
        parts = time_str.split(":")
        try:
            if len(parts) == 2:
                return float(parts[0]) * 60 + float(parts[1])
            elif len(parts) == 1:
                return float(parts[0])
        except ValueError:
            return None
        return None

    def parse_race(self, race_id: str) -> bool:
        """race_id に対応する生HTMLをパースしてraces, results, returnsに保存する"""
        file_path = os.path.join(self.html_dir, "race", f"{race_id}.html.gz")
        if not os.path.exists(file_path):
            return False

        with gzip.open(file_path, "rt", encoding="euc-jp", errors="replace") as f:
            html = f.read()

        soup = BeautifulSoup(html, "html.parser")
        
        # レース情報の取得
        # 開催日、場所など
        # 例: <p class="smalltxt">2020年12月20日 5回中山6日目</p>
        smalltxt = soup.find(class_="smalltxt")
        date_str = ""
        place = ""
        if smalltxt:
            text = smalltxt.get_text()
            # 2020年12月20日
            date_match = re.search(r"(\d{4})年(\d{1,2})月(\d{1,2})日", text)
            if date_match:
                date_str = f"{date_match.group(1)}-{int(date_match.group(2)):02d}-{int(date_match.group(3)):02d}"
            
            # 場所の抽出
            place_match = re.search(r"\d+回([^\d\s日]+)\d+日目", text)
            if place_match:
                place = place_match.group(1)

        # レース名
        race_name = ""
        intro = soup.find(class_="data_intro")
        if intro:
            h1 = intro.find("h1")
            if h1:
                race_name = h1.get_text().strip()

        # コース情報、天候、馬場状態など
        # 例: <diary_snap_cut><span>芝右1600m / 天候 : 晴 / 芝 : 良 / 発走 : 15:45</span></diary_snap_cut>
        # または data_intro の下にある span
        course_type = ""
        distance = 0
        direction = ""
        weather = ""
        surface_state = ""
        
        diary_snap = soup.find("diary_snap_cut")
        info_text = ""
        if diary_snap:
            info_text = diary_snap.get_text()
        elif intro:
            span = intro.find("span")
            if span:
                info_text = span.get_text()

        if info_text:
            # コースタイプと距離、方向
            # 例: "芝右1600m", "ダ左1800m", "障芝2800m"
            course_match = re.search(r"(芝|ダ|障)([左右直内外]*?)(\d+)m", info_text)
            if course_match:
                course_type = course_match.group(1)
                direction = course_match.group(2)
                distance = int(course_match.group(3))
            
            # 天候
            weather_match = re.search(r"天候\s*:\s*([^\s/]+)", info_text)
            if weather_match:
                weather = weather_match.group(1)

            # 馬場状態 (芝 / ダート : 状態)
            state_match = re.search(r"(芝|ダ|馬場)\s*:\s*([^\s/]+)", info_text)
            if state_match:
                surface_state = state_match.group(2)

        # クラス (例: 新馬, 未勝利, 1勝クラス, オープン, G1など)
        # 通常、p.smalltxt や race_name の中、あるいは他のテキストから推測
        # ここでは簡易的に、race_nameの中にG1/G2/G3が入っているか、または 1勝クラス 等の記述があるかでパースする
        race_class = "その他"
        if "G1" in race_name:
            race_class = "G1"
        elif "G2" in race_name:
            race_class = "G2"
        elif "G3" in race_name:
            race_class = "G3"
        elif "オープン" in race_name or "(L)" in race_name or "OP" in race_name:
            race_class = "オープン"
        elif "3勝" in race_name or "1600万" in race_name:
            race_class = "3勝クラス"
        elif "2勝" in race_name or "1000万" in race_name:
            race_class = "2勝クラス"
        elif "1勝" in race_name or "500万" in race_name:
            race_class = "1勝クラス"
        elif "未勝利" in race_name:
            race_class = "未勝利"
        elif "新馬" in race_name:
            race_class = "新馬"

        cursor = self.conn.cursor()
        
        # racesのインサート
        cursor.execute("""
        INSERT OR REPLACE INTO races (race_id, race_name, date, course_type, distance, direction, weather, surface_state, race_class, place)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (race_id, race_name, date_str, course_type, distance, direction, weather, surface_state, race_class, place))

        # レース結果のパース
        results_table = soup.find(class_="race_table_01")
        if results_table:
            rows = results_table.find_all("tr")
            for row in rows[1:]: # ヘッダーを飛ばす
                cols = row.find_all("td")
                if len(cols) < 10:
                    continue
                
                # 着順
                rank = cols[0].get_text().strip()
                # 枠番、馬番 (馬番は主キー相当のため、取得できない行はスキップ)
                bracket_num = self._to_int(cols[1].get_text())
                horse_num = self._to_int(cols[2].get_text())
                if horse_num is None:
                    continue

                # 馬ID
                horse_link = cols[3].find("a")
                if not horse_link:
                    continue
                h_match = re.search(r"horse/.*?(\d+)", horse_link.get("href"))
                horse_id = h_match.group(1) if h_match else ""

                # 性別・年齢 (例: "牡3", "牝4", "セ5")
                gender_age = cols[4].get_text().strip()
                gender = gender_age[0] if len(gender_age) > 0 else ""
                age = int(gender_age[1:]) if len(gender_age) > 1 and gender_age[1:].isdigit() else None

                # 斤量
                weight_carrier = self._to_float(cols[5].get_text())

                # 騎手ID
                jockey_link = cols[6].find("a")
                jockey_id = ""
                if jockey_link:
                    j_match = re.search(r"jockey/.*?(\d+)", jockey_link.get("href"))
                    jockey_id = j_match.group(1) if j_match else ""

                # タイム
                time_str = cols[7].get_text().strip()
                time_sec = self.parse_time(time_str)

                # 着差
                margin = cols[8].get_text().strip()

                # 通過順
                passing_order = cols[10].get_text().strip()

                # 上がり3F ('**' や '-' は計測なしとしてNone)
                last_3f = self._to_float(cols[11].get_text())

                # 単勝オッズ ('---' や '計不' はNone)
                tansho_odds = self._to_float(cols[12].get_text())

                # 人気
                popularity = self._to_int(cols[13].get_text())
                
                # 馬体重と増減 (例: "480(-2)", "502(+10)", "計不")
                weight_str = cols[14].get_text().strip()
                horse_weight = None
                weight_diff = None
                if weight_str and weight_str != "計不":
                    w_match = re.search(r"(\d+)\(([-+]*\d+)\)", weight_str)
                    if w_match:
                        horse_weight = float(w_match.group(1))
                        weight_diff = float(w_match.group(2))
                    else:
                        w_match_only = re.search(r"(\d+)", weight_str)
                        if w_match_only:
                            horse_weight = float(w_match_only.group(1))

                # 調教師ID
                trainer_link = cols[18].find("a") if len(cols) > 18 else None
                trainer_id = ""
                if trainer_link:
                    t_match = re.search(r"trainer/.*?(\d+)", trainer_link.get("href"))
                    trainer_id = t_match.group(1) if t_match else ""

                # 賞金(万円)
                prize = None
                if len(cols) > 20:
                    prize_str = cols[20].get_text().strip().replace(",", "")
                    try:
                        prize = float(prize_str) if prize_str and prize_str != "-" else None
                    except ValueError:
                        prize = None

                cursor.execute("""
                INSERT OR REPLACE INTO results (
                    race_id, horse_id, rank, bracket_num, horse_num, gender, age, weight_carrier,
                    horse_weight, weight_diff, jockey_id, trainer_id, time_sec, margin, last_3f,
                    passing_order, tansho_odds, popularity, prize
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    race_id, horse_id, rank, bracket_num, horse_num, gender, age, weight_carrier,
                    horse_weight, weight_diff, jockey_id, trainer_id, time_sec, margin, last_3f,
                    passing_order, tansho_odds, popularity, prize
                ))

        # 払戻情報のパース
        pay_tables = soup.find_all(class_="pay_table_01")
        for table in pay_tables:
            rows = table.find_all("tr")
            for row in rows:
                cols = row.find_all("th")
                tds = row.find_all("td")
                if not cols or not tds:
                    continue
                ticket_type = cols[0].get_text().strip()
                
                # 単勝・複勝のみ対象
                if ticket_type not in ["単勝", "複勝"]:
                    continue
                
                # 馬番と払戻金。同着などで複数行に分割、またはカンマ区切りの可能性あり。
                # 例: <td>1</td> <td>230</td>
                # 複勝の例: <td>1<br>2<br>3</td> <td>110<br>120<br>130</td>
                nums_text = tds[0].get_text(separator="|").split("|")
                payoffs_text = tds[1].get_text(separator="|").split("|")
                
                for num_str, pay_str in zip(nums_text, payoffs_text):
                    num_str = num_str.strip()
                    pay_str = pay_str.strip().replace(",", "")
                    if num_str.isdigit() and pay_str.isdigit():
                        cursor.execute("""
                        INSERT OR REPLACE INTO returns (race_id, ticket_type, horse_num, payoff)
                        VALUES (?, ?, ?, ?)
                        """, (race_id, ticket_type, int(num_str), int(pay_str)))

        self.conn.commit()
        return True

    def parse_horse(self, horse_id: str) -> bool:
        """horse_id に対応する生HTML(/horse/result/ページ)をパースしてhorse_resultsに保存する。
        netkeibaの成績テーブルは列構成が時期により変わる(2026年時点で33列)ため、
        列位置のハードコードではなくヘッダー行のth名から列位置を解決する。"""
        file_path = os.path.join(self.html_dir, "horse", f"{horse_id}.html.gz")
        if not os.path.exists(file_path):
            return False

        with gzip.open(file_path, "rt", encoding="euc-jp", errors="replace") as f:
            html = f.read()

        soup = BeautifulSoup(html, "html.parser")

        # 過去成績テーブルのパース (/horse/result/{id}/ ページの db_h_race_results)
        table = soup.find(class_="db_h_race_results")
        if not table:
            return False

        # ヘッダー名 -> 列インデックスのマップを構築する。
        # th内に補足テキストが含まれることがあるため、先頭行のみで完全一致させる。
        header_row = table.find("tr")
        if not header_row:
            return False
        header_map = {}
        for idx, th in enumerate(header_row.find_all("th")):
            name = th.get_text().strip().splitlines()[0].strip()
            if name not in header_map:
                header_map[name] = idx

        required = ["日付", "レース名", "着順", "距離", "タイム", "通過", "上り"]
        missing = [name for name in required if name not in header_map]
        if missing:
            # スキーマ検証アラート(仕様書§13): HTML構造変更を検知したら黙って続行しない
            raise ValueError(
                f"netkeiba horse result table layout changed for horse_id={horse_id}: "
                f"missing columns {missing}. Found headers: {list(header_map.keys())}"
            )

        def get_col(cols, name):
            idx = header_map.get(name)
            if idx is None or idx >= len(cols):
                return None
            return cols[idx]

        def get_text(cols, name):
            td = get_col(cols, name)
            return td.get_text().strip() if td is not None else ""

        cursor = self.conn.cursor()
        rows = table.find_all("tr")
        for row in rows[1:]: # ヘッダー飛ばす
            cols = row.find_all("td")
            if len(cols) < len(required):
                continue

            # 日付
            date_str = get_text(cols, "日付").replace("/", "-")

            # レースID
            race_td = get_col(cols, "レース名")
            race_link = race_td.find("a") if race_td is not None else None
            if not race_link:
                continue
            r_match = re.search(r"race/.*?(\d+)", race_link.get("href"))
            race_id = r_match.group(1) if r_match else ""

            # 着順
            rank = get_text(cols, "着順")

            # 距離・コース種別 (例: "芝1600", "ダ1800")
            dist_str = get_text(cols, "距離")
            dist_match = re.search(r"\d+", dist_str)
            distance = int(dist_match.group(0)) if dist_match else None
            course_match = re.search(r"(芝|ダ|障)", dist_str)
            course_type = course_match.group(1) if course_match else None

            # 開催場 (例: "3中山6" -> "中山", 地方は "佐賀" 等)
            place = None
            place_str = get_text(cols, "開催")
            if place_str:
                place_match = re.search(r"\d*([^\d]+)", place_str)
                if place_match and place_match.group(1):
                    place = place_match.group(1)

            # タイム
            time_sec = self.parse_time(get_text(cols, "タイム"))

            # 着差
            margin = self.parse_margin(get_text(cols, "着差"))

            # 通過
            passing_order = get_text(cols, "通過")

            # 上がり3F
            last_3f_str = get_text(cols, "上り")
            try:
                last_3f = float(last_3f_str) if last_3f_str and last_3f_str != "-" else None
            except ValueError:
                last_3f = None

            # 賞金(万円)
            prize = None
            prize_str = get_text(cols, "賞金").replace(",", "")
            try:
                prize = float(prize_str) if prize_str and prize_str != "-" else None
            except ValueError:
                prize = None

            cursor.execute("""
            INSERT OR REPLACE INTO horse_results (
                horse_id, date, race_id, rank, distance, course_type, place,
                time_sec, margin, last_3f, passing_order, prize
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                horse_id, date_str, race_id, rank, distance, course_type, place,
                time_sec, margin, last_3f, passing_order, prize
            ))

        self.conn.commit()
        return True

    def parse_ped(self, horse_id: str) -> bool:
        """horse_id に対応する生HTMLをパースしてpedsに保存する"""
        file_path = os.path.join(self.html_dir, "ped", f"{horse_id}.html.gz")
        if not os.path.exists(file_path):
            return False

        with gzip.open(file_path, "rt", encoding="euc-jp", errors="replace") as f:
            html = f.read()

        soup = BeautifulSoup(html, "html.parser")
        
        # 血統表テーブル
        table = soup.find(class_="blood_table")
        if not table:
            return False

        # 5代血統表から主要部分（父、母、父父、父母、母父、母母）のIDをパースする。
        # netkeibaの血統表の構造はrowspanでネストされている。
        # td内のリンクaのhrefから horse_id (あるいはpedのhorse_id) を抽出。
        # 構造上、特定のセル位置から抽出するか、特定のクラス/テキストから紐付け。
        # blood_table内のaタグを順番に取得し、ユニークな血統情報に割り当てるか、
        # あるいはセルのrowspanのパターンから決定。
        # 5代血統表の単純なアプローチとして、血統表内のすべてのリンクを抽出する。
        # リンクのhrefは "/horse/ped/(\d+)" または "/horse/(\d+)" である。
        links = table.find_all("a")
        ped_ids = []
        for l in links:
            href = l.get("href", "")
            match = re.search(r"(?:horse|ped)/.*?(\d+)", href)
            if match:
                pid = match.group(1)
                if pid != horse_id and pid not in ped_ids:
                    ped_ids.append(pid)

        # 2代前までをパース：
        # 順番に：
        # 1. 父 (Father)
        # 2. 母 (Mother)
        # 3. 父父 (F of F)
        # 4. 父母 (M of F)
        # 5. 母父 (F of M)
        # 6. 母母 (M of M)
        # rowspanによるnetkeibaのレイアウト：
        # 通常、上から順に：
        # 1列目（父・母）: 父が先、母が中ほど。
        # HTML内での出現順は深さ優先探索（DFS）またはそれに類する。
        # 例：
        # [父]
        #   [父父]
        #     [父父父] ...
        #   [父母] ...
        # [母]
        #   [母父] ...
        #   [母母] ...
        # このため、aタグの出現順はある程度規則的である。
        # より頑健にするために、BeautifulSoupの構造をパースするか、
        # netkeibaの血統表固有のインデックスを利用する。
        # 通常、td の rowspan 属性で世代が分かる。
        # rowspan=16 が1世代目(父、母)、rowspan=8 が2世代目(父父、父母、母父、母母)、
        # rowspan=4 が3世代目、rowspan=2 が4世代目、rowspan=1 が5世代目。
        # これを利用して、世代ごとに正確にマッピングできる。
        
        tds = table.find_all("td")
        ped_map = {
            "father_id": None,
            "mother_id": None,
            "father_of_father_id": None,
            "mother_of_father_id": None,
            "father_of_mother_id": None,
            "mother_of_mother_id": None
        }
        
        # 1世代目 (rowspan=16 または colspan=1 で特定位置にあるもの)
        # 2世代目 (rowspan=8)
        # 実際には、td要素を順番に走査して、そのrowspan値と出現順で判定する。
        gen1 = [] # rowspan=16
        gen2 = [] # rowspan=8
        
        for td in tds:
            rowspan = td.get("rowspan")
            if rowspan:
                rowspan = int(rowspan)
                a = td.find("a")
                if a:
                    href = a.get("href", "")
                    match = re.search(r"/(?:horse|horse/ped)/(\d+)", href)
                    if match:
                        pid = match.group(1)
                        if rowspan == 16 or rowspan == 32: # netkeibaの構造によっては32になることもある
                            gen1.append(pid)
                        elif rowspan == 8 or rowspan == 16:
                            # 32/16の場合もあるので、数で判断するか、順序で
                            pass
        
        # netkeibaの血統表は、tdの階層を素直に取得するのが難しい場合があるため、
        # 最も安全なのは、順番に現れるリンクのインデックスを利用すること。
        # netkeibaの5代血統表の構造における出現順は一定である。
        # 1番目: 父
        # 2番目: 父父
        # 3番目: 父父父
        # 4番目: 父父父父 ...
        # このように、血統構造は木構造の先行順巡回（Pre-order traversal）でHTMLに配置される。
        # 従って、2代前までに対応するインデックスを決定できる。
        # ただし、馬によって一部の親が不明でリンクがない場合がある。
        # そのため、td要素をパースして各セルに割り振るのが確実。
        
        # 簡易的かつ頑健な方法として、tdの階層的な走査を行う：
        # netkeibaの血統表は tr が32行ある。
        # 各 tr の td は左から順に第1世代、第2世代、... と並ぶ。
        # 行 0: 第1世代(父), 第2世代(父父), 第3世代(父父父), 第4世代...
        # 行 8: 父母 ...
        # 行 16: 母, 母父 ...
        # 行 24: 母母 ...
        # したがって、各行の最初のtdがどの世代か（rowspanで決定）を追跡することで、
        # 各位置の馬IDを取得できる。
        
        rows = table.find_all("tr")
        grid = {} # (row, col) -> horse_id
        
        # rowspan と colspan を考慮してグリッドを構築する
        current_row_td_idx = [0] * len(rows)
        for r_idx, row in enumerate(rows):
            tds_in_row = row.find_all("td")
            c_idx = 0
            for td in tds_in_row:
                # すでに上の行のrowspanで埋まっている列をスキップする
                while (r_idx, c_idx) in grid:
                    c_idx += 1
                    
                rowspan = int(td.get("rowspan", 1))
                colspan = int(td.get("colspan", 1))
                
                # 馬IDを抽出
                a = td.find("a")
                val = None
                if a:
                    href = a.get("href", "")
                    match = re.search(r"(?:horse|ped)/.*?(\d+)", href)
                    if match:
                        val = match.group(1)
                
                # グリッドのセルを埋める
                for dr in range(rowspan):
                    for dc in range(colspan):
                        grid[(r_idx + dr, c_idx + dc)] = val
                c_idx += colspan
                
        # 抽出したい位置の座標 (行, 列)
        # 列0: 1世代目 (父: 0-15行, 母: 16-31行)
        # 列1: 2世代目 (父父: 0-7行, 父母: 8-15行, 母父: 16-23行, 母母: 24-31行)
        ped_map["father_id"] = grid.get((0, 0))
        ped_map["mother_id"] = grid.get((16, 0))
        ped_map["father_of_father_id"] = grid.get((0, 1))
        ped_map["mother_of_father_id"] = grid.get((8, 1))
        ped_map["father_of_mother_id"] = grid.get((16, 1))
        ped_map["mother_of_mother_id"] = grid.get((24, 1))

        cursor = self.conn.cursor()
        cursor.execute("""
        INSERT OR REPLACE INTO peds (
            horse_id, father_id, mother_id, father_of_father_id, mother_of_father_id, father_of_mother_id, mother_of_mother_id
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            horse_id,
            ped_map["father_id"],
            ped_map["mother_id"],
            ped_map["father_of_father_id"],
            ped_map["mother_of_father_id"],
            ped_map["father_of_mother_id"],
            ped_map["mother_of_mother_id"]
        ))
        
        self.conn.commit()
        return True

    def parse_shutuba(self, race_id: str) -> bool:
        """race_id に対応する出馬表HTMLをパースしてraces, resultsに保存する (結果系カラムはNULL)。
        出馬表(race.netkeiba.com)はUTF-8で保存されている点に注意(dbページはEUC-JP)。"""
        file_path = os.path.join(self.html_dir, "shutuba", f"{race_id}.html.gz")
        if not os.path.exists(file_path):
            return False

        with gzip.open(file_path, "rt", encoding="utf-8", errors="replace") as f:
            html = f.read()

        soup = BeautifulSoup(html, "html.parser")
        
        # 1. レース名
        race_name = ""
        rname_div = soup.find(class_="RaceName")
        if rname_div:
            race_name = rname_div.get_text().strip()
            
        # 2. レースデータ
        course_type = ""
        distance = 0
        direction = ""
        weather = ""
        surface_state = ""
        date_str = ""
        place = ""
        
        if len(race_id) == 12:
            year = race_id[:4]
            place_map = {
                "01": "札幌", "02": "函館", "03": "福島", "04": "新潟", "05": "東京",
                "06": "中山", "07": "中京", "08": "京都", "09": "阪神", "10": "小倉"
            }
            place = place_map.get(race_id[4:6], "不明")

            # 開催日の取得。優先順:
            # 1) <title> 内の「YYYY年M月D日」表記
            # 2) ページ内リンクの kaisai_date=YYYYMMDD の最頻値
            # (旧実装の class="Active" は現行ページではタブ名になっており日付を含まない)
            title_tag = soup.find("title")
            if title_tag:
                t_match = re.search(r"(\d{4})年(\d{1,2})月(\d{1,2})日", title_tag.get_text())
                if t_match:
                    date_str = f"{t_match.group(1)}-{int(t_match.group(2)):02d}-{int(t_match.group(3)):02d}"

            if not date_str:
                from collections import Counter
                kd_counts = Counter(re.findall(r"kaisai_date=(\d{8})", html))
                if kd_counts:
                    kd = kd_counts.most_common(1)[0][0]
                    date_str = f"{kd[:4]}-{kd[4:6]}-{kd[6:8]}"

            if not date_str:
                # 開催日が取得できない場合、誤った日付(例:1月1日)で登録すると
                # merge_asofによる特徴量結合がすべて「未来」扱いとなり、
                # 過去成績がサイレントに空になったまま予測が実行されてしまう。
                # 検知できるよう確実に失敗させる。
                raise ValueError(
                    f"Failed to parse race date from shutuba page for race_id={race_id}. "
                    "Refusing to fall back to a default date."
                )

        rdata01 = soup.find(class_="RaceData01")
        if rdata01:
            info_text = rdata01.get_text()
            course_match = re.search(r"(芝|ダ|障)([左右直内外]*?)(\d+)m", info_text)
            if course_match:
                course_type = course_match.group(1)
                direction = course_match.group(2)
                distance = int(course_match.group(3))
                
            weather_match = re.search(r"天候:([^\s/]+)", info_text)
            if weather_match:
                weather = weather_match.group(1)

            state_match = re.search(r"馬場:([^\s/]+)", info_text)
            if state_match:
                surface_state = state_match.group(1)

        race_class = "その他"
        if "G1" in race_name:
            race_class = "G1"
        elif "G2" in race_name:
            race_class = "G2"
        elif "G3" in race_name:
            race_class = "G3"
        elif "オープン" in race_name or "(L)" in race_name or "OP" in race_name:
            race_class = "オープン"
        elif "3勝" in race_name:
            race_class = "3勝クラス"
        elif "2勝" in race_name:
            race_class = "2勝クラス"
        elif "1勝" in race_name:
            race_class = "1勝クラス"
        elif "未勝利" in race_name:
            race_class = "未勝利"
        elif "新馬" in race_name:
            race_class = "新馬"

        cursor = self.conn.cursor()
        cursor.execute("""
        INSERT OR REPLACE INTO races (race_id, race_name, date, course_type, distance, direction, weather, surface_state, race_class, place)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (race_id, race_name, date_str, course_type, distance, direction, weather, surface_state, race_class, place))

        # 出走馬テーブルのパース
        table = soup.find(class_="Shutuba_Table")
        if table:
            rows = table.find_all("tr", class_="HorseList")
            for row in rows:
                waku_td = row.find(class_=re.compile(r"Waku\d+"))
                bracket_num = int(waku_td.get_text().strip()) if waku_td else None
                
                umaban_td = row.find(class_=re.compile(r"Umaban\d+"))
                horse_num = int(umaban_td.get_text().strip()) if umaban_td else None
                
                horse_td = row.find(class_="HorseInfo")
                horse_id = ""
                if horse_td:
                    horse_link = horse_td.find("a")
                    if horse_link:
                        h_match = re.search(r"/horse/(\d+)", horse_link.get("href"))
                        if h_match:
                            horse_id = h_match.group(1)
                            
                if not horse_id:
                    continue
                    
                txt_c_tds = row.find_all("td", class_="Txt_C")
                gender = ""
                age = None
                weight_carrier = None
                if len(txt_c_tds) >= 2:
                    gender_age = txt_c_tds[0].get_text().strip()
                    if gender_age:
                        gender = gender_age[0]
                        if len(gender_age) > 1 and gender_age[1:].isdigit():
                            age = int(gender_age[1:])
                    try:
                        weight_carrier = float(txt_c_tds[1].get_text().strip())
                    except ValueError:
                        pass
                
                # 騎手・調教師リンクは "/jockey/01091/" と "/jockey/result/recent/01091/" の
                # 両形式が存在するため、パス中の末尾数値を許容する形で抽出する
                jockey_td = row.find(class_="Jockey")
                jockey_id = ""
                if jockey_td:
                    jockey_link = jockey_td.find("a")
                    if jockey_link:
                        j_match = re.search(r"/jockey/(?:[a-z/]+/)?(\d+)", jockey_link.get("href"))
                        if j_match:
                            jockey_id = j_match.group(1)

                trainer_td = row.find(class_="Trainer")
                trainer_id = ""
                if trainer_td:
                    trainer_link = trainer_td.find("a")
                    if trainer_link:
                        t_match = re.search(r"/trainer/(?:[a-z/]+/)?(\d+)", trainer_link.get("href"))
                        if t_match:
                            trainer_id = t_match.group(1)

                weight_td = row.find(class_="Weight")
                horse_weight = None
                weight_diff = None
                if weight_td:
                    weight_str = weight_td.get_text().strip()
                    w_match = re.search(r"(\d+)\(([-+]*\d+)\)", weight_str)
                    if w_match:
                        horse_weight = float(w_match.group(1))
                        weight_diff = float(w_match.group(2))
                    else:
                        w_match_only = re.search(r"(\d+)", weight_str)
                        if w_match_only:
                            horse_weight = float(w_match_only.group(1))
                            
                odds_td = row.find(class_="Odds")
                tansho_odds = None
                if odds_td:
                    try:
                        tansho_odds = float(odds_td.get_text().strip())
                    except ValueError:
                        pass
                        
                pop_td = row.find(class_="Popular")
                popularity = None
                if pop_td:
                    pop_str = pop_td.get_text().strip()
                    if pop_str.isdigit():
                        popularity = int(pop_str)

                cursor.execute("""
                INSERT OR REPLACE INTO results (
                    race_id, horse_id, rank, bracket_num, horse_num, gender, age, weight_carrier,
                    horse_weight, weight_diff, jockey_id, trainer_id, time_sec, margin, last_3f,
                    passing_order, tansho_odds, popularity
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    race_id, horse_id, None, bracket_num, horse_num, gender, age, weight_carrier,
                    horse_weight, weight_diff, jockey_id, trainer_id, None, None, None,
                    None, tansho_odds, popularity
                ))

        self.conn.commit()
        return True
