import os
import pandas as pd
import numpy as np
import sqlite3
import yaml

# 複勝率の縮小推定に使う事前分布。JRAの平均頭数(約14頭)での複勝圏の基準率 3/14 ≒ 0.21 を、10走分の重みで与える
PRIOR_TOP3_RATE = 0.21
PRIOR_WEIGHT = 10

# スピード指数: 基準タイムを作るのに必要な過去の勝ち馬数 / 馬場差を推定するのに必要な当日のレース数
MIN_PAR_RACES = 5
MIN_VARIANT_RACES = 3

# JRAの場コード(race_idの5-6桁目)
JRA_PLACE_CODES = [f"{i:02d}" for i in range(1, 11)]

# クラスの序列(相手関係の強さの近似)。"その他"(地方・海外など)は欠損とする
CLASS_LEVELS = {"新馬": 0, "未勝利": 0, "1勝クラス": 1, "2勝クラス": 2, "3勝クラス": 3,
                "オープン": 4, "G3": 5, "G2": 6, "G1": 7}

class FeatureBuilder:
    def __init__(self, config_path="config/config.yaml"):
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)
            
        base_dir = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        db_path_rel = self.config["paths"]["db_path"]
        self.db_path = os.path.join(base_dir, db_path_rel)
        self.features_dir = os.path.join(base_dir, self.config["paths"]["features_dir"])
        os.makedirs(self.features_dir, exist_ok=True)

    def _load_data(self):
        conn = sqlite3.connect(self.db_path)
        
        # races, results, peds を読み込む
        df_races = pd.read_sql_query("SELECT * FROM races", conn)
        df_results = pd.read_sql_query("SELECT * FROM results", conn)
        df_peds = pd.read_sql_query("SELECT * FROM peds", conn)
        df_horse_results = pd.read_sql_query("SELECT * FROM horse_results", conn)
        
        conn.close()
        
        # 日付変換
        df_races["date"] = pd.to_datetime(df_races["date"])
        df_horse_results["date"] = pd.to_datetime(df_horse_results["date"])
        
        return df_races, df_results, df_peds, df_horse_results

    def _parse_passing_order(self, passing_str):
        """通過順位文字列から1角と4角での位置取りを抽出する (例: "4-4-3-2" -> 4, 2)"""
        if not passing_str or pd.isna(passing_str) or passing_str == "-":
            return None, None
        parts = str(passing_str).split("-")
        if not parts:
            return None, None
        try:
            first_corner = float(parts[0])
            last_corner = float(parts[-1])
            return first_corner, last_corner
        except ValueError:
            return None, None

    def _determine_run_style(self, df_horse_results, df_races):
        """過去の通過順位から馬の脚質を決定する (逃げ/先行/差し/追込)"""
        # レース情報と結合して頭数(n_horses)を取得したいが、
        # horse_resultsには頭数がないため、出走馬数の代用として、
        # 同一race_idの出走数を results からカウントするか、簡易的に順位そのもので判定する。
        # ここでは results テーブルから各レースの頭数を取得しマージする。
        df_hr = df_horse_results.copy()
        
        # 1角と4角の位置を抽出
        first_corners = []
        last_corners = []
        for po in df_hr["passing_order"]:
            f, l = self._parse_passing_order(po)
            first_corners.append(f)
            last_corners.append(l)
        df_hr["first_corner"] = first_corners
        df_hr["last_corner"] = last_corners
        
        # 脚質の判定 (簡易版)
        # 1角での位置が1-2番手: 逃げ/先行
        # 各馬の過去走の平均位置から判定
        # 未来情報リーク防止のため、ここでは各「馬ID、日付」時点での過去成績から決定する必要があるが、
        # 処理を高速化するため、以下のように「過去走のみ」を集計する。
        return df_hr

    @staticmethod
    def _parse_rank(rank: pd.Series) -> pd.Series:
        """着順文字列を数値化する。降着・再騎乗(例: "4(降)", "9(再)")は先頭の数字を確定着順として扱い、
        "中止"・"失格"などの数字で始まらないものはNaNとする"""
        return pd.to_numeric(rank.astype(str).str.extract(r"^(\d+)")[0], errors="coerce")

    def _entity_top3_rate(self, df, keys, window):
        """keys(騎手ID、種牡馬ID×コース種別など)ごとの、対象日より前(当日を含まない)の複勝率を返す。
        window="365D" なら直近365日、None なら全期間。出馬表由来の結果未確定行は分母・分子から除外する。
        出走数が少ないと率のばらつきが大きいため、事前分布(PRIOR_TOP3_RATE を PRIOR_WEIGHT 走分)で縮小推定する。
        キーが欠損(NaN・空文字)の行は事前分布の値になる。"""
        d = df[keys + ["date", "is_top3", "is_unresolved"]]
        valid = d[keys].notna().all(axis=1) & (d[keys].astype(str) != "").all(axis=1)
        d = d[valid]

        all_keys = d[keys + ["date"]].drop_duplicates()
        daily = d[~d["is_unresolved"]].groupby(keys + ["date"])["is_top3"].agg(["sum", "count"]).reset_index()
        g = all_keys.merge(daily, on=keys + ["date"], how="left").fillna({"sum": 0, "count": 0})
        g = g.sort_values(keys + ["date"]).reset_index(drop=True)

        if window is None:
            grp = g.groupby(keys, sort=False)
            g["n_past"] = grp["count"].cumsum() - g["count"]
            g["top3_past"] = grp["sum"].cumsum() - g["sum"]
        else:
            # closed="left": 集計対象日(当日)自身を含まないウィンドウ(リーク防止)
            rolled = g.set_index("date").groupby(keys, sort=False)[["sum", "count"]].rolling(window, closed="left").sum()
            g["n_past"] = rolled["count"].fillna(0).values
            g["top3_past"] = rolled["sum"].fillna(0).values

        g["rate"] = (g["top3_past"] + PRIOR_WEIGHT * PRIOR_TOP3_RATE) / (g["n_past"] + PRIOR_WEIGHT)
        rate = df[keys + ["date"]].merge(g[keys + ["date", "rate"]], on=keys + ["date"], how="left")["rate"]
        return rate.fillna(PRIOR_TOP3_RATE).values

    @staticmethod
    def _mean_before(df, keys, value_col):
        """keys ごとに、各日付より前(当日を含まない)の value_col の平均と件数を返す (keys + date + mean + count)"""
        daily = df.groupby(keys + ["date"])[value_col].agg(["sum", "count"]).reset_index()
        daily = daily.sort_values(keys + ["date"]).reset_index(drop=True)
        grp = daily.groupby(keys, sort=False)
        count = grp["count"].cumsum() - daily["count"]
        daily["mean"] = (grp["sum"].cumsum() - daily["sum"]) / count.where(count > 0)
        daily["count"] = count
        return daily[keys + ["date", "mean", "count"]]

    def _speed_figures(self, df):
        """各出走のスピード指数(1000mあたり秒。大きいほど速い)を返す (horse_id, race_id, speed_fig)。
        - 基準タイム: 同じ場・コース種別・距離の、その日より前の勝ち馬タイムの平均(MIN_PAR_RACES頭未満なら欠損)
        - クラス補正: クラスごとの、その日より前の勝ち馬の基準タイムとの差の平均(履歴が無ければ0)
        - 馬場差: 当日・同じ場・同じコース種別の勝ち馬の(基準タイムとの差 - クラス補正)の平均
          (MIN_VARIANT_RACES レース未満なら補正しない)
        - 指数 = -(基準タイムとの差 - 馬場差)
        いずれも対象レースの日付以前の情報のみで計算する。当日の他レースは、その走りが過去走として参照される時点では過去である。"""
        keys = ["place", "course_type", "distance"]
        d = df.loc[df["time_sec"] > 0, ["horse_id", "race_id", "date", "race_class", "time_sec", "rank_num"] + keys].copy()
        d["distance"] = d["distance"].astype(float)

        par = self._mean_before(d[d["rank_num"] == 1], keys, "time_sec")
        par.loc[par["count"] < MIN_PAR_RACES, "mean"] = np.nan
        d = d.merge(par[keys + ["date", "mean"]].rename(columns={"mean": "par"}), on=keys + ["date"], how="left")
        d["dev"] = (d["time_sec"] - d["par"]) / d["distance"] * 1000

        class_offset = self._mean_before(d[d["rank_num"] == 1], ["race_class"], "dev")
        d = d.merge(class_offset[["race_class", "date", "mean"]].rename(columns={"mean": "class_offset"}),
                    on=["race_class", "date"], how="left")
        d["class_offset"] = d["class_offset"].fillna(0)

        winners = d[d["rank_num"] == 1].assign(resid=lambda x: x["dev"] - x["class_offset"])
        variant = winners.groupby(["date", "place", "course_type"])["resid"].agg(["mean", "count"]).reset_index()
        variant["variant"] = variant["mean"].where(variant["count"] >= MIN_VARIANT_RACES, 0.0)
        d = d.merge(variant[["date", "place", "course_type", "variant"]], on=["date", "place", "course_type"], how="left")
        d["speed_fig"] = -(d["dev"] - d["variant"].fillna(0))
        return d[["horse_id", "race_id", "speed_fig"]]

    def _distance_band(self, distance: pd.Series) -> pd.Series:
        """距離を距離帯カテゴリに変換する (短距離<=1400, マイル<=1800, 中距離<=2200, 長距離>2200)"""
        return pd.cut(
            distance,
            bins=[-np.inf, 1400, 1800, 2200, np.inf],
            labels=["短距離", "マイル", "中距離", "長距離"]
        )

    def _asof_condition_feature(self, df, df_hr, by_cols, src_col, feat_name):
        """horse_id (+ 追加条件カラム) が一致する過去走に対して、
        「対象レースより厳密に前(allow_exact_matches=False)」の直近の値までの
        拡大平均(expanding mean)をmerge_asofで結合するヘルパー。
        学習時(horse_resultsに当該レース自身の行が存在する)・予測時(存在しない)の
        いずれでも同じ「直近の過去走まで」の値を参照するため、学習と予測での特徴量ズレが生じない。"""
        cols = ["horse_id"] + by_cols + ["date", src_col]
        d_hr = df_hr[cols].copy()
        # dfとhorse_id/date/条件列の型を合わせる(merge_asofは両側の型一致を要求する)
        d_hr["horse_id"] = d_hr["horse_id"].astype(str)
        d_hr["date"] = pd.to_datetime(d_hr["date"]).astype("datetime64[ns]")

        d_target = df[["horse_id"] + by_cols + ["date"]].copy()
        d_target["horse_id"] = d_target["horse_id"].astype(str)
        d_target["date"] = pd.to_datetime(d_target["date"]).astype("datetime64[ns]")
        for col in by_cols:
            if not isinstance(d_target[col].dtype, pd.CategoricalDtype):
                d_target[col] = d_target[col].astype(str)
                d_hr[col] = d_hr[col].astype(str)

        d_hr = d_hr.sort_values(["horse_id"] + by_cols + ["date"])
        grp = d_hr.groupby(["horse_id"] + by_cols)[src_col]
        d_hr[feat_name] = grp.expanding(min_periods=1).mean().reset_index(level=list(range(len(by_cols) + 1)), drop=True)

        # merge_asofはby=指定時でもonキー(date)のグローバルソートを要求する。
        # 行順序が並べ替えられても元のdfの行順に正しく戻せるよう行IDを保持する
        d_hr = d_hr.sort_values("date", kind="stable")
        d_target["_row_id"] = np.arange(len(d_target))
        d_target = d_target.sort_values("date", kind="stable")

        merged = pd.merge_asof(
            d_target,
            d_hr[["horse_id"] + by_cols + ["date", feat_name]],
            on="date",
            by=["horse_id"] + by_cols,
            direction="backward",
            allow_exact_matches=False
        )
        merged = merged.sort_values("_row_id")
        return merged[feat_name].values

    def build_features(self, race_ids=None):
        """特徴量を生成し、年ごとにParquetへ保存して返す。
        race_ids を指定した場合(当日予測用)は、そのレースの行だけを返しParquetは保存しない。
        騎手・調教師の成績は全データから計算し、馬ごとの過去走集計は対象レースの出走馬に限定して計算する
        (集計値は全件生成時と同一になる)。"""
        df_races, df_results, df_peds, df_horse_results = self._load_data()

        # 基礎マージ
        df = df_results.merge(df_races, on="race_id", how="inner")

        # 仕様書§2: 対象はJRA平地のみ。race_idの5-6桁目(場コード)が01〜10以外(地方・海外)と障害レースを除外する
        is_jra = df["race_id"].astype(str).str[4:6].isin(JRA_PLACE_CODES)
        df = df[is_jra & (df["course_type"] != "障")]
        # 出走取消・競走除外の馬は出走していないため除外する(負例として学習やレース内確率の正規化に混入させない)
        df = df[~df["rank"].isin(["取消", "除外"])].reset_index(drop=True)

        # 頭数を計算してマージ
        df_n_horses = df.groupby("race_id").size().reset_index(name="n_horses")
        df = df.merge(df_n_horses, on="race_id", how="inner")

        # 着順の数値化
        # 「1」や「2」は数値化、「計不」や「中止」は NaN
        df["rank_num"] = self._parse_rank(df["rank"])
        # 3着以内フラグ (ベースライン二値分類用)
        df["is_top3"] = (df["rank_num"] <= 3).astype(int)
        df.loc[df["rank_num"].isna(), "is_top3"] = 0 # 中止などは3着以内ではない

        # 出馬表由来(結果未確定)の行かどうかのフラグ。
        # parse_shutuba は rank=NULL で results に行を挿入するため、この行は
        # 「全馬凡走」の負例として学習に混入させてはならない(下流の学習/シミュレーション側で除外する)。
        df["is_unresolved"] = df["rank"].isna()

        # ラベル用スコア (LambdaRank用: 1着=3, 2着=2, 3着=1, その他=0)
        df["rank_label"] = 0
        df.loc[df["rank_num"] == 1, "rank_label"] = 3
        df.loc[df["rank_num"] == 2, "rank_label"] = 2
        df.loc[df["rank_num"] == 3, "rank_label"] = 1

        # 1着タイムを基準にしたタイム差の計算
        # 中止などを除外した最小タイム
        df_valid_time = df[df["time_sec"] > 0]
        df_min_time = df_valid_time.groupby("race_id")["time_sec"].min().reset_index(name="min_time")
        df = df.merge(df_min_time, on="race_id", how="left")
        df["time_diff"] = df["time_sec"] - df["min_time"]

        # ソート（時系列順）
        df = df.sort_values(by=["date", "race_id", "horse_num"]).reset_index(drop=True)

        # --- 騎手・調教師の直近1年複勝率 (リーク防止) ---
        df["jockey_top3_rate"] = self._entity_top3_rate(df, ["jockey_id"], "365D")
        df["trainer_top3_rate"] = self._entity_top3_rate(df, ["trainer_id"], "365D")

        # --- 血統マージ ---
        df = df.merge(df_peds, on="horse_id", how="left")

        # --- 種牡馬の産駒成績 (同コース種別・同距離帯での、対象日より前の全期間の複勝率) ---
        # 父IDを高カーディナリティのカテゴリ変数として直接学習させると過学習しやすいため、統計量として与える
        df["distance_band"] = self._distance_band(df["distance"])
        df["_distance_band_key"] = df["distance_band"].astype(str)
        df["sire_top3_rate_course"] = self._entity_top3_rate(df, ["father_id", "course_type"], None)
        df["sire_top3_rate_distance"] = self._entity_top3_rate(df, ["father_id", "_distance_band_key"], None)
        df = df.drop(columns=["_distance_band_key"])

        # --- スピード指数 (全レースから計算し、下で馬の過去走に結合する) ---
        df_speed = self._speed_figures(df)

        # --- 馬の過去成績集計 (リーク防止) ---
        # horse_results から集計する。
        # horse_results には過去走の情報が格納されている。
        # 各馬について、レース日時点での過去3走、5走、全走の集計値を計算する。
        if race_ids is not None:
            df = df[df["race_id"].isin(race_ids)].reset_index(drop=True)
            df_horse_results = df_horse_results[df_horse_results["horse_id"].isin(df["horse_id"])]

        df_hr = df_horse_results.copy()
        # 頭数・クラス列が無い旧DB(reparse前)でも動作させる。ただし相対着順・脚質・クラス特徴量は欠損になる
        for col in ["n_horses", "race_class"]:
            if col not in df_hr.columns:
                df_hr[col] = np.nan
        if len(df_hr) > 0 and df_hr["n_horses"].isna().all():
            print("Warning: horse_results.n_horses is empty. Run `python main.py reparse` to re-parse saved HTML.")
        df_hr["rank_num"] = self._parse_rank(df_hr["rank"])

        # 1角と4角の位置を抽出
        first_corners = []
        last_corners = []
        for po in df_hr["passing_order"]:
            f, l = self._parse_passing_order(po)
            first_corners.append(f)
            last_corners.append(l)
        df_hr["first_corner"] = first_corners
        df_hr["last_corner"] = last_corners

        # 距離帯 (条件別集計用)
        df_hr["distance_band"] = self._distance_band(df_hr["distance"])

        # タイム差の近似計算：
        # horse_resultsには自分のタイムしかないため、タイムそのものの推移、
        # または results と結合して正確な time_diff を取得する。
        # results に存在する過去レースについては正確な time_diff を使用し、
        # 存在しないものは NaN とする。
        df_valid_results = df_results.merge(df_races[["race_id", "date"]], on="race_id", how="inner")
        # 1着タイムマージ
        df_valid_results = df_valid_results.merge(df_min_time, on="race_id", how="left")
        df_valid_results["time_diff"] = df_valid_results["time_sec"] - df_valid_results["min_time"]

        # horse_results と results を結合して time_diff を取得
        df_hr = df_hr.merge(
            df_valid_results[["horse_id", "date", "time_diff"]],
            on=["horse_id", "date"],
            how="left"
        )

        df_hr = df_hr.merge(df_speed, on=["horse_id", "race_id"], how="left")

        df_hr = df_hr.sort_values(["horse_id", "date"])

        # 頭数で相対化した着順(1着=1/頭数〜最下位=1)と1角位置(先頭=0〜最後方=1)
        n_horses_hr = df_hr["n_horses"].astype(float)
        df_hr["rank_ratio"] = df_hr["rank_num"] / n_horses_hr
        df_hr["first_corner_pos"] = (df_hr["first_corner"] - 1) / (n_horses_hr - 1).where(n_horses_hr > 1)
        df_hr["class_level"] = df_hr["race_class"].map(CLASS_LEVELS)

        # 前走間隔は地方・障害を含む全出走から求める
        df_hr_all = df_hr[["horse_id", "date"]].copy()

        # 成績の集計はJRA平地の過去走のみで行う(地方・海外・障害の着順やタイムは中央平地と同じ尺度ではない)
        is_jra_flat_hr = (df_hr["race_id"].astype(str).str[4:6].isin(JRA_PLACE_CODES)
                          & df_hr["course_type"].isin(["芝", "ダ"]))
        df_hr = df_hr[is_jra_flat_hr]

        # 各馬の過去走集計
        # horse_results には対象レース自身の行が(学習時は)存在しうるため、
        # merge_asof(allow_exact_matches=False) で「対象レースより厳密に前」の
        # 過去走のみを参照する。この方式では、学習時(horse_resultsに当該レースの行が
        # 存在する)でも予測時(まだ走っていないので行が存在しない)でも常に
        # 「直近の過去走までの集計値」が一貫して得られるため、
        # 旧実装(shift(1)+allow_exact_matches=True)で発生していた
        # 「予測時にのみ直近1走分の情報が欠落する」学習・予測間のズレを解消する。
        features_to_calc = {
            "rank_mean_3": ("rank_num", 3, "mean"),
            "rank_mean_5": ("rank_num", 5, "mean"),
            "rank_mean_all": ("rank_num", None, "mean"),
            "rank_ratio_mean_3": ("rank_ratio", 3, "mean"),
            "rank_ratio_mean_5": ("rank_ratio", 5, "mean"),
            "rank_ratio_mean_all": ("rank_ratio", None, "mean"),
            "class_level_mean_5": ("class_level", 5, "mean"),
            "class_level_last": ("class_level", 1, "mean"),
            "prize_sum_3": ("prize", 3, "sum"),
            "prize_sum_5": ("prize", 5, "sum"),
            "prize_sum_all": ("prize", None, "sum"),
            "margin_mean_3": ("margin", 3, "mean"),
            "margin_mean_5": ("margin", 5, "mean"),
            "last_3f_mean_3": ("last_3f", 3, "mean"),
            "last_3f_mean_5": ("last_3f", 5, "mean"),
            "time_diff_mean_3": ("time_diff", 3, "mean"),
            "time_diff_mean_5": ("time_diff", 5, "mean"),
            "speed_fig_mean_3": ("speed_fig", 3, "mean"),
            "speed_fig_mean_5": ("speed_fig", 5, "mean"),
            "speed_fig_max_5": ("speed_fig", 5, "max"),
            # 脚質分類用の1角・4角平均位置
            "first_corner_mean_5": ("first_corner", 5, "mean"),
            "last_corner_mean_5": ("last_corner", 5, "mean"),
            "first_corner_pos_mean_5": ("first_corner_pos", 5, "mean"),
        }

        df_hr_feats = df_hr[["horse_id", "date"]].copy()

        for feat_name, (src_col, window, agg_func) in features_to_calc.items():
            grp = df_hr.groupby("horse_id")[src_col]
            if window is not None:
                # rolling: min_periods=1 で過去1走しかなくても計算する
                roll = grp.rolling(window, min_periods=1)
                calc_val = getattr(roll, agg_func)().reset_index(0, drop=True)
            else:
                # expanding
                exp = grp.expanding(min_periods=1)
                calc_val = getattr(exp, agg_func)().reset_index(0, drop=True)

            df_hr_feats[feat_name] = calc_val

        # 直近の過去走(全出走)の日付を保持しておき、interval_daysの計算に用いる
        df_hr_all["hr_date"] = df_hr_all["date"]

        # df とマージ (merge_asof を使用)
        # 注意: merge_asofはby=指定時でも「onキー(date)がフレーム全体でソート済み」であることを
        # 要求するため、必ずdate単独でグローバルにソートする(horse_id優先ソートでは複数開催日で失敗する)
        df["horse_id"] = df["horse_id"].astype(str)
        df["date"] = pd.to_datetime(df["date"]).astype("datetime64[ns]")
        df = df.sort_values("date", kind="stable")
        for frame in (df_hr_feats, df_hr_all):
            frame["horse_id"] = frame["horse_id"].astype(str)
            frame["date"] = pd.to_datetime(frame["date"]).astype("datetime64[ns]")
        df_hr_feats = df_hr_feats.sort_values("date", kind="stable")
        df_hr_all = df_hr_all.sort_values("date", kind="stable")

        # merge_asof で「dateより厳密に前(allow_exact_matches=False)」の直近の過去走集計を結合する。
        df = pd.merge_asof(
            df,
            df_hr_feats,
            on="date",
            by="horse_id",
            direction="backward",
            allow_exact_matches=False
        )
        df = pd.merge_asof(
            df,
            df_hr_all,
            on="date",
            by="horse_id",
            direction="backward",
            allow_exact_matches=False
        )

        # 前走からの間隔（日数）: 結合された直近過去走の日付との差分から計算する
        df["interval_days"] = (df["date"] - df["hr_date"]).dt.days
        df = df.drop(columns=["hr_date"])

        # 前走からのクラス変化(昇級>0、降級<0)。相手関係の変化の近似
        df["class_diff"] = df["race_class"].map(CLASS_LEVELS) - df["class_level_last"]

        # --- 条件別集計 (同距離帯・同コース種別・同開催場での過去の相対着順の平均) ---
        df["rank_ratio_mean_by_distance_band"] = self._asof_condition_feature(
            df, df_hr, ["distance_band"], "rank_ratio", "rank_ratio_mean_by_distance_band"
        )
        df["rank_ratio_mean_by_course_type"] = self._asof_condition_feature(
            df, df_hr, ["course_type"], "rank_ratio", "rank_ratio_mean_by_course_type"
        )
        df["rank_ratio_mean_by_place"] = self._asof_condition_feature(
            df, df_hr, ["place"], "rank_ratio", "rank_ratio_mean_by_place"
        )

        # 脚質(run_style)の決定: 過去5走の1角の相対位置(先頭=0〜最後方=1)の平均から分類する。
        # 閾値は平均的な頭数(約14頭)で 逃げ≒平均1.9番手以内, 先行≒4.9番手以内, 差し≒9.5番手以内 に相当する。
        # 過去走が無い(または頭数・通過順が取れない)馬は "不明" とする
        pos = df["first_corner_pos_mean_5"]
        df["run_style"] = np.select(
            [pos <= 0.07, pos <= 0.3, pos <= 0.65, pos > 0.65],
            ["逃げ", "先行", "差し", "追込"],
            default="不明"
        )
        df["run_style"] = df["run_style"].astype("category")

        # 脚質×コース属性(開催場)の交互作用特徴量
        df["run_style_place"] = (df["run_style"].astype(str) + "_" + df["place"].astype(str)).astype("category")

        # 血統カラムのカテゴリ化
        ped_cols = ["father_id", "mother_id", "father_of_father_id", "mother_of_father_id", "father_of_mother_id", "mother_of_mother_id"]
        for col in ped_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).replace("None", np.nan).astype("category")

        # --- その他のカテゴリ・数値変数の処理 ---
        df["gender"] = df["gender"].astype("category")
        df["course_type"] = df["course_type"].astype("category")
        df["direction"] = df["direction"].astype("category")
        df["weather"] = df["weather"].astype("category")
        df["surface_state"] = df["surface_state"].astype("category")
        df["race_class"] = df["race_class"].astype("category")
        df["place"] = df["place"].astype("category")
        df["distance_band"] = df["distance_band"].astype("category")

        # 欠損値の処理 (LightGBMはNaNを許容するのでそのまま残す)

        # 保存用に年を取得
        df["year"] = df["date"].dt.year

        if race_ids is not None:
            return df
        
        # Parquet形式で年ごとに保存
        # 特徴量ディレクトリに保存
        for year, group in df.groupby("year"):
            year_dir = os.path.join(self.features_dir, f"year={year}")
            os.makedirs(year_dir, exist_ok=True)
            file_path = os.path.join(year_dir, "data.parquet")
            
            # カテゴリ列を適切に保持してParquetに保存
            group.to_parquet(file_path, index=False)
            
        print(f"Features created and saved to {self.features_dir}")
        return df

if __name__ == "__main__":
    builder = FeatureBuilder()
    builder.build_features()
