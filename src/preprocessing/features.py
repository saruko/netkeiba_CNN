import os
import pandas as pd
import numpy as np
import sqlite3
import yaml

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
        is_jra = df["race_id"].astype(str).str[4:6].isin([f"{i:02d}" for i in range(1, 11)])
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
        # 騎手複勝率の計算
        df["jockey_top3_rate"] = 0.3 # デフォルト値
        df["trainer_top3_rate"] = 0.3 # デフォルト値

        # 日付でソートした結果から、過去365日の成績をローリング集計する。
        # 出馬表由来の結果未確定行(is_unresolved)は集計対象(分母・分子)から除外する。
        for col_name, feat_name in [("jockey_id", "jockey_top3_rate"), ("trainer_id", "trainer_top3_rate")]:
            all_keys = df[[col_name, "date"]].drop_duplicates()
            evidence = df.loc[~df["is_unresolved"], [col_name, "date", "is_top3"]]
            daily = evidence.groupby([col_name, "date"])["is_top3"].agg(["sum", "count"]).reset_index()

            df_grp = all_keys.merge(daily, on=[col_name, "date"], how="left")
            df_grp["sum"] = df_grp["sum"].fillna(0)
            df_grp["count"] = df_grp["count"].fillna(0)
            df_grp = df_grp.sort_values([col_name, "date"]).set_index("date")

            grouped = df_grp.groupby(col_name)
            # closed="left": 集計対象日(当日)自身を含まない直近365日ウィンドウ(リーク防止)
            df_grp["window_sum"] = grouped["sum"].rolling("365D", closed="left").sum().reset_index(level=0, drop=True)
            df_grp["window_count"] = grouped["count"].rolling("365D", closed="left").sum().reset_index(level=0, drop=True)
            df_grp = df_grp.reset_index()

            df_grp[feat_name] = df_grp["window_sum"] / df_grp["window_count"]
            df_grp.loc[df_grp["window_count"].fillna(0) < 5, feat_name] = 0.3  # 出走数が少ない場合はデフォルト値
            df_grp[feat_name] = df_grp[feat_name].fillna(0.3)

            # マージ (all_keysはdf自身から作っているため、必ず自己結合が成立する)
            df = df.merge(df_grp[[col_name, "date", feat_name]], on=[col_name, "date"], how="left", suffixes=("", "_new"))
            if f"{feat_name}_new" in df.columns:
                df[feat_name] = df[f"{feat_name}_new"]
                df.drop(columns=[f"{feat_name}_new"], inplace=True)

        # --- 馬の過去成績集計 (リーク防止) ---
        # horse_results から集計する。
        # horse_results には過去走の情報が格納されている。
        # 各馬について、レース日時点での過去3走、5走、全走の集計値を計算する。
        if race_ids is not None:
            df = df[df["race_id"].isin(race_ids)].reset_index(drop=True)
            df_horse_results = df_horse_results[df_horse_results["horse_id"].isin(df["horse_id"])]

        df_hr = df_horse_results.copy()
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

        df_hr = df_hr.sort_values(["horse_id", "date"])

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
            "prize_sum_3": ("prize", 3, "sum"),
            "prize_sum_5": ("prize", 5, "sum"),
            "prize_sum_all": ("prize", None, "sum"),
            "margin_mean_3": ("margin", 3, "mean"),
            "margin_mean_5": ("margin", 5, "mean"),
            "last_3f_mean_3": ("last_3f", 3, "mean"),
            "last_3f_mean_5": ("last_3f", 5, "mean"),
            "time_diff_mean_3": ("time_diff", 3, "mean"),
            "time_diff_mean_5": ("time_diff", 5, "mean"),
            # 脚質分類用の1角・4角平均位置
            "first_corner_mean_5": ("first_corner", 5, "mean"),
            "last_corner_mean_5": ("last_corner", 5, "mean"),
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

        # 直近の過去走自身の日付を保持しておき、interval_daysの計算に用いる
        df_hr_feats["hr_date"] = df_hr_feats["date"]

        # df とマージ (merge_asof を使用)
        # 注意: merge_asofはby=指定時でも「onキー(date)がフレーム全体でソート済み」であることを
        # 要求するため、必ずdate単独でグローバルにソートする(horse_id優先ソートでは複数開催日で失敗する)
        df["horse_id"] = df["horse_id"].astype(str)
        df_hr_feats["horse_id"] = df_hr_feats["horse_id"].astype(str)
        df["date"] = pd.to_datetime(df["date"]).astype("datetime64[ns]")
        df_hr_feats["date"] = pd.to_datetime(df_hr_feats["date"]).astype("datetime64[ns]")
        df = df.sort_values("date", kind="stable")
        df_hr_feats = df_hr_feats.sort_values("date", kind="stable")

        # merge_asof で「dateより厳密に前(allow_exact_matches=False)」の直近の過去走集計を結合する。
        df = pd.merge_asof(
            df,
            df_hr_feats,
            on="date",
            by="horse_id",
            direction="backward",
            allow_exact_matches=False
        )

        # 前走からの間隔（日数）: 結合された直近過去走の日付との差分から計算する
        df["interval_days"] = (df["date"] - df["hr_date"]).dt.days
        df = df.drop(columns=["hr_date"])

        # --- 条件別集計 (同距離帯・同コース種別・同開催場での過去平均着順) ---
        df["distance_band"] = self._distance_band(df["distance"])
        df["rank_mean_by_distance_band"] = self._asof_condition_feature(
            df, df_hr, ["distance_band"], "rank_num", "rank_mean_by_distance_band"
        )
        df["rank_mean_by_course_type"] = self._asof_condition_feature(
            df, df_hr, ["course_type"], "rank_num", "rank_mean_by_course_type"
        )
        df["rank_mean_by_place"] = self._asof_condition_feature(
            df, df_hr, ["place"], "rank_num", "rank_mean_by_place"
        )

        # 脚質(run_style)の決定
        # 過去5走の1角位置が前方にいれば逃げ・先行
        # first_corner_mean_5 から分類
        # 1番手付近: 逃げ, 2〜4番手: 先行, 5〜8番手: 差し, それ以降: 追込
        # 頭数比で考えるのが理想だが、簡易的に順位で判定
        df["run_style"] = "差し"
        df.loc[df["first_corner_mean_5"] <= 1.5, "run_style"] = "逃げ"
        df.loc[(df["first_corner_mean_5"] > 1.5) & (df["first_corner_mean_5"] <= 4.0), "run_style"] = "先行"
        df.loc[(df["first_corner_mean_5"] > 4.0) & (df["first_corner_mean_5"] <= 8.0), "run_style"] = "差し"
        df.loc[df["first_corner_mean_5"] > 8.0, "run_style"] = "追込"
        df["run_style"] = df["run_style"].astype("category")

        # 脚質×コース属性(開催場)の交互作用特徴量
        df["run_style_place"] = (df["run_style"].astype(str) + "_" + df["place"].astype(str)).astype("category")

        # --- 血統マージ ---
        df = df.merge(df_peds, on="horse_id", how="left")

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
