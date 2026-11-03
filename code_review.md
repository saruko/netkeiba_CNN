# コードレビュー: 対応状況

初版レビュー(仕様書 `keiba_ai_spec.md` と実装の照合)で挙げた指摘の対応状況と、現時点で残っている問題をまとめる。
2026-11 改定。

---

## 1. 初版レビュー指摘の対応状況

| # | 区分 | 内容 | 状況 | 対応箇所 |
|---|------|------|------|----------|
| 1 | バグ | `main.py` の `parse_main` 不要代入 | 修正済み | `main.py` `args = parser.parse_args()` |
| 2 | リーク | 同日複数騎乗時の騎手・調教師複勝率リーク | 修正済み | `features.py` 日単位集計 + `rolling("365D", closed="left")` で当日を除外 |
| 3 | リーク | 同日複数出走馬の `merge_asof` リーク | 修正済み | `features.py` `allow_exact_matches=False` |
| 4 | リーク | early stopping とキャリブレーションが同一データ | 修正済み | `train.py` `split_validation` で検証データを日付で前半/後半に分割 |
| 5 | 仕様不適合 | 条件別集計(距離帯・コース種別・開催場)未実装 | 修正済み | `features.py` `_asof_condition_feature` |
| 6 | 仕様不適合 | `distance` が特徴量に無い | 修正済み | `train.py` `num_cols` |
| 7 | 仕様不適合 | 騎手・調教師が全期間累積 | 修正済み | `features.py` 直近365日ローリング |
| 8 | 仕様不適合 | 賞金合計・着差平均の未実装 | 修正済み | `parser.py` に `prize`/`margin` 列、`features.py` で集計 |
| 9 | 仕様不適合 | 複勝の回収率シミュレーション不在 | 修正済み | `simulator.py` `evaluate_bets` |
| 10 | 仕様不適合 | 二値分類ベースラインの比較不在 | 修正済み | `simulator.py` 両モデルを同条件で評価 |
| 11 | 仕様不適合 | 実験ログのJSON保存未実装 | 修正済み | `train.py` `_save_experiment_log` |
| 12 | テスト欠落 | 騎手・調教師リーク検証が空 | 修正済み | `tests/test_leakage.py` |
| 13 | テスト欠落 | 分割検査テスト未実装 | 修正済み | `tests/test_leakage.py` `test_split_boundary` |
| 14 | テスト欠落 | 払戻計算の単体テスト未実装 | 修正済み | `tests/test_simulation.py` |
| 15 | CLI | `YYYY-MM` 形式のサポート不在 | 修正済み | `main.py` `_parse_scrape_date` |
| 16 | 堅牢性 | クラス判定 `"L" in race_name` の誤マッチ | 一部対応 | `"(L)"` に限定済み。`"OP" in race_name` は残っており、レース名に "OP" を含む非オープン戦を誤判定しうる |
| 17 | 堅牢性 | `__init__.py` の欠落 | 修正済み | `src/*/__init__.py` |
| 18 | 効率 | 開催のない日も毎日HTTPリクエスト | **未対応** | `main.py` `handle_scrape`。開催カレンダーのHTML構造の確認が必要 |
| 4-1 | 堅牢性 | レース結果パーサーの列インデックスがハードコード | **未対応** | `parser.py` `parse_race` は `cols[0]`〜`cols[20]` を直接参照。馬成績パーサー(`parse_horse`)はヘッダー名ベースに修正済み |
| 4-4 | 堅牢性 | 血統パーサーが rowspan 構造に依存 | **未対応** | `parser.py` `parse_ped` |

---

## 2. 初版以降に見つかり修正した問題

| 内容 | 対応箇所 |
|------|----------|
| 障害レース・JRA以外のレースが学習対象に混入(仕様書§2違反) | `features.py` 場コード01〜10・芝ダのみに限定 |
| 出走取消・競走除外の馬が負例として学習・勝率正規化に混入 | `features.py` |
| 降着・再騎乗(例: `"4(降)"`)の着順が数値化されず着外扱い | `features.py` `_parse_rank` |
| 全レース混合のIsotonic回帰でLambdaRankスコアを校正していた(スコアはレース内相対値) | `calibration.py` `SoftmaxCalibrator` (レース内softmaxの温度推定) |
| 市場(オッズ)との確率予測精度の比較が無く、エッジの有無を判定できない | `simulator.py` レース単位LogLoss/Brierをmarketと比較 |
| 回収率に信頼区間が無い | `simulator.py` レース単位ブートストラップ95%CI |
| 当日予測のたびに全レースの特徴量を再生成・保存していた | `features.py` `build_features(race_ids=...)` |
| ウォークフォワードのテスト開始日がハードコード | `config.yaml` `simulation.test_start` |
| 着順・1角位置を頭数で相対化していない(脚質判定も絶対位置) | `parser.py` `horse_results.n_horses`、`features.py` `rank_ratio_*`・`run_style` |
| 相手関係(クラス)を考慮していない | `parser.py` `horse_results.race_class`、`features.py` `class_level_mean_5`・`class_diff` |
| 地方・海外・障害の過去走が成績集計に混入 | `features.py` 成績集計をJRA平地に限定(前走間隔は全出走) |
| 母・祖母などのIDを高カーディナリティのカテゴリ変数として学習(過学習) | `train.py` 父・母父のみに限定、`features.py` 種牡馬の産駒複勝率 |
| 騎手・調教師の出走数5未満で固定値0.3 | `features.py` `_entity_top3_rate` 事前分布による縮小推定 |
| モデル単独で市場と比較しており、市場情報との結合が無い | `calibration.py` `MarketCombiner`、`simulator.py` で市場のみ再校正と比較 |
| タイム系特徴量がコース・馬場の違いを補正していない | `features.py` `_speed_figures` |

---

## 3. 残っている問題

| 優先度 | 内容 | 箇所 |
|--------|------|------|
| 中 | バックテストは確定オッズで評価しており、購入時点オッズとの乖離で回収率が楽観的に出る(仕様書§13の既知リスク。`odds_margin` で一律補正のみ) | `simulator.py` |
| 中 | スピード指数のクラス補正・馬場差は勝ち馬のみから推定しており、少頭数・少レースの開催では誤差が大きい。上がり3Fは未補正 | `features.py` `_speed_figures` |
| 中 | `horse_results.n_horses`・`race_class` は `reparse` で埋まる。クラス判定はレース名の文字列に依存し、馬成績ページのレース名表記(G1等)は未検証 | `parser.py` |
| 低 | 二値分類モデルのパラメータが `train.py` と `simulator.py` に重複定義 | `train.py`, `simulator.py` |
| 低 | `features.py` `_determine_run_style` は未使用のコード | `features.py` |
| 低 | 上記 #16, #18, 4-1, 4-4 | — |
