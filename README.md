# 競馬AI予測システム

netkeibaからデータを自動収集し、機械学習で各馬の勝率を推定して、期待値の高い馬券を提案するシステムです。

---

## できること

| 機能 | 説明 |
|------|------|
| **データ収集** | netkeiba.comからレース結果・馬の過去成績・血統・オッズを自動取得 |
| **学習** | LightGBM LambdaRankで「レース内の着順」を予測するモデルを訓練 |
| **予測** | 未来のレース(出馬表が公開済み)に対して各馬の勝率と期待値を算出 |
| **シミュレーション** | 過去データで「このモデルで賭けていたら回収率は何%だったか」を検証 |

---

## セットアップ (初回のみ)

### 1. Python環境の準備

Python 3.10以上が必要です。

```
cd netkeiba_CNN
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. 動作確認

```
python -m pytest tests/
```

全テストがパスすればOKです。

---

## 使い方: ステップバイステップ

**全体の流れ:**

```
スクレイピング → 特徴量生成 → 学習 → 予測
    (1)            (2)        (3)    (4)
```

各ステップは必ずこの順番で実行してください。

---

### ステップ1: データ収集 (スクレイピング)

netkeibaからレース結果・馬の過去成績・血統データを取得します。

#### 基本コマンド

```
python main.py scrape --start 開始日 --to 終了日
```

日付は `YYYY-MM-DD`(日指定) または `YYYY-MM`(月指定) のどちらでもOKです。

#### 実行例

```bash
# 2024年1月だけ取得
python main.py scrape --start 2024-01 --to 2024-01

# 2024年の1年分を取得
python main.py scrape --start 2024-01 --to 2024-12

# 2020年から2025年まで5年分を一括取得
python main.py scrape --start 2020-01 --to 2025-12
```

#### 全データを取得して本格的に学習したい場合

netkeibaに過去データがある範囲(おおむね2008年以降)を取得できます。
推奨は **5年分以上** です。多いほどモデルの精度が上がります。

```bash
# 推奨: 2018年〜現在(約7年分)
python main.py scrape --start 2018-01 --to 2025-12
```

**所要時間の目安:**
- 1年分 ≒ 約3,500レース × 各レース平均15頭 = 約52,500ページ
- netkeibaへの負荷を避けるため1秒間隔でアクセスするので、1年分 ≒ **約15時間**
- 5年分 ≒ 約3〜4日 (PCをつけっぱなしにしておく必要があります)
- 一度取得したページはローカルに保存されるので、**2回目以降は未取得分だけ**が追加取得されます

**途中で止まっても大丈夫:**
- 取得済みデータはすべて `data/html/` にgzip保存されており、再実行時にスキップされます
- ネットワークエラーで個別レースの取得に失敗しても、バッチ全体は止まりません
- 中断後に同じコマンドを再実行すれば、続きから取得します

#### テスト用 (少量で動作確認)

```bash
# 最初の10レースだけ取得して動作確認
python main.py scrape --start 2024-01 --to 2024-01 --limit-races 10
```

---

### ステップ2: 特徴量生成

収集したデータから、モデルが学習に使う数値データ(特徴量)を計算します。

```
python main.py build-features
```

SQLiteのデータを読み込み、以下のような特徴量をParquetファイルとして出力します:
- 各馬の直近3走/5走/全走の平均着順・賞金・着差・上がり3F
- 距離適性・コース適性・競馬場適性
- 騎手・調教師の複勝率(直近1年ローリング)
- 血統情報(父・母・父父・母父など6系統)
- 馬体重・斤量・枠番・出走頭数・レース間隔 など

---

### ステップ3: 学習

#### configの設定 (学習前に確認)

[config/config.yaml](config/config.yaml) を開いて、スクレイピングした期間に合わせて設定を変更してください:

```yaml
model:
  # 学習に使う年(リスト)
  target_years_train: [2018, 2019, 2020, 2021, 2022, 2023]
  # 検証用の年(early stoppingとキャリブレーションに使用)
  target_year_val: 2024
  # テスト用の年(最終評価用)
  target_year_test: 2025
```

**ルール:** train < val < test の時系列順にしてください。
未来のデータで学習してしまうと、見かけ上の精度は高くなりますが実際には使い物になりません。

#### 学習の実行

```bash
# 通常の学習
python main.py train

# ハイパーパラメータの自動探索付き (時間がかかるが精度が上がる可能性あり)
python main.py train --tune
```

学習が完了すると `models/` フォルダに以下が保存されます:
- `lgb_lambdarank.model` — メインの予測モデル
- `lgb_binary.model` — 比較用の二値分類モデル
- `softmax_calibration.model` — 勝率の校正モデル
- `categories.json` — カテゴリ変数の水準(予測時の整合性に必須)
- `experiment_*.json` — 実験ログ(再現用)

---

### ステップ4: 予測

**レースIDが必要です。** レースIDはnetkeibaのURLに含まれる12桁の数字です。

例: `https://race.netkeiba.com/race/shutuba.html?race_id=202506050811`
→ レースID = `202506050811`

```bash
python main.py predict --race-id 202506050811
```

出力される情報:
- 各馬の **勝率** (校正済み、レース内合計1.0)
- 各馬の **期待値** (勝率 × オッズ × 安全マージン)
- 期待値が1.1を超える馬が **購入候補** として表示されます

結果は `data/prediction_レースID.csv` に保存されます。

---

### (任意) シミュレーション

学習済みモデルの性能を、過去データで検証します。
半年ごとにテスト期間をスライドさせる「ウォークフォワード検証」を行います。
`build-features` で生成済みの特徴量を読み込むため、先に `build-features` を実行してください。
最初のテスト期間の開始日は `config.yaml` の `simulation.test_start` で設定します。

```
python main.py simulate
```

各期間ごとに以下が表示されます:
- 購入件数 / 的中率 / 回収率 (単勝・複勝それぞれ)
- LambdaRankモデル vs 二値分類モデル vs 市場(単勝オッズの暗黙勝率) の勝率予測精度 (レース単位LogLoss・Brier)

最後に全期間通算の結果として、上記の精度と、回収率の95%信頼区間(レース単位ブートストラップ)が表示されます。

**見方:**
- モデルの LogLoss が market より小さくない限り、「市場より正確な確率推定」という根拠はありません
- 回収率の信頼区間の下限が100%を超えていない限り、利益が出るとは判断できません(1期間の回収率100%超は偶然でも頻繁に起こります)

---

## よくある質問

### Q. スクレイピングが途中で止まったら？

そのまま同じコマンドを再実行してください。取得済みのページは自動でスキップされます。

### Q. データを追加したい

新しい期間を `scrape` して、`build-features` → `train` を再実行するだけです。
古いデータを消す必要はありません。

### Q. 予測の精度を上げるには？

1. **データ量を増やす** — 最も効果が大きい。5年分以上を推奨
2. `python main.py train --tune` でハイパーパラメータ探索を行う
3. `config.yaml` の `betting.min_expected_value` を上げると購入基準が厳しくなる(的中率は下がるが回収率は上がりやすい)

### Q. 地方競馬にも使える？

現在はJRA(中央競馬)のみ対応です。`get_race_ids_by_date` はnetkeibaの全レースを返すため、地方競馬のIDも混入する可能性があります。JRAのレースIDは先頭4桁の場コードが01〜10です。

### Q. 本当に儲かる？

**保証はできません。** 競馬の控除率は約25%(単勝)であり、ランダムに購入した場合の期待回収率は約75%です。このシステムは期待値ベースの判断により回収率の向上を目指しますが、利益を保証するものではありません。投資は自己責任でお願いします。

---

## フォルダ構成

```
競馬/
├── main.py                  ← メインのCLI (ここからすべて実行)
├── config/
│   └── config.yaml          ← 設定ファイル (学習期間・パラメータ)
├── src/
│   ├── scraping/            ← データ取得 (netkeiba)
│   ├── preprocessing/       ← データ整形・特徴量生成
│   ├── training/            ← モデル学習・キャリブレーション
│   ├── predict/             ← 未来レース予測
│   ├── simulation/          ← 過去データでの検証
│   └── policies/            ← 購入判定ロジック
├── tests/                   ← テストコード
├── data/                    ← 取得データ・特徴量 (自動生成)
│   ├── html/                ← スクレイピングしたHTML (gzip)
│   ├── db.sqlite            ← パース済みデータ
│   └── features/            ← 特徴量Parquet
└── models/                  ← 学習済みモデル (自動生成)
```

---

## コマンド一覧

| コマンド | 説明 | 所要時間目安 |
|---------|------|-------------|
| `python main.py scrape --start YYYY-MM --to YYYY-MM` | データ収集 | 1年分≒15時間 |
| `python main.py build-features` | 特徴量生成 | 数分 |
| `python main.py train` | モデル学習 | 数分〜10分 |
| `python main.py train --tune` | 学習+ハイパラ探索 | 30分〜1時間 |
| `python main.py simulate` | 過去データ検証 | データ量による |
| `python main.py predict --race-id XXXXXXXXXXXX` | レース予測 | 数分 |
| `python -m pytest tests/` | テスト実行 | 数秒 |

---

## 典型的な運用フロー

### 初回セットアップ (一度だけ)

```bash
# 1. 環境構築
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# 2. 過去データの大量取得 (数日かかる。途中で止めても再開可能)
python main.py scrape --start 2018-01 --to 2025-12

# 3. config.yaml の学習年を設定
#    target_years_train: [2018, 2019, 2020, 2021, 2022, 2023]
#    target_year_val: 2024
#    target_year_test: 2025

# 4. 特徴量生成 → 学習
python main.py build-features
python main.py train

# 5. 性能確認
python main.py simulate
```

### 毎週のレース予測

```bash
# 仮想環境を有効化
.venv\Scripts\activate

# 当週のレースデータを更新 (既存データはスキップされる)
python main.py scrape --start 2025-07 --to 2025-07

# 予測したいレースのIDをnetkeibaのURLから確認して実行
python main.py predict --race-id 202506050811
```

### 月1回のモデル更新 (推奨)

```bash
# 最新月までデータを追加取得
python main.py scrape --start 2025-07 --to 2025-07

# 特徴量再生成 → 再学習
python main.py build-features
python main.py train
```
