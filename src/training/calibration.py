import os
import joblib
import pandas as pd
from sklearn.isotonic import IsotonicRegression
import yaml
from src.training.train import Trainer

class Calibrator:
    def __init__(self, config_path="config/config.yaml"):
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)
            
        base_dir = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        self.models_dir = os.path.join(base_dir, self.config["paths"]["models_dir"])
        self.trainer = Trainer(config_path)

    def run(self):
        # 検証データのロード
        # early stopping に使用したデータと同一データでキャリブレーションを学習すると
        # 校正が楽観的になるため、検証データの後半(early stoppingに使われていない部分)のみを使う。
        print("Loading validation dataset for calibration...")
        df_val_all = self.trainer.load_dataset([self.trainer.val_year])
        _, df_val = self.trainer.split_validation(df_val_all)

        # 学習時と同一のカテゴリ水準を適用する(水準がズレるとスコアが黙って狂う)
        df_val = self.trainer.apply_categories(df_val)

        # モデルのロード
        model_path = os.path.join(self.models_dir, "lgb_lambdarank.model")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}. Train model first.")
        model = joblib.load(model_path)
        
        # 特徴量
        X_val = df_val[self.trainer.features]
        # ターゲット (1着なら1, その他は0)
        df_val["is_winner"] = (df_val["rank_num"] == 1).astype(int)
        y_val = df_val["is_winner"]
        
        # 予測スコア (Raw Score)
        raw_scores = model.predict(X_val)
        
        # キャリブレーションの学習
        print("Fitting Isotonic Regression for calibration...")
        ir = IsotonicRegression(out_of_bounds="clip")
        ir.fit(raw_scores, y_val)
        
        # 保存
        calib_path = os.path.join(self.models_dir, "isotonic_calibration.model")
        joblib.dump(ir, calib_path)
        print(f"Calibration model saved to {calib_path}")
        
if __name__ == "__main__":
    calibrator = Calibrator()
    calibrator.run()
