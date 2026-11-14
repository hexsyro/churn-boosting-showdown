
import time
import joblib
import pandas as pd
import xgboost as xgb

def train(x_train, y_train, x_val, y_val):
    start = time.time()
    xgb_modal = xgb.XGBClassifier(max_depth=3, learning_rate=0.0075, n_estimators=1000,
                          subsample=0.8, colsample_bytree=0.8, random_state=42)
    xgb_modal.fit(x_train, y_train,eval_set=[(x_train, y_train),(x_val, y_val)], verbose=False)
    xgb_time = time.time() - start
    return xgb_modal, xgb_time


def main():
    print()
    print()
    y_train=pd.read_csv('data/processed/y_train.csv')
    x_train=pd.read_csv('data/processed/train_enc.csv')
    x_val=pd.read_csv('data/processed/validate_enc.csv')
    y_val=pd.read_csv('data/processed/y_val.csv')
    print('loaded the training data for XGBoost model.')
    print()
    print('Training XGBoost model...')
    xgb_modal, xgb_time = train(x_train, y_train, x_val, y_val)
    print('XGBoost model trained successfully.')
    print(f"XGBoost training time: {xgb_time:.2f} seconds")
    print()
    joblib.dump(xgb_modal, 'models/xgb_model.joblib')
    print('Saved the XGBoost model to models/xgb_model.joblib')
    print('done!')

if __name__ == "__main__":
    main()