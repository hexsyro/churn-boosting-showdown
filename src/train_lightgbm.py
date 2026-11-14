
import time
import joblib
import pandas as pd
import lightgbm as lgb

def train(x_train, y_train,x_val, y_val):
    start = time.time()

    lgb_model = lgb.LGBMClassifier( learning_rate=0.035, max_depth=-1,min_data_in_leaf=30, 
                               num_leaves=6, n_estimators=2500, reg_alpha=0.1, reg_lambda=5)
    lgb_model.fit(x_train, y_train, eval_set=[(x_train, y_train), (x_val, y_val)])    
    lgb_time = time.time() - start

    return lgb_model, lgb_time


def main():
    print()
    print()
    y_train=pd.read_csv('data/processed/y_train.csv')
    x_train=pd.read_csv('data/processed/train_enc.csv')
    x_val=pd.read_csv('data/processed/validate_enc.csv')
    y_val=pd.read_csv('data/processed/y_val.csv')
    print('loaded the training data for LightGBM model.')
    print()
    print('Training LightGBM model...')
    lgb_modal, lgb_time = train(x_train, y_train, x_val, y_val)
    print('LightGBM model trained successfully.')
    print(f"LightGBM training time: {lgb_time:.2f} seconds")
    print()
    joblib.dump(lgb_modal, 'models/lgb_model.joblib')
    print('Saved the LightGBM model to models/lgb_model.joblib')
    print('done!')

if __name__ == "__main__":
    main()