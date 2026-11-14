
import time
import joblib
import pandas as pd
from catboost import CatBoostClassifier

def train(x_train_unenc, y_train, x_val, y_val, cats):
    start = time.time()
    cb_model = CatBoostClassifier(cat_features=cats, verbose=0, random_state=42, 
                              learning_rate=0.03, iterations=500, l2_leaf_reg=15, depth=4)
    cb_model.fit(x_train_unenc, y_train,eval_set=[(x_train_unenc, y_train), (x_val, y_val)])
    cb_time = time.time() - start

    return cb_model, cb_time


def main():
    cats=['gender','Partner','Dependents','PhoneService','MultipleLines','InternetService','OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies','Contract','PaperlessBilling','PaymentMethod']

    print()
    print()
    y_train=pd.read_csv('data/processed/y_train.csv')
    x_train=pd.read_csv('data/processed/x_train.csv')
    x_val=pd.read_csv('data/processed/x_val.csv')
    y_val=pd.read_csv('data/processed/y_val.csv')
    print('loaded the training data for CatBoost model.')
    print()
    print('Training CatBoost model...')
    cb_modal, cb_time = train(x_train, y_train, x_val, y_val, cats)
    print('CatBoost model trained successfully.')
    print(f"CatBoost training time: {cb_time:.2f} seconds")
    print()
    joblib.dump(cb_modal, 'models/cb_model.joblib')
    print('Saved the CatBoost model to models/cb_model.joblib')
    print('done!')

if __name__ == "__main__":
    main()


