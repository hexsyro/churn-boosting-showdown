import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

def cnv_tot_charge(df):
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)
    return df

def split(df):
    x=df.drop(columns=['Churn'])
    y=df.Churn

    x_temp,x_test,y_temp,y_test=train_test_split(x, y, stratify=y, test_size=0.2, random_state=42)
    x_train,x_val,y_train,y_val=train_test_split(x_temp, y_temp, stratify=y_temp, test_size=0.15, random_state=42)

    return x_train, x_test, x_val, y_train, y_test, y_val

def encode(x_train, x_test, x_val):
    results = {}
    cats=['gender','Partner','Dependents','PhoneService','MultipleLines','InternetService','OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies','Contract','PaperlessBilling','PaymentMethod']

    ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore", drop=None)
    ohe.fit(x_train[cats])

    for name, x in {'train_enc': x_train, 'test_enc': x_test, 'validate_enc': x_val}.items():    
        x_enc=pd.DataFrame(ohe.transform(x[cats]), columns=ohe.get_feature_names_out(), index=x.index)
        results[name] = x.join(x_enc).drop(columns=cats)

    return results

def export(results, x_train, x_test, x_val, y_train, y_test, y_val):
    x_train.to_csv('data/processed/x_train.csv',index=False)
    x_test.to_csv('data/processed/x_test.csv',index=False)
    x_val.to_csv('data/processed/x_val.csv',index=False)
    y_train.to_csv('data/processed/y_train.csv',index=False)
    y_test.to_csv('data/processed/y_test.csv',index=False)
    y_val.to_csv('data/processed/y_val.csv',index=False)
    results['train_enc'].to_csv('data/processed/train_enc.csv',index=False)
    results['test_enc'].to_csv('data/processed/test_enc.csv',index=False)
    results['validate_enc'].to_csv('data/processed/validate_enc.csv',index=False)


def main():
    print()
    print()
    print('Preprocessing the data...')
    print()
    df=pd.read_csv('data/processed/Telco_customer_churn.csv')
    df=cnv_tot_charge(df)
    print('  Data loaded and TotalCharges converted to numeric.')
    print()

    x_train, x_test, x_val, y_train, y_test, y_val = split(df)
    print('  Data split into train, test, and validation sets.')
    print()

    results = encode(x_train, x_test, x_val)
    print('  Data encoded using OneHotEncoder.')
    print()

    export(results, x_train, x_test, x_val, y_train, y_test, y_val)
    print('  Saved the preprocessed files to data/processed/')

    print('Done!')
    

if __name__ == '__main__':
   main()