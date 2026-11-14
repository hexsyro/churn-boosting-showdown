import pandas as pd

def rmv_nan_churn(df):
    df=df.dropna(subset=['Churn'])
    return df

def main():
    df=pd.read_csv('data/raw/Telco_customer_churn.csv').drop(columns=['customerID'])
    df=rmv_nan_churn(df)
    print('  Data loaded from data/raw/Telco_customer_churn.csv\n plus, dropped the customerID column')

    df.Churn=df.Churn.map({'Yes':1, 'No':0})
    print('  Churn column mapped to binary values.')
    print()
    print('  Value counts (normalized):')
    print(df.Churn.value_counts(normalize=True))
    print()

    df.info()

    df.to_csv('data/processed/Telco_customer_churn.csv', index=False) 
    print('  Saved the file to data/processed/')

    print('Done!')
    

if __name__ == '__main__':
   main()