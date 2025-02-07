import pandas as pd
lower_bound = 0.05
upper_bound = 0.95

def handling_missing_values(df):
    # Drop rows with missing values
    df.dropna(inplace=True)
    return df
def remove_outliers(df):
    # Remove outliers
    df = df[(df['column_name'] > lower_bound) & (df['column_name'] < upper_bound)]
    return df
def remove_duplicates(df):
    # Remove duplicates
    df.drop_duplicates(inplace=True)
    return df
def correct_data_types(df):

    if 'transaction_date' in df.columns:
        df['transaction_date'] = pd.to_datetime(df['transaction_date'],errors='coerce')
    if 'transaction_amount' in df.columns:
        df['transaction_amount'] = pd.to_numeric(df['transaction_amount'],errors='coerce')
    if 'transaction_id' in df.columns:
        df['transaction_id'] = df['transaction_id'].astype('str')
    if 'customer_id' in df.columns:
        df['customer_id'] = df['customer_id'].astype('str')
    if 'product_id' in df.columns:
        df['product_id'] = df['product_id'].astype('str')
    if 'ip_address' in df.columns:
        df['ip_address'] = df['ip_address'].astype('str')
    if 'signup_date' in df.columns:
        df['signup_date'] = pd.to_datetime(df['signup_date'],errors='coerce')
    if 'purchase_date' in df.columns:
        df['purchase_date'] = pd.to_datetime(df['purchase_date'],errors='coerce')
    return df
def data_cleaning_pipeline(df):
    df = handling_missing_values(df)
    df = remove_outliers(df)
    df = remove_duplicates(df)
    return df
def main():
    df = pd.read_csv('data.csv')
    df = data_cleaning_pipeline(df)
    df.to_csv('cleaned_data.csv', index=False)
if __name__ == '__main__':
    main() 