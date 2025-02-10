import pandas as pd
def handling_missing_values(df):
    """Drop rows with missing values."""
    df.dropna(inplace=True)
    return df

def remove_outliers(df, column):
    """Remove outliers in the specified column using the 5th and 95th quantiles."""
    lower_bound = df[column].quantile(0.05)
    upper_bound = df[column].quantile(0.95)
    # Use .loc to ensure we are working on the DataFrame copy properly
    return df.loc[(df[column] > lower_bound) & (df[column] < upper_bound)]

def remove_duplicates(df):
    """Remove duplicate rows."""
    df.drop_duplicates(inplace=True)
    return df

def correct_data_types(df):
    """Convert columns to their proper data types."""
    if 'transaction_date' in df.columns:
        df.loc[:, 'transaction_date'] = pd.to_datetime(df['transaction_date'], errors='coerce')
    if 'transaction_amount' in df.columns:
        df.loc[:, 'transaction_amount'] = pd.to_numeric(df['transaction_amount'], errors='coerce')
    if 'transaction_id' in df.columns:
        df.loc[:, 'transaction_id'] = df['transaction_id'].astype('str')
    if 'customer_id' in df.columns:
        df.loc[:, 'customer_id'] = df['customer_id'].astype('str')
    if 'product_id' in df.columns:
        df.loc[:, 'product_id'] = df['product_id'].astype('str')
    if 'ip_address' in df.columns:
        df.loc[:, 'ip_address'] = df['ip_address'].astype('str')
    if 'signup_date' in df.columns:
        df.loc[:, 'signup_date'] = pd.to_datetime(df['signup_date'], errors='coerce')
    if 'purchase_date' in df.columns:
        df.loc[:, 'purchase_date'] = pd.to_datetime(df['purchase_date'], errors='coerce')
    return df

def data_cleaning_pipeline(df):
    """Apply all cleaning steps to the dataframe."""
    df = handling_missing_values(df)
    df = correct_data_types(df)
    # Example: remove outliers for the 'transaction_amount' column if it exists
    if 'transaction_amount' in df.columns:
        df = remove_outliers(df, 'transaction_amount')
    df = remove_duplicates(df)
    return df