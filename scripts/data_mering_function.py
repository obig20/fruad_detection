import pandas as pd
def merge_geolocation_data(df_fraud, df_ip):
    """
    Merge the fraud data with the IP mapping data on the 'ip_int' column.
    """
    df_merged = pd.merge(df_fraud, df_ip, on='ip_int', how='left')
    return df_merged

def merge_creditcard_fraud_data(df_fraud, df_credit, merge_key='credit_card_id'):
    """
    Merge the fraud data with the credit card data on the specified merge key.
    """
    if merge_key in df_fraud.columns and merge_key in df_credit.columns:
        df_merged = pd.merge(df_fraud, df_credit, on=merge_key, how='left')
        return df_merged
    else:
        print(f"Merge key '{merge_key}' not found in both dataframes. Skipping merge with credit card data.")
        return df_fraud