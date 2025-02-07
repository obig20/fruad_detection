import pandas as pd
def load_fraud_data(path):
    """Load the fraud data from a CSV file."""
    return pd.read_csv(path)

def load_creditcard_data(path):
    """Load the credit card data from a CSV file."""
    return pd.read_csv(path)

def load_ip_data(path):
    """Load the IP mapping data from a CSV file."""
    return pd.read_csv(path)

def load_all_data(fraud_path, creditcard_path, ip_path):
    """
    Load all three datasets and return them as a tuple.
    """
    df_fraud = load_fraud_data(fraud_path)
    df_credit = load_creditcard_data(creditcard_path)
    df_ip = load_ip_data(ip_path)
    return df_fraud, df_credit, df_ip