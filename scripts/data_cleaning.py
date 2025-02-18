# Fraud Detection Preprocessing Pipeline
# Imports
import pandas as pd
import numpy as np
import ipaddress
from sklearn.preprocessing import StandardScaler

# Load datasets
def load_data():
    """Load datasets"""
    fraud_df = pd.read_csv('../Data/Fraud_Data.csv')
    ip_df = pd.read_csv('../Data/IpAddress_to_Country.csv')
    return fraud_df, ip_df

# Clean data
def clean_data(fraud_df):
    """Data cleaning steps"""
    # Handle missing values
    fraud_df = fraud_df.dropna()
    
    # Remove duplicates
    fraud_df = fraud_df.drop_duplicates()
    
    # Convert data types
    fraud_df['signup_time'] = pd.to_datetime(fraud_df['signup_time'])
    fraud_df['purchase_time'] = pd.to_datetime(fraud_df['purchase_time'])
    
    return fraud_df

# Add geolocation
def add_geolocation(fraud_df, ip_df):
    """Merge fraud data with IP country data"""
    # Convert floating-point numbers to valid IP addresses
    def int_to_ip(ip_float):
        """Convert a floating-point number to a valid IP address."""
        if pd.isna(ip_float):  # Handle missing values
            return None
        try:
            # Convert the floating-point number to an integer
            ip_int = int(ip_float)
            # Convert the integer to a valid IP address
            return str(ipaddress.IPv4Address(ip_int))
        except (ValueError, TypeError):  # Handle invalid values
            return None
    
    # Apply the function to the IP address column
    fraud_df['ip_address'] = fraud_df['ip_address'].astype(str).apply(int_to_ip)
    
    # Drop rows with invalid IP addresses
    fraud_df = fraud_df.dropna(subset=['ip_address'])
    
    # Convert IP to integer for merging
    def ip_to_int(ip):
        try:
            return int(ipaddress.IPv4Address(ip))
        except (ValueError, TypeError):
            return None
    
    fraud_df['ip_int'] = fraud_df['ip_address'].apply(ip_to_int)
    
    # Sort IP country data for efficient merging
    ip_df = ip_df.sort_values(by='lower_bound_ip_address')
    
    # Merge with IP country data
    merged_df = pd.merge_asof(
        fraud_df.sort_values('ip_int'),
        ip_df,
        left_on='ip_int',
        right_on='lower_bound_ip_address',
        direction='backward'
    )
    
    return merged_df

def transactions_last_hour(group):
    """
    Calculate the cumulative count of transactions made by a user within the last hour.
    """
    # Sort the group by purchase_time (no need for 'by' argument since it's a Series)
    group = group.sort_values()
    
    # Calculate the time difference in hours between consecutive transactions
    time_diff = (group - group.shift()).dt.total_seconds() / 3600
    
    # Cumulative count of transactions within the last hour
    return (time_diff <= 1).cumsum()

def create_features(fraud_df, ip_df):
    """Feature engineering"""
    # Ensure 'signup_time' and 'purchase_time' are in datetime format
    fraud_df['signup_time'] = pd.to_datetime(fraud_df['signup_time'])
    fraud_df['purchase_time'] = pd.to_datetime(fraud_df['purchase_time'])

    # Time-based features
    #fraud_df['purchase_hour_of_day'] = fraud_df['purchase_time'].dt.hour
    #fraud_df['signup_hour_of_day'] = fraud_df['signup_time'].dt.hour
    #fraud_df['purchase_day_of_week'] = fraud_df['purchase_time'].dt.dayofweek
    #fraud_df['signup_day_of_week'] = fraud_df['signup_time'].dt.dayofweek
    
    # Rolling mean of purchase value
    fraud_df['purchase_value_rolling_mean'] = fraud_df.groupby('device_id')['purchase_value'].transform(lambda x: x.rolling(window=7, min_periods=1).mean())
    
    # Mean purchase value per source
    fraud_df['mean_purchase_per_source'] = fraud_df.groupby('source')['purchase_value'].transform('mean')
    
    # Fraud rate by source
    fraud_rates = fraud_df.groupby('source')['class'].mean().rename('source_fraud_rate')
    fraud_df = fraud_df.merge(fraud_rates, left_on='source', right_index=True)
    
    # Sort by user_id and purchase_time before applying transactions_last_hour
    fraud_df = fraud_df.sort_values(by=['user_id', 'purchase_time'])
    
    # Apply the transactions_last_hour function to each group
    fraud_df['transactions_last_hour'] = fraud_df.groupby('user_id')['purchase_time'] \
                                    .transform(lambda x: transactions_last_hour(x))
    
    # Encode categorical features
    fraud_df = pd.get_dummies(fraud_df, columns=['source', 'browser', 'sex'])
    
    return fraud_df

# Full preprocessing pipeline
def preprocess_data():
    """Full preprocessing pipeline"""
    # Load and clean data
    fraud_df, ip_df = load_data()
    cleaned_df = clean_data(fraud_df)
    
    # Merge and feature engineering
    merged_df = add_geolocation(cleaned_df, ip_df)
    final_df = create_features(merged_df, ip_df)
    
    # Normalization
    scaler = StandardScaler()
    numerical_features = ['purchase_value', 'purchase_value_rolling_mean', 'mean_purchase_per_source']
    final_df[numerical_features] = scaler.fit_transform(final_df[numerical_features])
    
    return final_df

if __name__ == "__main__":
    # Run the full pipeline when executed as a script
    processed_data = preprocess_data()
    print("Preprocessing complete!")
    print("Final dataset shape:", processed_data.shape)
    processed_data.to_csv('processed_fraud_data.csv', index=False)