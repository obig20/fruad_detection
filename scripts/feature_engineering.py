import pandas as pd
def feature_engineering(df):
     #Engineer additional features:
     # - Daily and hourly transaction counts per user.
      #- Extract time-based features such as hour of day and day of week.
    
    if 'user_id' in df.columns and 'purchase_time' in df.columns:
        # Daily transaction count
        df['transaction_day'] = df['purchase_time'].dt.date
        daily_counts = (
            df.groupby(['user_id', 'transaction_day'])
            .size()
            .reset_index(name='daily_transaction_count')
        )
        df = pd.merge(df, daily_counts, on=['user_id', 'transaction_day'], how='left')

        # Hourly transaction count
        df['transaction_hour'] = df['purchase_time'].dt.floor('H')
        hourly_counts = (
            df.groupby(['user_id', 'transaction_hour'])
            .size()
            .reset_index(name='hourly_transaction_count')
        )
        df = pd.merge(df, hourly_counts, on=['user_id', 'transaction_hour'], how='left')

    if 'purchase_time' in df.columns:
        df['hour_of_day'] = df['purchase_time'].dt.hour
        df['day_of_week'] = df['purchase_time'].dt.dayofweek  # Monday=0, Sunday=6

    return df