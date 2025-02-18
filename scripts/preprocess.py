# Example preprocessing script (preprocess.py)
import pandas as pd

# Load raw data
raw_data = pd.read_csv('data/Fraud_Data.csv')

# Perform preprocessing
processed_data = raw_data  # Add your preprocessing steps here

# Save processed data
processed_data.to_csv('data/processed_fraud_data.csv', index=False)