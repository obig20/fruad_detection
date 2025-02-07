import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder,LabelEncoder
def normalize_and_scale(df, features):
    """
    Scale specified numeric features using StandardScaler.
    """
    if features:
        scaler = StandardScaler()
        df[features] = scaler.fit_transform(df[features])
    return df
