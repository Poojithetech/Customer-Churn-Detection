import pandas as pd
from data_loading import load_data
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Convert 'TotalCharges' to numeric if it exists
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

    # Encode binary categorical variables (e.g., Yes/No -> 1/0, Female/Male)
    binary_cols = [col for col in df.columns if df[col].nunique() == 2 and df[col].dtype == 'object']
    for col in binary_cols:
        df[col] = df[col].map({'Yes': 1, 'No': 0, 'Female': 1, 'Male': 0})

    # One-hot encode remaining categorical variables
    df = pd.get_dummies(df, drop_first=True)

    # 🔴 Drop rows with missing values after all conversions
    before = df.shape[0]
    df.dropna(inplace=True)
    after = df.shape[0]
    print(f"Dropped {before - after} rows due to NaNs.")

    return df
