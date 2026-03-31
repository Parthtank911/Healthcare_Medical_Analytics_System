import pandas as pd
import os

def clean_data(input_path, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df = pd.read_csv(input_path)

    for col in df.columns:
        if df[col].dtype != "object":
            df[col] = df[col].fillna(df[col].median())

    df.to_csv(output_path, index=False)
    print("✅ Data cleaning completed")

    return output_path
