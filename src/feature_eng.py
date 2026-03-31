import pandas as pd
import os

def feature_engineering(input_path, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df = pd.read_csv(input_path)

    df["Age_BMI"] = df["Age"] * df["BMI"]
    df["BP_HeartRate"] = df["BloodPressure"] * df["HeartRate"]
    df["Cholesterol_BMI"] = df["Cholesterol"] * df["BMI"]

    df.to_csv(output_path, index=False)
    print("✅ Feature engineering completed")

    return output_path
