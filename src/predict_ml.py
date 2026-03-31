import pandas as pd
from src.utils import load_model
import os

def predict_ml(input_path):
    os.makedirs("data/predictions", exist_ok=True)

    df = pd.read_csv(input_path)
    X = df.drop("HeartDiseaseRisk", axis=1)

    model = load_model("models/ml_model.pkl")
    df["Prediction"] = model.predict(X)

    df.to_csv("data/predictions/ml_predictions.csv", index=False)
    print("✅ ML predictions saved")
