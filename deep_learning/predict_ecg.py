import numpy as np
from tensorflow.keras.models import load_model
import pandas as pd
import os

def predict_ecg():
    os.makedirs("data/predictions", exist_ok=True)

    X = np.load("data/processed/ecg_X.npy")

    cnn = load_model("models/ecg_cnn.keras")
    lstm = load_model("models/ecg_lstm.keras")

    cnn_preds = cnn.predict(X).flatten()
    lstm_preds = lstm.predict(X).flatten()

    df = pd.DataFrame({
        "CNN_Prediction": cnn_preds,
        "LSTM_Prediction": lstm_preds
    })

    df.to_csv("data/predictions/dl_ecg_predictions.csv", index=False)
    print("✅ ECG predictions saved")
