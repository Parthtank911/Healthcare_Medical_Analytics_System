import numpy as np
import os

def preprocess_ecg():
    os.makedirs("data/processed", exist_ok=True)

    X = np.load("data/raw/ecg_signals.npy")
    y = np.load("data/raw/ecg_labels.npy")

    X = X.reshape(X.shape[0], X.shape[1], 1)

    np.save("data/processed/ecg_X.npy", X)
    np.save("data/processed/ecg_y.npy", y)

    print("✅ ECG preprocessing completed")
