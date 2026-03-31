import numpy as np
import os

def generate_ecg():
    os.makedirs("data/raw", exist_ok=True)

    X = np.random.randn(1000, 300)
    y = np.random.randint(0, 2, 1000)

    np.save("data/raw/ecg_signals.npy", X)
    np.save("data/raw/ecg_labels.npy", y)

    print("✅ ECG raw data generated")
