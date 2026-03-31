import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
import os

def train_ecg_lstm():
    os.makedirs("models", exist_ok=True)

    X = np.load("data/processed/ecg_X.npy")
    y = np.load("data/processed/ecg_y.npy")

    model = Sequential([
        Input(shape=(300, 1)),
        LSTM(64),
        Dense(1, activation="sigmoid")
    ])

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.fit(X, y, epochs=10, batch_size=32, validation_split=0.2)

    model.save("models/ecg_lstm.keras")
    print("✅ ECG LSTM model trained and saved")
