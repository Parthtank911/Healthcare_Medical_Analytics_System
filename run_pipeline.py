from src.data_cleaning import clean_data
from src.eda import run_eda
from src.feature_eng import feature_engineering
from src.train_ml_model import train_ml_model
from src.predict_ml import predict_ml

from deep_learning.generate_ecg import generate_ecg
from deep_learning.preprocess_ecg import preprocess_ecg
from deep_learning.train_ecg_cnn import train_ecg_cnn
from deep_learning.train_ecg_lstm import train_ecg_lstm
from deep_learning.predict_ecg import predict_ecg

print("🚀 STARTING HEALTHCARE ML + DL PIPELINE")

# TABULAR ML
cleaned = clean_data(
    "data/raw/patient_data_10000.csv",
    "data/processed/patient_cleaned.csv"
)

run_eda(cleaned)

features = feature_engineering(
    cleaned,
    "data/processed/patient_features.csv"
)

train_ml_model(features)
predict_ml(features)

# DEEP LEARNING ECG
generate_ecg()
preprocess_ecg()
train_ecg_cnn()
train_ecg_lstm()
predict_ecg()

print("🎉 PIPELINE COMPLETED SUCCESSFULLY")
