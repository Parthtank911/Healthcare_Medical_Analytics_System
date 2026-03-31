import os
import joblib

def create_dir(path):
    os.makedirs(path, exist_ok=True)

def save_model(model, path):
    create_dir(os.path.dirname(path))
    joblib.dump(model, path)

def load_model(path):
    return joblib.load(path)
