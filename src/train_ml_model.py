import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from src.utils import save_model

def train_ml_model(input_path):
    df = pd.read_csv(input_path)

    X = df.drop("HeartDiseaseRisk", axis=1)
    y = df["HeartDiseaseRisk"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    acc = accuracy_score(y_test, model.predict(X_test))

    save_model(model, "models/ml_model.pkl")

    print(f"✅ ML model trained with accuracy: {acc:.2%}")

    return model
