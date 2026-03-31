import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def run_eda(input_path):
    os.makedirs("data/eda", exist_ok=True)

    df = pd.read_csv(input_path)

    plt.figure(figsize=(8, 5))
    sns.countplot(x="HeartDiseaseRisk", data=df)
    plt.savefig("data/eda/target_distribution.png")
    plt.close()

    plt.figure(figsize=(12, 8))
    sns.heatmap(df.corr(), cmap="coolwarm")
    plt.savefig("data/eda/correlation_heatmap.png")
    plt.close()

    print("✅ EDA completed")
