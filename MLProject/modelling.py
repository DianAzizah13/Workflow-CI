import mlflow
import mlflow.sklearn
import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def main():
    mlflow.set_experiment("Crop Recommendation - Random Forest")
    mlflow.sklearn.autolog()

    X_train = pd.read_csv("data_preprocessing/X_train.csv")
    X_test = pd.read_csv("data_preprocessing/X_test.csv")
    y_train = pd.read_csv("data_preprocessing/y_train.csv").values.ravel()
    y_test = pd.read_csv("data_preprocessing/y_test.csv").values.ravel()

    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train, y_train)

    preds = rf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Accuracy: {acc:.4f}")

    # Simpan model ke folder MLProject/model
    os.makedirs("model", exist_ok=True)
    mlflow.sklearn.save_model(rf, path="model")

if __name__ == "__main__":
    main()
