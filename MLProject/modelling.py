import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def main():
    
    mlflow.set_experiment("Crop Recommendation - Random Forest")

    mlflow.sklearn.autolog(log_input_examples=True, log_model_signatures=True)

    try:
        X_train = pd.read_csv("data_preprocessing/X_train.csv")
        X_test = pd.read_csv("data_preprocessing/X_test.csv")
        y_train = pd.read_csv("data_preprocessing/y_train.csv").values.ravel()
        y_test = pd.read_csv("data_preprocessing/y_test.csv").values.ravel()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        exit()

    with mlflow.start_run(run_name="RF_Without_Tuning"):
        rf = RandomForestClassifier(random_state=42)
        rf.fit(X_train, y_train)

        preds = rf.predict(X_test)
        acc = accuracy_score(y_test, preds)

        print(f"[Without Tuning] Accuracy: {acc:.4f}")
        print("\nMLflow autologging selesai.")

if __name__ == "__main__":
    main()
