import argparse
import os
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, f1_score
from mlflow.models import infer_signature
from dotenv import load_dotenv
import joblib

# Argparse untuk CLI
parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, default="dataset_preprocessing")
args = parser.parse_args()

# Load .env
load_dotenv()
os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("MLFLOW_TRACKING_USERNAME")
os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("MLFLOW_TRACKING_PASSWORD")

mlflow.set_tracking_uri("https://dagshub.com/DianAzizah13/Membangun-Model.mlflow")
mlflow.set_experiment("Crop Recommendation - Random Forest + GridSearch")

# Load data
X_train = pd.read_csv("Code/Eksperimen_SML/preprocessing/data_preprocessing/X_train.csv")
X_test = pd.read_csv("Code/Eksperimen_SML/preprocessing/data_preprocessing/X_test.csv")
y_train = pd.read_csv("Code/Eksperimen_SML/preprocessing/data_preprocessing/y_train.csv").values.squeeze()
y_test = pd.read_csv("Code/Eksperimen_SML/preprocessing/data_preprocessing/y_test.csv").values.squeeze()

# GridSearchCV
param_grid = {
    "n_estimators": [50, 100],
    "max_depth": [5, 10, None],
    "min_samples_split": [2, 5]
}

with mlflow.start_run(run_name="RF_GridSearch_CI"):
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(rf, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    for param_name, param_value in grid_search.best_params_.items():
        mlflow.log_param(param_name, param_value)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")
    precision = precision_score(y_test, y_pred, average="macro")

    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_score_macro", f1)
    mlflow.log_metric("precision_macro", precision)

    signature = infer_signature(X_train.iloc[:5], best_model.predict(X_train.iloc[:5]))

    mlflow.sklearn.log_model(
        sk_model=best_model,
        artifact_path="model",
        input_example=X_train.iloc[:5],
        signature=signature
    )

    print("[INFO] Training selesai dan model dilog ke MLflow.")
