# utils/model_handler.py
import os
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from xgboost import XGBClassifier, XGBRegressor
from config import MODEL_SAVE_PATH

# Define supported models (classification + regression)
MODEL_MAP = {
    # Classification models
    "logistic_regression": LogisticRegression(max_iter=500),
    "svm_linear": SVC(kernel="linear"),
    "svm_rbf": SVC(kernel="rbf"),
    "knn_classifier": KNeighborsClassifier(),
    "random_forest": RandomForestClassifier(n_estimators=100),
    "xgboost_classifier": XGBClassifier(use_label_encoder=False, eval_metric="mlogloss"),
    
    # Regression models
    "linear_regression": LinearRegression(),
    "svm_regressor": SVR(),
    "knn_regressor": KNeighborsRegressor(),
    "random_forest_regressor": RandomForestRegressor(n_estimators=100),
    "xgboost_regressor": XGBRegressor()
}

def is_regression(model_name):
    return "regression" in model_name or "regressor" in model_name

def train_model(data, target_column, model_name="random_forest"):
    if model_name not in MODEL_MAP:
        raise ValueError(f"Model '{model_name}' not supported. Choose from {list(MODEL_MAP.keys())}")
    
    X = data.drop(columns=[target_column])
    y = data[target_column]

    model = MODEL_MAP[model_name]
    model.fit(X, y)

    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
    model_file = os.path.join(MODEL_SAVE_PATH, f"{model_name}.pkl")
    joblib.dump(model, model_file)

    return model_file

def test_model(data, target_column, model_name="random_forest"):
    model_file = os.path.join(MODEL_SAVE_PATH, f"{model_name}.pkl")
    if not os.path.exists(model_file):
        raise FileNotFoundError("Model not trained yet. Train before testing.")

    model = joblib.load(model_file)
    X = data.drop(columns=[target_column])
    y_true = data[target_column]

    if is_regression(model_name):
        y_pred = model.predict(X)
        mse = mean_squared_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        return {"mse": mse, "r2": r2}
    else:
        y_pred = model.predict(X)
        accuracy = accuracy_score(y_true, y_pred)
        return {"accuracy": accuracy}

def predict_datapoint(json_data, model_name="random_forest"):
    model_file = os.path.join(MODEL_SAVE_PATH, f"{model_name}.pkl")
    if not os.path.exists(model_file):
        raise FileNotFoundError("Model not trained yet. Train before predicting.")

    model = joblib.load(model_file)
    input_df = [list(json_data.values())]

    prediction = model.predict(input_df)

    if is_regression(model_name):
        return float(prediction[0])
    else:
        return int(prediction[0])
