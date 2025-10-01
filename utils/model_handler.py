# utils/model_handler.py
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from config import MODEL_SAVE_PATH

# Define supported models
MODEL_MAP = {
    "logistic_regression": LogisticRegression(max_iter=500),
    "random_forest": RandomForestClassifier(n_estimators=100),
    "svm": SVC()
}

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
    y_pred = model.predict(X)
    accuracy = accuracy_score(y_true, y_pred)

    return accuracy

def predict_datapoint(json_data, model_name="random_forest"):
    model_file = os.path.join(MODEL_SAVE_PATH, f"{model_name}.pkl")
    if not os.path.exists(model_file):
        raise FileNotFoundError("Model not trained yet. Train before predicting.")

    model = joblib.load(model_file)
    input_df = [list(json_data.values())]
    prediction = model.predict(input_df)
    return prediction[0]
