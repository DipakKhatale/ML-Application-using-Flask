# app.py
import os
from flask import Flask, request, jsonify
from utils.data_handler import allowed_file, load_dataset
from utils.model_handler import train_model, test_model, predict_datapoint
from config import ALLOWED_EXTENSIONS, DEFAULT_MODEL

app = Flask(__name__)

@app.route("/train", methods=["POST"])
def train():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files["file"]
    target_column = request.form.get("target_column")
    model_name = request.form.get("model", DEFAULT_MODEL)

    if not file or not allowed_file(file.filename, ALLOWED_EXTENSIONS):
        return jsonify({"error": "Invalid file format"}), 400
    
    file_path = os.path.join("uploads", file.filename)
    os.makedirs("uploads", exist_ok=True)
    file.save(file_path)

    try:
        data = load_dataset(file_path)
        if target_column not in data.columns:
            return jsonify({"error": f"Target column '{target_column}' not found"}), 400
        
        model_file = train_model(data, target_column, model_name)
        return jsonify({"message": f"Model trained and saved at {model_file}"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/test", methods=["POST"])
def test():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files["file"]
    target_column = request.form.get("target_column")
    model_name = request.form.get("model", DEFAULT_MODEL)

    if not file or not allowed_file(file.filename, ALLOWED_EXTENSIONS):
        return jsonify({"error": "Invalid file format"}), 400
    
    file_path = os.path.join("uploads", file.filename)
    os.makedirs("uploads", exist_ok=True)
    file.save(file_path)

    try:
        data = load_dataset(file_path)
        if target_column not in data.columns:
            return jsonify({"error": f"Target column '{target_column}' not found"}), 400
        
        accuracy = test_model(data, target_column, model_name)
        return jsonify({"accuracy": accuracy})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/predict", methods=["POST"])
def predict():
    model_name = request.form.get("model", DEFAULT_MODEL)
    json_data = request.get_json()

    if not json_data:
        return jsonify({"error": "No JSON data provided"}), 400

    try:
        prediction = predict_datapoint(json_data, model_name)
        return jsonify({"prediction": str(prediction)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
