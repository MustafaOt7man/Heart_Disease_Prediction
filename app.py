from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import json
import os

app = Flask(__name__)

# Load the trained ECG classification model
model = load_model(r"D:\ai\Deployed_Heart_Disease_withflask\ecg_heartbeat_model.keras")

# Map class labels to human-readable names
class_map = {
    0: "Normal",
    1: "Supraventricular premature",
    2: "Premature ventricular contraction",
    3: "Fusion of ventricular and normal",
    4: "Unclassifiable"
}

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        return super().default(obj)

@app.route('/predict-ecg', methods=['POST'])
def predict_ecg():
    try:
        # Expect JSON body with an "ECG" key
        data = request.json
        if "ECG" not in data:
            return jsonify({"error": "Missing ECG data"}), 400

        ecg_signal = np.array(data["ECG"]).flatten()
        if len(ecg_signal) < 187:
            return jsonify({"error": "ECG signal too short. Minimum 187 samples required."}), 400

        # Segment into 187-sample non-overlapping windows
        segment_length = 187
        segments = [
            ecg_signal[i:i + segment_length]
            for i in range(0, len(ecg_signal) - segment_length + 1, segment_length)
        ]
        X = np.array(segments)

        # Normalize using MinMaxScaler
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)

        # Predict
        predictions = model.predict(X_scaled)
        predicted_classes = np.argmax(predictions, axis=1)

        # Get the most frequent class
        (unique, counts) = np.unique(predicted_classes, return_counts=True)
        dominant_class = unique[np.argmax(counts)]
        dominant_prediction = class_map[dominant_class]

        # Response
        response = {
            "prediction": dominant_prediction
        }

        return app.response_class(
            response=json.dumps(response, cls=NumpyEncoder),
            status=200,
            mimetype='application/json'
        )

    except Exception as e:
        import traceback
        return jsonify({
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy"})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
