from flask import Flask, request, jsonify
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler


app = Flask(__name__)

# Load the saved model: path is for docker
model = joblib.load("/app/model.joblib")
scaler = joblib.load("/app/scaler.joblib")

@app.route("/")
def home():
    return "RandomForest Model is Ready for Predictions!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get input data as JSON
        data = request.get_json()

        # Ensure the data is in the correct format (2D array)
        input_features = np.array(data["features"]).reshape(1, -1)
        scaled_data = scaler.transform(input_features)

        # Make a prediction
        prediction = model.predict(scaled_data)

        # Return the prediction as a JSON response
        response = {"prediction": int(prediction[0])}
        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(host='0.0.0.0',port=5050, debug=True)
 