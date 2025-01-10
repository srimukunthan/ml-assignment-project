import joblib
import numpy as np


# Load the model and scaler
def load_model(model_path):
    """Load the trained model from a file."""
    try:
        model = joblib.load(model_path)
        print("Model loaded successfully.")
        return model
    except FileNotFoundError:
        print(f"Model file not found at {model_path}. Please ensure the model is trained and saved.")
        exit(1)


def load_scaler(scaler_path):
    """Load the scaler from a file."""
    try:
        scaler = joblib.load(scaler_path)
        print("Scaler loaded successfully.")
        return scaler
    except FileNotFoundError:
        print(f"Scaler file not found at {scaler_path}. Please ensure the scaler is saved.")
        exit(1)


def preprocess_input(input_features, scaler):
    """Preprocess the input features to match the model's expected format."""
    input_array = np.array(input_features).reshape(1, -1)
    print("Original input array shape:", input_array.shape)
    
    # Ensure the input data has the correct number of features (31)
    if input_array.shape[1] != 31:  # Adjust this number to match the training feature count
        raise ValueError(f"Expected 31 features, but got {input_array.shape[1]} features.")
    
    # Apply scaling transformation using the loaded scaler
    return scaler.transform(input_array)


def predict(model, input_features, scaler):
    """Make predictions using the trained model."""
    print("Input features:", input_features)
    input_array = preprocess_input(input_features, scaler)  # Preprocess input before prediction
    prediction = model.predict(input_array)
    print("Prediction:", prediction)
    return prediction


def main():
    model_path = "artifacts/model.joblib"
    scaler_path = "artifacts/scaler.joblib"

    # Load the model and scaler
    model = load_model(model_path)
    scaler = load_scaler(scaler_path)

    # Define the input features (ensure these match the expected format)
    example_input_1 = [
        17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 0.3001, 0.1471, 0.2419, 0.07871, 1.095, 0.9053, 
        8.589, 153.4, 0.006399, 0.04904, 0.05373, 0.01587, 0.03003, 0.006193, 25.38, 17.33, 184.6, 2019.0, 
        0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189, 0.1189
    ]

    example_input_2 = [
        20.57, 17.77, 132.9, 1140.0, 0.09622, 0.08143, 0.06124, 0.04757, 0.05101, 0.04394, 0.1294, 0.1188, 
        0.2722, 0.004788, 0.01456, 0.01724, 0.003878, 0.002134, 0.001300, 0.000476, 0.000220, 0.002225, 
        0.008845, 0.01423, 0.006298, 0.000549, 0.001433, 0.000670, 0.000258, 0.000385, 0.000000
    ]

    # Choose which input to use
    example_input = example_input_2  # Change this to example_input_2 for the second example

    # Make prediction
    prediction = predict(model, example_input, scaler)
    print(f"Predicted class: {prediction[0]}")


if __name__ == "__main__":
    main()
