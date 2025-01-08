import os
import joblib
from preprocess import preprocess_data
from model import train_model, evaluate_model
from sklearn.model_selection import train_test_split

def main():
    # Ensure artifacts directory exists
    if not os.path.exists('artifacts'):
        os.makedirs('artifacts')

    # Preprocess the data
    X, y = preprocess_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model = train_model(X_train, y_train)

    # Evaluate the model
    accuracy, report = evaluate_model(model, X_test, y_test)
    print(f"Accuracy: {accuracy}")
    print(f"Classification Report:\n{report}")
    
    # Save the trained model
    joblib.dump(model, "artifacts/model.joblib")
    print("Model saved to artifacts/model.joblib")

if __name__ == "__main__":
    main()
