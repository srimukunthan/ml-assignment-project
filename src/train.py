from preprocess import load_data, preprocess_data
from model import train_model, evaluate_model

def main():
    # Load and preprocess data
    data = load_data("data/winequality-red.csv")
    X_train, X_test, y_train, y_test = preprocess_data(data)

    # Train the model
    model = train_model(X_train, y_train)

    # Evaluate the model
    accuracy, report = evaluate_model(model, X_test, y_test)
    print(f"Accuracy: {accuracy}")
    print(f"Classification Report:\n{report}")

if __name__ == "__main__":
    main()
