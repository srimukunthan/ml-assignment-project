import numpy as np
from src.model import train_model, evaluate_model
from sklearn.ensemble import RandomForestClassifier

def test_train_and_evaluate_model():
    # Generate dummy data with the correct number of features (31 in this case)
    X_train = np.random.rand(100, 31)  # 31 features
    y_train = np.random.randint(0, 2, size=100)
    X_test = np.random.rand(20, 31)    # 31 features
    y_test = np.random.randint(0, 2, size=20)

    # Define parameter grid
    param_grid = {
        "n_estimators": [10, 50, 100, 200],
        "max_depth": [5, 10, 15, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4]
    }

    model = train_model(X_train, y_train, param_grid)
    accuracy, _ = evaluate_model(model, X_test, y_test)

    # Check if the accuracy is within a reasonable range (between 0 and 1)
    assert 0 <= accuracy <= 1

    # Additional check to confirm that the trained model is a RandomForestClassifier
    assert isinstance(model, RandomForestClassifier)

