import pandas as pd
from src.preprocess import preprocess_data

def test_preprocess_data():
    # Sample data with 31 features (similar to the model input requirements)
    data = pd.DataFrame({
        "feature" + str(i): [1, 2, 3, 4] for i in range(1, 32)  # 31 features
    })
    data["quality"] = [3, 7, 5, 8]  # Target variable

    X_train, X_test, y_train, y_test = preprocess_data(data)

    # Check the shapes of the splits
    assert len(X_train) == 3  # 75% for training
    assert len(X_test) == 1   # 25% for testing
    assert len(y_train) == 3
    assert len(y_test) == 1

    # Check that the feature columns have been processed correctly
    assert X_train.shape[1] == 31  # 31 features
    assert X_test.shape[1] == 31   # 31 features
