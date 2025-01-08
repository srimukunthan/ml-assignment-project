import pandas as pd
from src.preprocess import preprocess_data

def test_preprocess_data():
    data = pd.DataFrame({
        "feature1": [1, 2, 3, 4],
        "feature2": [5, 6, 7, 8],
        "quality": [3, 7, 5, 8],
    })
    X_train, X_test, y_train, y_test = preprocess_data(data)
    assert len(X_train) == 3
    assert len(X_test) == 1
    assert len(y_train) == 3
    assert len(y_test) == 1
