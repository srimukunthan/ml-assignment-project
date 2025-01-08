import numpy as np
from src.model import train_model, evaluate_model

def test_train_and_evaluate_model():
    X_train = np.random.rand(100, 5)
    y_train = np.random.randint(0, 2, size=100)
    X_test = np.random.rand(20, 5)
    y_test = np.random.randint(0, 2, size=20)

    model = train_model(X_train, y_train)
    accuracy, _ = evaluate_model(model, X_test, y_test)
    assert 0 <= accuracy <= 1
