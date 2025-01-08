from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report



def train_model(X_train, y_train):
    """Train a Random Forest classifier."""
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model



def evaluate_model(model, X_test, y_test):
    """Evaluate the model on the test set."""
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    return accuracy, report