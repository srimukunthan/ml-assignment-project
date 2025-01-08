from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

def train_model(X_train, y_train):
    """
    Trains a Random Forest model with hyperparameter tuning using GridSearchCV.
    """
    # Define the model
    rf = RandomForestClassifier(random_state=42)

    # Define hyperparameter grid
    param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
    }

    # Use GridSearchCV for hyperparameter tuning
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, scoring='accuracy', verbose=1)
    grid_search.fit(X_train, y_train)

    # Get the best model
    best_model = grid_search.best_estimator_

    print("Best parameters found:", grid_search.best_params_)
    return best_model

def evaluate_model(model, X_test, y_test):
    """
    Evaluates the model on the test set.
    """
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    return accuracy
