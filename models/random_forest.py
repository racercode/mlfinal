import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def random_forest(X_train, y_train, X_validation, y_validation, test_dataset):
    """
    Performs hyperparameter tuning for a Random Forest model and returns predictions.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training labels.
        X_validation (pd.DataFrame): Validation features.
        y_validation (pd.Series): Validation labels.
        test_dataset (pd.DataFrame): Testing features.

    Returns:
        tuple: (accuracy, test_y_predicted)
            - accuracy (float): Accuracy score on the validation set.
            - test_y_predicted (pd.Series): Predicted labels on the test set.
    """

    # Define hyperparameter search space
    param_dist = {
        'n_estimators': np.arange(100, 501, 50),  # Number of trees (adjust range as needed)
        'max_depth': np.arange(3, 10),             # Maximum depth of trees (adjust range)
        'min_samples_split': np.arange(2, 11),      # Minimum samples to split a node
        'min_samples_leaf': np.arange(1, 6),         # Minimum samples in a leaf node
        'max_features': ['auto', 'sqrt', 'log2'],  # Number of features considered
    }

    # Use RandomizedSearchCV for efficient hyperparameter tuning
    rfc = RandomForestClassifier()
    random_search = RandomizedSearchCV(estimator=rfc, param_distributions=param_dist,
                                       n_iter=50, scoring='accuracy', cv=5)  # Adjust n_iter and cv as needed
    random_search.fit(X_train, y_train)

    # Get best hyperparameters and create a tuned model
    best_params = random_search.best_params_
    tuned_forest = RandomForestClassifier(**best_params)
    tuned_forest.fit(X_train, y_train)

    # Make predictions on validation and test sets
    validation_y_predicted = tuned_forest.predict(X_validation)
    test_y_predicted = tuned_forest.predict(test_dataset)

    # Evaluate performance on validation set
    accuracy = accuracy_score(y_validation, validation_y_predicted)
    print("Accuracy on validation set:", accuracy)

    # Provide classification report for further analysis (optional)
    print(classification_report(y_validation, validation_y_predicted))

    return accuracy, test_y_predicted

# Example usage (assuming you have your data loaded)
