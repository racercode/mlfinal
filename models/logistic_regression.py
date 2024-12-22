def grid_search(X_train, y_train):
    from sklearn.model_selection import GridSearchCV
    from sklearn.linear_model import LogisticRegression

    param_grid = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear']
    }

    grid_search = GridSearchCV(
        LogisticRegression(max_iter=400),
        param_grid,
        cv=3,
        scoring='accuracy',
        verbose=1
    )

    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    print(grid_search.best_params_)
    return best_model


def logistic_regression(X_train, y_train, X_validation, y_validation, test_dataset):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    from sklearn.preprocessing import PolynomialFeatures
    import numpy as np
    import pandas as pd

    X_train = pd.concat([X_train, X_validation])
    y_train = pd.concat([y_train, y_validation])

    poly = PolynomialFeatures(degree=1, include_bias=True)
    X_train = poly.fit_transform(X_train)
    X_validation = poly.transform(X_validation)
    test_dataset = poly.transform(test_dataset)

    model = grid_search(X_train, y_train)
    result = model.predict(X_validation)
    accuracy = accuracy_score(y_validation, result)


    proba = model.predict_proba(test_dataset)
    return accuracy, [1 if prob[1] > 0.5 else 0 for prob in proba]