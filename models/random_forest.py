def random_forest(X_train, y_train, X_validation, y_validation, test_dataset):
    from sklearn import ensemble, preprocessing, metrics

    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report
    import xgboost as xgb

    # Load or create your dataset
    # Example: df = pd.read_csv('your_dataset.csv')
    # Assume 'features' are the independent variables and 'target' is the binary dependent variable

    # For demonstration, let's create a synthetic datase

    # Initialize the XGBoost classifier
    forest = ensemble.RandomForestClassifier(n_estimators = 100)
    forest_fit = forest.fit(X_train, y_train)

    # 預測
    test_y_predicted = forest.predict(X_validation)

    # 績效
    accuracy = metrics.accuracy_score(y_validation, test_y_predicted)
    print(accuracy)
    return accuracy, test_y_predicted