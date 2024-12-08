def decision_tree(X_train, y_train, X_validation, y_validation, test_dataset):
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import accuracy_score

    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_validation)
    accuracy = accuracy_score(y_validation, y_pred)

    if test_dataset is not None:
        y_test = model.predict(test_dataset)
        return accuracy, y_test
    return accuracy, None