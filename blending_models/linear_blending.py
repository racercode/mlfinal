def linear_blending(X_train, y_train, X_validation, y_validation, test_dataset):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score

    model = LogisticRegression()
    model.fit(X_train, y_train)
    result = model.predict(X_validation)
    accuracy = accuracy_score(y_validation, result)
    return accuracy, model.predict(test_dataset)