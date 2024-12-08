def knn(X_train, y_train, X_validation, y_validation, test_dataset):
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import accuracy_score

    model = KNeighborsClassifier()
    model.fit(X_train, y_train)
    result = model.predict(X_validation)
    accuracy = accuracy_score(y_validation, result)
    return accuracy, model.predict(test_dataset)