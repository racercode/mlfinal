from catboost import CatBoostClassifier

def catboost(X_train, y_train, X_validation, y_validation, test_dataset):
    model = CatBoostClassifier(iterations=1000, learning_rate=0.1, depth=6, loss_function='MultiClass', verbose=True)
    model.fit(X_train, y_train, eval_set=(X_validation, y_validation))
    result = model.predict(test_dataset)
    accuracy = model.score(X_validation, y_validation)
    return accuracy, result