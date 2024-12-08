from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score

def lightgbm(X_train, y_train, X_validation, y_validation, test_dataset):
    model = LGBMClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_validation)
    accuracy = accuracy_score(y_validation, y_pred)
    result = model.predict(test_dataset)
    return accuracy, result