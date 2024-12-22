from sklearn.metrics import roc_auc_score, accuracy_score
import xgboost as xgb
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
import pandas as pd

def xgboost(X_train, y_train, X_validation, y_validation, test_dataset):
    raw_X_train = X_train
    raw_y_train = y_train

    X_test = X_train.sample(frac=0.2, random_state=42)
    y_test = y_train.sample(frac=0.2, random_state=42)
    X_train = X_train.drop(X_test.index)
    y_train = y_train.drop(y_test.index)

    space = {
        'max_depth': hp.quniform("max_depth", 3, 20, 1),
        'gamma': hp.uniform('gamma', 0, 5),
        'reg_alpha': hp.uniform('reg_alpha', 0, 50),
        'reg_lambda': hp.uniform('reg_lambda', 0, 1),
        'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1),
        'min_child_weight': hp.quniform('min_child_weight', 1, 10, 1),
        'learning_rate': hp.uniform('learning_rate', 0.01, 0.3),
        'n_estimators': 300,  # Early stopping will determine the effective count
    }

    # Objective function for hyperparameter tuning
    def objective(params):
        params['max_depth'] = int(params['max_depth'])
        params['min_child_weight'] = int(params['min_child_weight'])
        
        clf = xgb.XGBClassifier(
            **params,
            objective='binary:logistic',
            eval_metric='logloss',
            early_stopping_rounds=10,
            seed=42,
            enable_categorical=True,  # Enable categorical handling
        )
        
        # Training with early stopping
        clf.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False,
        )
        
        preds = clf.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, preds)
        return {'loss': -auc, 'status': STATUS_OK}

    # Perform hyperparameter tuning
    trials = Trials()
    best_hyperparams = fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        max_evals=50,
        trials=trials,
    )

    # Train final model with the best hyperparameters
    best_hyperparams['max_depth'] = int(best_hyperparams['max_depth'])
    best_hyperparams['min_child_weight'] = int(best_hyperparams['min_child_weight'])
    
    final_model = xgb.XGBClassifier(
        **best_hyperparams,
        objective='binary:logistic',
        eval_metric='logloss',
        seed=42,
        enable_categorical=True,  # Enable categorical handling
    )
    
    final_model.fit(raw_X_train, raw_y_train)

    # Validate final model on the validation set
    val_preds = final_model.predict_proba(X_validation)[:, 1]
    validation_auc = roc_auc_score(y_validation, val_preds)
    validation_accuracy = accuracy_score(y_validation, (val_preds > 0.5).astype(int))
    print("Validation AUC:", validation_auc)

    # Train final model with both train and validation sets for testing
    final_test_model = xgb.XGBClassifier(
        **best_hyperparams,
        objective='binary:logistic',
        eval_metric='logloss',
        seed=42,
        enable_categorical=True,  # Enable categorical handling
    )
    
    final_test_model.fit(pd.concat([raw_X_train, X_validation]), pd.concat([raw_y_train, y_validation]))

    importance = final_test_model.feature_importances_
    important_features = pd.Series(importance, index=X_train.columns).sort_values(ascending=True)[:20]
    important_features.plot(kind='bar', title='Feature Importance')

    # Predict on the test dataset
    test_preds = final_test_model.predict_proba(test_dataset)[:, 1]
    test_output = (test_preds > 0.5).astype(int)

    return validation_accuracy, test_output
