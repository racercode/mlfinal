from sklearn.metrics import roc_auc_score, accuracy_score
import xgboost as xgb
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
import pandas as pd
import numpy as np
from functools import partial


def xgboost(X_train, y_train, X_validation, y_validation, test_dataset):
    def xgb_objective(new_params, data):
        X_train,y_train,X_valid,y_valid, params = data
        
        if 'max_depth' in new_params.keys():
            new_params['max_depth']=int(new_params['max_depth'])
        if 'reg_alpha' in new_params.keys():
            new_params['reg_alpha']=np.exp(new_params['reg_alpha'])
        
        for x in new_params.keys():
            params[x] = new_params[x]
        
        gbm = xgb.XGBClassifier(**params, random_state=13, n_jobs=-1, early_stopping_rounds=15)
        model = gbm.fit(X_train, y_train,
                        verbose=False,
                        eval_set = [[X_train, y_train],
                                [X_valid, y_valid]],
                        )
        xgb_test_preds = model.predict(X_valid)
        proba = model.predict_proba(X_valid)[:,1]
        
        #penalize low discrimination
        score = -roc_auc_score(y_valid, xgb_test_preds)
        
        return(score)

    def get_xgbparams(space, full_params, evals=15):
        fmin_objective = partial(xgb_objective, data=(X_train,y_train,X_validation,y_validation,full_params))
        
        params = fmin(fmin_objective,
            space=space,
            algo=tpe.suggest,
            max_evals=evals)
        
        #update full_params
        for x in full_params.keys():
            if x in params.keys():
                full_params[x]=params[x]
        full_params['max_depth']=int(full_params['max_depth'])
        if 'reg_alpha' in params.keys():
            full_params['reg_alpha']=np.exp(params['reg_alpha'])
        
        return full_params

    space = {
        'max_depth':  hp.quniform('max_depth', 5, 10, 1),
        'min_child_weight': hp.quniform('min_child_weight', 2, 10, 1),
        'learning_rate': hp.quniform('learning_rate', .01, .08, .01),
        'subsample': hp.quniform('subsample', 0.1, 1, 0.1),
        'colsample_bytree': hp.quniform('colsample_bytree', 0.1, 1, 0.1),
        'reg_alpha': hp.quniform('reg_alpha', -2, 4,0.5),
    }
    xgb_params = {
        'colsample_bytree': 1.0,
        'gamma': 1e-08,
        'learning_rate': 0.1,
        'max_depth': 10,
        'min_child_weight': 10.0,
        'reg_alpha': 25.2379554214079,
        'subsample': 0.10694933059531121
    }
    xgb_params = get_xgbparams(space,xgb_params,100)
    gbm = xgb.XGBClassifier(**xgb_params, n_jobs=-1)
    model = gbm.fit(X_train, y_train,
                    verbose=True,
                    eval_set = [[X_train, y_train],
                            [X_validation, y_validation]],
                    )
    xgb_test_preds = model.predict(X_validation)
    print(accuracy_score(y_validation, xgb_test_preds))

    model = gbm.fit(X_train, y_train, verbose=True)
    return accuracy_score(y_validation, xgb_test_preds), model.predict(test_dataset)