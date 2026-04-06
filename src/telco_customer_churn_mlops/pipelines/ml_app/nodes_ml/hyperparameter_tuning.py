from typing import Any

import optuna
import pandas as pd
from sklearn.metrics import make_scorer, average_precision_score
from xgboost import XGBClassifier
from optuna_integration import OptunaSearchCV
from optuna.distributions import IntDistribution, FloatDistribution


def tune_xgb_pr_auc(
    X_train: pd.DataFrame,
    y_train: pd.Series | pd.DataFrame,
    optuna_options: dict
) -> tuple[XGBClassifier, dict[str, Any], OptunaSearchCV]:
    """
    Perform hyperparameter tuning for an XGBoost classifier using OptunaSearchCV,
    optimizing for PR-AUC (average precision).

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series | pd.DataFrame): Training labels.
        optuna_options (dict): Dictionary containing options:
            - 'n_trials': int, number of Optuna trials
            - 'random_state': int, random seed
            - 'cv': int, number of cross-validation folds

    Returns:
        Tuple[XGBClassifier, dict[str, Any], OptunaSearchCV]:
            - best_model: fitted XGBClassifier
            - best_params: dict of best hyperparameters
            - optuna_search: the OptunaSearchCV object with full optimization results
    """
    pos_count = (y_train == 1).sum()
    neg_count = (y_train == 0).sum()
    scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1

    param_distributions = {
        "n_estimators": IntDistribution(300, 800, step=100),
        "learning_rate": FloatDistribution(0.01, 0.2, log=True),
        "max_depth": IntDistribution(3, 10),
        "subsample": FloatDistribution(0.5, 1.0),
        "colsample_bytree": FloatDistribution(0.5, 1.0),
        "min_child_weight": IntDistribution(1, 15),
        "gamma": FloatDistribution(0.0, 5.0),
        "reg_alpha": FloatDistribution(1e-8, 10.0, log=True),
        "reg_lambda": FloatDistribution(1e-8, 10.0, log=True),
    }

    xgb_clf = XGBClassifier(
        random_state=optuna_options.get("random_state", 42),
        n_jobs=-1,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr"
    )

    pr_auc_scorer = make_scorer(
        average_precision_score, 
        response_method="predict_proba"
    )

    optuna_search = OptunaSearchCV(
        estimator=xgb_clf,
        param_distributions=param_distributions,
        n_trials=optuna_options.get("n_trials", 50),
        cv=optuna_options.get("cv", 3),
        scoring=pr_auc_scorer,
        random_state=optuna_options.get("random_state", 42),
        verbose=1
    )

    optuna_search.fit(X_train, y_train)

    best_model = optuna_search.best_estimator_
    best_params = optuna_search.best_params_

    return best_model, best_params, optuna_search