from typing import Any

import mlflow
import pandas as pd
from optuna.distributions import FloatDistribution, IntDistribution
from optuna_integration import OptunaSearchCV
from sklearn.metrics import average_precision_score, make_scorer
from xgboost import XGBClassifier

from telco_customer_churn_mlops.configs import OptunaConfig


def tune_xgb_pr_auc(
    X_train: pd.DataFrame, y_train: pd.Series | pd.DataFrame, optuna_options: dict
) -> tuple[XGBClassifier, dict[str, Any], OptunaSearchCV]:
    """
    Perform hyperparameter tuning for an XGBoost classifier using OptunaSearchCV,
    optimizing for PR-AUC (average precision).

    Parameters
    ----------
    X_train : pd.DataFrame
        Training feature matrix.

    y_train : pd.Series or pd.DataFrame
        Training target values.

    optuna_options : dict
        Dictionary of Optuna configuration options. Validated using OptunaConfig.
        Expected keys include:
        - 'n_trials' : int
            Number of Optuna trials.
        - 'random_state' : int
            Random seed for reproducibility.
        - 'cv' : int
            Number of cross-validation folds.

    Returns
    -------
    best_model : XGBClassifier
        Fitted model with the best-found hyperparameters.
    best_params : dict of str to Any
        Best hyperparameter configuration identified by Optuna.
    optuna_search : OptunaSearchCV
        Fitted OptunaSearchCV object containing full optimization results.

    Notes
    -----
    The function returns a tuple of (best_model, best_params, optuna_search).
    """
    pos_count = (y_train == 1).sum()
    neg_count = (y_train == 0).sum()
    scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1
    mlflow.log_param("scale_pos_weight", scale_pos_weight)

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
        eval_metric="aucpr",
    )

    pr_auc_scorer = make_scorer(
        average_precision_score, response_method="predict_proba"
    )

    optuna_cfg = OptunaConfig.from_params(optuna_options)

    optuna_search = OptunaSearchCV(
        estimator=xgb_clf,
        param_distributions=param_distributions,
        n_trials=optuna_cfg.n_trials,
        cv=optuna_cfg.cv,
        scoring=pr_auc_scorer,
        random_state=optuna_cfg.random_state,
        verbose=1,
    )

    optuna_search.fit(X_train, y_train)

    best_model = optuna_search.best_estimator_
    best_params = optuna_search.best_params_

    mlflow.log_params(_sanitize(best_params))

    return best_model, best_params, optuna_search


def _sanitize(params: dict) -> dict:
    """
    Convert parameter values to JSON-serializable Python scalars.

    Parameters
    ----------
    params : dict
        Dictionary of parameters, potentially containing NumPy scalar types.

    Returns
    -------
    dict
        Dictionary with values converted to native Python types where applicable.

    Notes
    -----
    Values exposing a `.item()` method (e.g., NumPy scalars) are converted
    using that method. Other values are returned unchanged.
    """
    return {k: (v.item() if hasattr(v, "item") else v) for k, v in params.items()}
