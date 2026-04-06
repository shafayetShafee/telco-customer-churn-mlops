from typing import Any

import optuna
import pandas as pd
from sklearn.metrics import average_precision_score
from xgboost import XGBClassifier


def tune_xgb_pr_auc(
    X_train: pd.DataFrame,
    y_train: pd.Series | pd.DataFrame,
    X_valid: pd.DataFrame,
    y_valid: pd.Series | pd.DataFrame,
    n_trials: int = 50,
    random_state: int = 42
) -> tuple[XGBClassifier, dict[str, Any], optuna.Study]:
    """
    Perform hyperparameter tuning for an XGBoost classifier using Optuna,
    optimizing for PR-AUC (Average Precision).

    This function searches for the best set of hyperparameters by training
    multiple XGBoost models and evaluating their performance on a validation
    set using the average precision score, which is suitable for imbalanced
    classification problems such as churn prediction.

    Args:
        X_train (pd.DataFrame):
            Training feature data.
        y_train (pd.Series):
            Training target labels.
        X_valid (pd.DataFrame):
            Validation feature data.
        y_valid (pd.Series):
            Validation target labels.
        n_trials (int):
            Number of Optuna trials to run.
        random_state (int):
            Random seed for reproducibility.

    Returns:
        Tuple[XGBClassifier, Dict[str, Any], optuna.Study]:
            - best_model: Trained XGBoost model with optimal hyperparameters
            - best_params: Dictionary of best hyperparameters
            - study: Optuna study object containing optimization history
    """

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 300, 800),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0, 5),
            "reg_alpha": trial.suggest_float("reg_alpha", 0, 5),
            "reg_lambda": trial.suggest_float("reg_lambda", 0, 5),
            "random_state": random_state,
            "n_jobs": -1,
            "scale_pos_weight": scale_pos_weight,
            "eval_metric": "logloss"
        }

        model = XGBClassifier(**params)
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_valid)[:, 1]
        return average_precision_score(y_valid, proba)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    # Train final model with best params
    best_params = study.best_params
    best_params.update({
        "random_state": random_state,
        "n_jobs": -1,
        "scale_pos_weight": scale_pos_weight,
        "eval_metric": "logloss"
    })

    best_model = XGBClassifier(**best_params)
    best_model.fit(X_train, y_train)

    return best_model, best_params, study
