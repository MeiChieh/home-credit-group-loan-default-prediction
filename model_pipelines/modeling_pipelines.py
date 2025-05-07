# import sklearnex
# sklearnex.unpatch_sklearn()
import joblib
from hepler import *
from imblearn.pipeline import Pipeline as imb_pipeline
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, TargetEncoder
import numpy as np
import pandas as pd
from typing import Callable

from hepler.optuna_transformers_and_estimators import balanced_lgbm_clf, balanced_logit, balanced_rf, imputers, lgbm_clf, scalers, smote

def objective(trial: Trial, x: np.ndarray | pd.DataFrame, y: np.ndarray, pipeline: Callable, cv: int | StratifiedKFold = 5):
    """
    Objective function for Optuna to optimize

    Args:
        trial (Trial): Optuna trial
        x (np.ndarray | pd.DataFrame): dependent variable
        y (np.ndarray): independent variable
        pipeline (Callable): pipeline wrapper with trial as parameter
        cv (int | StratifiedKFold, optional): cv splits. Defaults to 5.

    Returns:
        _type_: metric score of pipeline (model)
    """
    model = pipeline(trial)

    score = cross_val_score(model, x, y, n_jobs=-1, cv=cv,
                            scoring="roc_auc")
    roc_auc = score.mean()

    return roc_auc




def lgbm_balanced_pipeline(trial: Trial):
    """
    lgbm class_balanced with internal imputation pipeline for prediction

    Args:
        trial (Trial): Optuna trial

    Returns:
        Pipeline: pipeline
    """

    column_transformer = ColumnTransformer(
        transformers=[
            ("categorcal_pipeline", TargetEncoder(), cat_cols),
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")

    app_imputer = joblib.load("data/models/adjusted_app_df_transformer2")

    pipeline = Pipeline(
        [
            ("app_imputer", app_imputer),
            ("column_transformer", column_transformer),
            ("model", balanced_lgbm_clf(trial)),
        ]
    )

    return pipeline

def lgbm_smote_pipeline(trial: Trial):
    """
    lgbm class_balanced with internal imputation pipeline for prediction

    Args:
        trial (Trial): Optuna trial

    Returns:
        Pipeline: pipeline
    """
    
    num_pipe = Pipeline(
        steps=[("imputer", imputers(trial))]
    )

    column_transformer = ColumnTransformer(
        transformers=[
            ("numerical_pipeline", num_pipe, num_cols),
            ("categorcal_pipeline", TargetEncoder(), cat_cols),
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")

    app_imputer = joblib.load("data/models/adjusted_app_df_transformer2")

    pipeline = imb_pipeline(
        [
            ("app_imputer", app_imputer),
            ("column_transformer", column_transformer),
            ("smote", smote(trial)),
            ("model", lgbm_clf(trial)),
        ]
    )

    return pipeline


def balanced_rf_pipeline(trial: Trial):
    """
    BalancedRandomForest pipeline for prediction

    Args:
        trial (Trial): Optuna trial

    Returns:
        Pipeline: pipeline
    """

    num_pipe = Pipeline(
        steps=[("imputer", imputers(trial)), ("scaler", scalers(trial))]
    )

    column_transformer = ColumnTransformer(
        transformers=[
            ("numerical_pipeline", num_pipe, num_cols),
            ("categorcal_pipeline", TargetEncoder(), cat_cols),
        ],
        remainder="passthrough",
    )

    pipeline = Pipeline(
        [
            ("column_transformer", column_transformer),
            ("model", balanced_rf(trial)),
        ]
    )

    return pipeline


def balanced_logit_pipeline(trial: Trial) -> Pipeline:
    """
    class balanced logistic regression pipeline for prediction

    Args:
        trial (Trial): Optuna trial

    Returns:
        Pipeline: pipeline
    """
    
    num_pipe = Pipeline(
        steps=[
            ("imputer", imputers(trial)), 
            ("scaler", scalers(trial))]
    )

    column_transformer = ColumnTransformer(
        transformers=[
            ("numerical_pipeline", num_pipe, num_cols),
            ("categorcal_pipeline", TargetEncoder(), cat_cols),
        ],
        remainder="passthrough",
    )
    
    app_imputer = joblib.load("data/models/adjusted_app_df_transformer2")

    pipeline = imb_pipeline(
        [
            ('app_imputer', app_imputer),
            ("column_transformer", column_transformer),
            ("model", balanced_logit(trial)),
        ]
    )

    return pipeline


def app_lgbm_balanced_pipeline(trial: Trial):
    """
    lgbm class_balanced with internal imputation pipeline for prediction

    Args:
        trial (Trial): Optuna trial

    Returns:
        Pipeline: pipeline
    """

    column_transformer = ColumnTransformer(
        transformers=[
            ("categorcal_pipeline", TargetEncoder(), app_cat_cols),
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")


    pipeline = Pipeline(
        [
            ("column_transformer", column_transformer),
            ("model", balanced_lgbm_clf(trial)),
        ]
    )

    return pipeline

def app_balanced_rf_pipeline(trial: Trial):
    """
    BalancedRandomForest pipeline for prediction

    Args:
        trial (Trial): Optuna trial

    Returns:
        Pipeline: pipeline
    """

    num_pipe = Pipeline(
        steps=[("imputer", imputers(trial)), ("scaler", scalers(trial))]
    )

    column_transformer = ColumnTransformer(
        transformers=[
            ("numerical_pipeline", num_pipe, app_num_cols),
            ("categorcal_pipeline", TargetEncoder(), app_cat_cols),
        ],
        remainder="passthrough",
    )

    pipeline = Pipeline(
        [
            ("column_transformer", column_transformer),
            ("model", balanced_rf(trial)),
        ]
    )

    return pipeline