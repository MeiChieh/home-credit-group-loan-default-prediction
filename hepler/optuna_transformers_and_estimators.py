# import sklearnex
# sklearnex.unpatch_sklearn()
from joblib.parallel import _verbosity_filter
from sklearn.svm import SVC
from hepler import *
# sklearn
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.base import TransformerMixin
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression
# imblearn
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.over_sampling import SMOTE

# boosting models
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# tune
from optuna import Trial

## Transformers
def imputers(trial: Trial) -> SimpleImputer:
    """
    Wrapper function for SimpleImputer transformer which enables Optuna to try out and select the most optimal imputer for the pipeline.
    
    Args:
        trial (Trial): Optuna trial

    Returns:
        SimpleImputer(Transformer): imputer
    """
    strategy = trial.suggest_categorical(
        "num_impute", ["mean", "median", "most_frequent"]
    )
    return SimpleImputer(strategy=strategy)


def scalers(trial: Trial) -> TransformerMixin:
    scaler = trial.suggest_categorical("scaler", ["robust", "standard", "minmax"])
    """
    Wrapper function for different scalers which enables Optuna to try out and select the most optimal scaler for the pipeline.
    
    Args:
        trial (Trial): Optuna trial

    Returns:
        Scaler(Transformer): scaler
    """

    if scaler == "robust":
        return RobustScaler()
    elif scaler == "minmax":
        return MinMaxScaler()
    else:
        return StandardScaler()

def smote(trial: Trial) -> SMOTE:
    """
    Wrapper function for SMOTE which enables Optuna to try out and select the most optimal SMOTE k_neighbors parameter for the pipeline.
    
    Args:
        trial (Trial): Optuna trial

    Returns:
        SMOTE(Transformer): SMOTE transformer
    """
    k_neighbors = trial.suggest_int("k_neighbors", 2, 15)

    return SMOTE(k_neighbors=k_neighbors, random_state=0)

def nearest_neighbors(trial: Trial) -> NearestNeighbors:
    """
    Wrapper function for NearestNeighbors transformer which enables Optuna to try out and select the most optimal imputer for the pipeline.
    
    Args:
        trial (Trial): Optuna trial

    Returns:
        NearestNeighbors(Transformer): imputer
    """
    strategy = trial.suggest_categorical(
        "num_impute", ["mean", "median", "most_frequent"]
    )
    return NearestNeighbors(strategy=strategy)


## Estimators

def balanced_rf(trial: Trial) -> BalancedRandomForestClassifier:
    """
    BalancedRandomForestClassifier wrapper for Optuna to select optimal model parameters for pipeline.

    Args:
        trial (Trial): Optuna Trial

    Returns:
        BalancedRandomForestClassifier: Classifier
    """
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 500),
        "criterion": trial.suggest_categorical("criterion", ["gini", "entropy"]),
        "max_depth": trial.suggest_int("max_depth", 5, 400),
    }

    model = BalancedRandomForestClassifier(
        **params,
        replacement=True,
        bootstrap=True,
        sampling_strategy="all",
        class_weight= 'balanced',
        random_state=0,
        n_jobs=-1,
    )

    return model


def xgb_clf(trial: Trial) -> XGBClassifier:
    """
    XGBClassifier wrapper for Optuna to select optimal model parameters for pipeline.

    Args:
        trial (Trial): Optuna Trial

    Returns:
        XGBClassifier: Classifier
    """
    params = {
        "max_depth": trial.suggest_int("max_depth", 1, 20),
        "gamma": trial.suggest_loguniform("gamma", 0.01, 1),
        "eta": trial.suggest_loguniform("eta", 0.01, 0.04),
        "reg_lambda": trial.suggest_loguniform("reg_lambda", 0.001, 10),
        "subsample": trial.suggest_float("subsample", 0.1, 1),
        "n_estimators": trial.suggest_int("n_estimators", 10, 300),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.1, 1),
        "reg_alpha": trial.suggest_loguniform("reg_alpha", 0.001, 10),
    }

    model = XGBClassifier(
        **params,
        seed=0,
        objective="binary:logistic",
        n_jobs = -1
    )

    return model


def balanced_xgb_clf(trial: Trial, scale_pos_weight: float = 10.89) -> XGBClassifier:
    """
    Class balanced XGBClassifier wrapper for Optuna to select optimal model parameters for pipeline. User can pass in scale_pos_weight to scale the classes.
    

    Args:
        trial (Trial): Optuna Trial
        scale_pos_weight (float, optional): scale class weight of positive class. Defaults to 10.89 for current project.

    Returns:
        XGBClassifier: Classifier
    """

    params = {
        "max_depth": trial.suggest_int("max_depth", 1, 10),
        "gamma": trial.suggest_loguniform("gamma", 0.01, 1),
        "eta": trial.suggest_loguniform("eta", 0.01, 0.04),
        "reg_lambda": trial.suggest_loguniform("reg_lambda", 0.001, 10),
        "subsample": trial.suggest_float("subsample", 0.1, 1),
        "n_estimators": trial.suggest_int("n_estimators", 10, 300),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.1, 1),
        "reg_alpha": trial.suggest_loguniform("reg_alpha", 0.001, 10),
    }

    model = XGBClassifier(
        **params,
        scale_pos_weight=scale_pos_weight,
        seed=0,
        objective="binary:logistic",
        n_jobs = -1
    )

    return model


def lgbm_clf(trial: Trial) -> LGBMClassifier:
    """
    LGBMClassifier wrapper for Optuna to select optimal model parameters for pipeline.

    Args:
        trial (Trial): Optuna Trial


    Returns:
        LGBMClassifier: Classifier
    """

    params = {
        "max_depth": trial.suggest_int("max_depth", 1, 100),
        "lambda_l1": trial.suggest_loguniform("lambda_l1", 0.01, 1),
        "learning_rate": trial.suggest_loguniform("learning_rate", 0.01, 0.1),
        'num_leaves': trial.suggest_int("max_leaves", 20, 50),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 10, 30),
        'bagging_fraction': trial.suggest_loguniform("bagging_fraction", 0.01, 1),
        'feature_fraction': trial.suggest_loguniform("feature_fraction", 0.01, 1),
    }

    model = LGBMClassifier(
        **params,
        seed=0,
        objective="binary",
        n_jobs = -1,
        force_col_wise=True,
        verbose=-1

    )

    return model




def balanced_lgbm_clf(trial: Trial) -> LGBMClassifier:
    """
    Class balanced LGBMClassifier wrapper for Optuna to select optimal model parameters for pipeline.

    Args:
        trial (Trial): Optuna Trial


    Returns:
        LGBMClassifier: Classifier
    """

    params = {
        "max_depth": trial.suggest_int("max_depth", 1, 100),
        "lambda_l1": trial.suggest_loguniform("lambda_l1", 0.01, 1),
        "learning_rate": trial.suggest_loguniform("learning_rate", 0.01, 0.1),
        'num_leaves': trial.suggest_int("max_leaves", 20, 50),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 10, 30),
        'bagging_fraction': trial.suggest_loguniform("bagging_fraction", 0.01, 1),
        'feature_fraction': trial.suggest_loguniform("feature_fraction", 0.01, 1),
    }

    model = LGBMClassifier(
        **params,
        is_unbalance=True,
        seed=0,
        objective="binary",
        n_jobs = -1,
        force_col_wise=True,
        verbose=-1
    )

    return model

def balanced_svc_rbf(trial: Trial) -> SVC:
    """
    SVC rbf kernel wrapper for Optuna to select optimal model parameters for pipeline.

    Args:
        trial (Trial): Optuna Trial


    Returns:
        SVC: Classifier
    """

    params = {
        "C": trial.suggest_loguniform("C", 0.01, 1),
        "gamma": trial.suggest_loguniform("gamma", 0.01, 1),
    }

    model = SVC(
        **params,
        kernel = 'rbf',
        class_weight='balanced'
        
    )

    return model

def balanced_svc_linear(trial: Trial) -> SVC:
    """
    SVC linear kernel wrapper for Optuna to select optimal model parameters for pipeline.

    Args:
        trial (Trial): Optuna Trial


    Returns:
        SVC: Classifier
    """

    params = {
        "C": trial.suggest_loguniform("C", 0.01, 1),
        "gamma": trial.suggest_loguniform("gamma", 0.01, 1),
    }

    model = SVC(
        **params,
        kernel = 'linear',
        class_weight='balanced'
        
    )

    return model


def logit(trial: Trial) -> LogisticRegression:
    """
    LogisticRegression wrapper for Optuna to select optimal model parameters for pipeline.

    Args:
        trial (Trial): Optuna Trial


    Returns:
        LogisticRegression: Classifier
    """

    params = {
        "C": trial.suggest_loguniform("C", 0.01, 1),
        'solver': trial.suggest_categorical('solver', ['lbfgs', 'liblinear', 'sag', 'newton-cg', 'newton-cholesky', 'saga']),
    }

    model = LogisticRegression(
        **params,
        random_state = 0,
        n_jobs = -1
        
    )

    return model

def balanced_logit(trial: Trial) -> LogisticRegression:
    """
    LogisticRegression wrapper for Optuna to select optimal model parameters for pipeline.

    Args:
        trial (Trial): Optuna Trial


    Returns:
        LogisticRegression: Classifier
    """

    params = {
        "C": trial.suggest_loguniform("C", 0.01, 1),
        'solver': trial.suggest_categorical('solver', ['lbfgs', 'liblinear', 'sag', 'newton-cg', 'newton-cholesky', 'saga']),
    }

    model = LogisticRegression(
        **params,
        class_weight= 'balanced',
        random_state = 0,
        n_jobs = -1
        
    )

    return model
