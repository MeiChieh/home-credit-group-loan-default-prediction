# import sklearnex
# sklearnex.unpatch_sklearn()
import numpy as np
import pandas as pd
import modin.pandas as md
import re
import matplotlib.pyplot as plt
from IPython.display import display as dp
from termcolor import cprint
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    multilabel_confusion_matrix
)
from sklearn.model_selection import cross_validate
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.base import RegressorMixin
import shap
from typing import List, Callable, Any, Dict, Union
import math
import matplotlib.axes._axes as axes
import warnings
from sklearn.pipeline import Pipeline
import optuna
from optuna import Trial, create_study
import joblib



# Filter out the specific warning by category
warnings.filterwarnings("ignore", category=UserWarning)

# data wrangling related


def convert_large_csv_file_to_dataframe(file_path: str) -> pd.DataFrame:
    """
    Load large file using memmap, convert it to suitable datatype, and store in dataframe

    Args:
        file_path (str): file path

    Returns:
        pd.DataFrame: file stored in dataframe

    """
    large_df = np.memmap(file_path, mode="r")
    large_df = pd.read_csv(file_path)
    large_df_data_types = large_df.convert_dtypes().dtypes
    large_df_data_types = large_df_data_types.to_dict()
    large_df = pd.read_csv(file_path, dtype=large_df_data_types)
    return large_df


def to_slug(input_str: str) -> str:
    """
    Convert string with space, underscore or hyphen as deliminator to slug case

    Args:
        input_str (str): target string to be converted to slug case

    Returns:
        str: sluggified_string
    """
    sluggified_string = "_".join(re.split(r"[_ -]", input_str)).lower()
    return sluggified_string


def input_to_df(
    input: pd.Series | dict, 
    is_dict: bool = False, 
    col_name: str = "col"
) -> pd.DataFrame:
    """
    Convert pd.Series to pd.DataFrame

    Args:
        input (pd.Series | dict): target series or dict to convert
        is_dict(dict): if input is a dict instead of pd.Series type
        col_name (str, optional): column name for converted column. Defaults to 'col'.

    Returns:
        pd.DataFrame: df with single target series column
    """
    if is_dict == True:
        return pd.DataFrame(pd.Series(input), columns=[col_name])
    else:
        return pd.DataFrame(pd.Series(input.to_dict()), columns=[col_name])


# plotting related
def fig_size(w: int = 3, h: int = 3) -> None:
    """
    Lazy function for setting figure size

    Args:
        w (int, optional): set fig width. Defaults to 3.
        h (int, optional): set fig length. Defaults to 3.
    """
    plt.figure(figsize=(w, h))


def bprint(input: str) -> None:
    """
    Style printing with color

    Args:
        input (any): content to print
    """
    cprint(f"\n{input}", "green", attrs=["bold"])


def mark_bar(plot: axes.Axes) -> None:
    """
    Mark bar values on the histplot or barplot with rounded values

    Args:
        plot (matplotlib axis): plot
    """
    for i in plot.containers:
        plot.bar_label(i, fmt="%.2f")


def mark_percent(
    ax: axes.Axes,
    col: pd.Series,
    hue: pd.Series,
    target_class: str | int,
    hide_total: List[str] = [],
    hide_target: List[str] = [],
) -> None:
    """
    Mark percentage for stacked histplot or barplot for categorical features. When the total and target percentage overlaps due to similar values, hide_total or hide_target can be provided to hide the values.

    Args:
        ax: axes of plot
        col (pd.Series): dataframe column for x-axis
        hue (pd.Series): dataframe column for hue
        target_class (str): target class in hue column
    """
    tab = pd.crosstab(col, hue)
    tab_index = tab.index.tolist()
    tab_norm = pd.crosstab(col, hue, normalize="index").round(3) * 100
    total_val = tab.sum().sum()
    col_total = tab.sum(axis=1)
    # column percentage
    col_percent = (col_total * 100 / total_val).round(2).tolist()
    # percentage of target class in column
    col_target_class_percent = tab_norm[target_class].tolist()
    # add percentages to df
    percent_df = pd.DataFrame([col_percent], columns=tab_index, index=["col_percent"]).T
    percent_df["col_target_class_percent"] = col_target_class_percent
    # add value count to df
    percent_df["col_count"] = tab.sum(axis=1).tolist()
    percent_df["col_target_class_count"] = tab[target_class]

    ticks = [i.get_text() for i in ax.get_xticklabels()]

    # append percentages to plot
    for i in ticks:
        # total percentage
        if i not in hide_total:
            ax.text(
                i,
                percent_df.col_count.loc[i],
                "%0.1f" % percent_df.col_percent.loc[i] + "%",
                ha="center",
            )
        # target class percentage
        if i not in hide_target:
            ax.text(
                i,
                percent_df.col_target_class_count.loc[i],
                "%0.1f" % percent_df.col_target_class_percent.loc[i] + "%",
                ha="center",
            )


def mark_df_color(col: pd.Series, id, color="rosybrown") -> List[str]:
    """
    Mark specified column_id or row_id with color on dataframe

    Args:
        col: pandas series passed in from apply method
        id: index of the column or row
        color: color for marking, default to rosybrown

    Returns:
        List[str]: list of background color styles for each cell in the column or row
    """

    def mark():
        return [
            (f"background-color: {color}" if idx == id else "background-color: ")
            for idx, _ in enumerate(col)
        ]

    return mark()


# modeling related


def plot_confusion(y_pred: np.ndarray, y_train: np.ndarray, title: str) -> None:
    """
    Plot confusion matrix

    Args:
        y_pred (np.ndarray): model prediction
        y_train (np.ndarray): y_train
        title (str): confusion matrix plot title
    """
    cf = confusion_matrix(y_train, y_pred, labels=[0, 1])
    sns.heatmap(cf, annot=True, fmt=".0f", cbar=False, cmap='coolwarm')
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(title)


def plot_tree_shap_feature_importance(
    model: RegressorMixin, model_name: str, X_train: pd.DataFrame
) -> None:
    """
    Plot shap feature importance for model

    Args:
        model: tree models
        model_name (str)
    """

    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_train)

    fig = plt.figure()
    shap.plots.bar(shap_values, show=False)
    plt.gcf().set_size_inches(5, 3)
    plt.title(f"{model_name} Shap Feature Importance")
    plt.show()

def cv_score_summary(
    model: Pipeline,
    x: pd.DataFrame,
    y: np.ndarray,
    cv: int | StratifiedKFold = 5,
) -> None:
    """
    Print out cv scores, using average_precision as scorer

    Args:
        model (Pipeline): _description_
        x (pd.DataFrame): _description_
        y (np.ndarray): _description_
        cv (int | StratifiedKFold, optional): _description_. Defaults to 5.
    """
    cv_score = cross_val_score(model, x, y, n_jobs=-1, cv=cv, scoring="roc_auc")
    print(f"cv: {cv_score}")
    print(f"cv_mean: {round(cv_score.mean(), 4)}")
    print(f"cv_std: {round(cv_score.std(), 6)}")
    
def get_cv_score(
    model: Pipeline,
    x: pd.DataFrame,
    y: np.ndarray,
    cv: int | StratifiedKFold = 5,
):
    """
    Print out cv scores, using average_precision as scorer

    Args:
        model (Pipeline): _description_
        x (pd.DataFrame): _description_
        y (np.ndarray): _description_
        cv (int | StratifiedKFold, optional): _description_. Defaults to 5.
    """
    cv_score = cross_val_score(model, x, y, n_jobs=-1, cv=cv, scoring="roc_auc")

    return {
        "mean": round(cv_score.mean(), 4),
        "std": round(cv_score.std(), 6),
    }
    
    
def multilabel_cf(
    y_true: np.ndarray, y_pred: np.ndarray, plt_title: str = "multilabel_cf"
) -> None:
    """
    Plot multilabel confusion matrixes in a subplot

    Args:
        y_true (np.ndarray): y true label
        y_pred (np.ndarray): model prediction
        plt_title (str, optional): name for the subplot suptitle. Defaults to "multilabel_cf".
    """
    cfm = multilabel_confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3, 4, 5, 6])
    class_name_ls = ["A", "B", "C", "D", "E", "F", "G"]
    fig_size(18, 2)

    for i in range(7):
        plot_i = i + 1
        plt.subplot(1, 7, plot_i)
        sns.heatmap(cfm[i].T, annot=True, cbar=False, fmt=".0f")
        plt.title(class_name_ls[i])
        plt.xlabel("predicted class")
        if i == 0:
            plt.ylabel("true class")

    plt.suptitle(plt_title, y=1.2)
    plt.show()




def run_study(
    objective_function: Callable, trials_count: int = 5, name: str = "model_study"
) -> List[Trial]:
    """
    Wrapper function for running an optuna study and returns the study

    Args:
        objective_function (Callable): objective_function to optimize
        trials_count (int, optional): trials to run. Defaults to 5.
        name (str, optional): study name. Defaults to "model_study".

    Returns:
        list: list of trials
    """

    study = optuna.create_study(
        direction="maximize",
        study_name=name,
    )

    study.optimize(
        objective_function,
        n_trials=trials_count,
        n_jobs=-1,
    )

    return study

def cast_typ_obj_to_cat(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cast dtype object to category columns in dataframe.

    Args:
        df (pd.DataFrame): target dataframe

    Returns:
        pd.DataFrame: category-casted dataframe
    """

    df_cols = df.columns
    for i in df_cols:
        if df[i].dtype == "object":
            df[i] = df[i].astype("category")
        else:
            continue
    return df


def down_cast_float(df: pd.DataFrame) -> pd.DataFrame:
    """
    Down-cast dtype float64 to float32 columns in dataframe.

    Args:
        df (pd.DataFrame): target dataframe

    Returns:
        pd.DataFrame: down-casted dataframe
    """
    df_cols = df.columns
    for i in df_cols:
        if df[i].dtype == "float64":
            df[i] = df[i].astype("float32")
        else:
            continue

    return df

def lower_case_col_name(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cast dataframe columns from upper to lower case.

    Args:
        df (pd.DataFrame): target dataframe

    Returns:
        pd.DataFrame: converted dataframe
    """

    new_cols_name = pd.Series(
        df.columns).apply(lambda i: i.lower()).tolist()

    old_cols_name = df.columns.tolist()
    ls_ln = len(new_cols_name)

    col_dict = {old_cols_name[i]: new_cols_name[i] for i in range(ls_ln)}
    
    df = df.rename(columns=col_dict)

    return df

def format_and_squeeze_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Format and squeeze a DataFrame by downcasting float columns and converting column names to lowercase.

    This function applies two transformations to the input DataFrame: it downcasts any float columns to reduce memory usage, and it converts all column names to lowercase.

    Args:
        df (pd.DataFrame): The input DataFrame to be formatted and squeezed.

    Returns:
        pd.DataFrame: The formatted DataFrame with downcasted float columns and lowercase column names.
    """
    df = down_cast_float(df)
    df = lower_case_col_name(df)

    return df


def merge_selected_app_and_historical_feats(x_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge selected current application and historical features before going into the data pipeline.

    Args:
        x_df (pd.DataFrame): current application dataframe for train, val, test

    Returns:
        pd.DataFrame: dataframe with selected current application and history application features
    """
    hist_df = joblib.load("data/models/selected_historical_feats_df")
    selected_app_cols = joblib.load("data/models/selected_app_cols")

    filtered_x_df = x_df[selected_app_cols]
    full_df = filtered_x_df.merge(hist_df, on="sk_id_curr", how="left")
    full_df = full_df.drop(columns=["sk_id_curr"])
    full_df = cast_typ_obj_to_cat(full_df)
    return full_df


def get_null_id(col_name: str, dataframe: pd.DataFrame) -> List[int]:
    """
    Get the indices of rows where the specified column has null values.

    Args:
        col_name (str): The name of the column to check for null values.
        dataframe (pd.DataFrame): The DataFrame to search within.

    Returns:
        List[int]: A list of indices where the specified column has null values.
    """
    return dataframe.loc[dataframe[col_name].isnull()].index.tolist()


def check_null(dataframe: pd.DataFrame) -> pd.Series:
    """
    Check for columns with null values in the DataFrame and return the count of null values for those columns.

    Args:
        dataframe (pd.DataFrame): The DataFrame to check for null values.

    Returns:
        pd.Series: A Series with column names as index and count of null values as values, for columns with null values.
    """
    return dataframe.isnull().sum()[dataframe.isnull().sum() != 0]

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



def tuned_model(model_name: str, pipeline: Callable, X: pd.DataFrame | np.ndarray, y: np.ndarray, cv: StratifiedKFold | int) -> Pipeline:
    """
    Wrapper function for running optuna hyperparameter tuning to optimize a model and output the fitted model.

    Args:
        model_name (str): name of the model
        pipeline (Callable): model pipeline 
        X (pd.DataFrame | np.ndarray): X_train data
        y (np.ndarray): y_train data
        cv (StratifiedKFold | int): cv for model tuning

    Returns:
        Pipeline: Optuna tuned model
    """
    study = optuna.create_study(direction="maximize", study_name=model_name)
    study.optimize(
        lambda trial: objective(
            trial, X, y, pipeline, cv=cv
        ),
        n_trials=150,
        n_jobs=-1,
    )
    best_model = pipeline(study.best_trial).fit(
        X, y
    )
    
    return best_model


def get_classification_result(y_true: List, y_pred: List) -> List:
  """
  Returns the classification result of the model.

  Args:
      y_true (List): y_true
      y_pred (List): y_pred

  Returns:
      List: results (tp/ tn/ fp/ fn)
  """
  
  res_map = {
    (0,0): 'tn',
    (0,1): 'fp',
    (1,0): 'fn',
    (1,1): 'tp'
  }
  comp = list(zip(y_true, y_pred))
  res = [res_map[x] for x in comp]
  return res




def clean_data(obj: Any) -> Any:
    """
    Recursively clean the input data by replacing NaN values with None.
    
    Args:
        obj (Any): The input data which can be of any type (float, dict, list, etc.)

    Returns:
        Any: The cleaned data with NaN values replaced by None.
    """
    if isinstance(obj, float) and math.isnan(obj):
        return None
    if isinstance(obj, dict):
        return {k: clean_data(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [clean_data(i) for i in obj]
    return obj
