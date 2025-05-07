import pandas as pd
import numpy as np
from typing import Callable, List, Optional, Union
from pandas.core.generic import freq_to_period_freqstr
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.preprocessing import FunctionTransformer


class FlagNullTransformer(TransformerMixin, BaseEstimator):
    """
    A custom transformer to add binary indicator columns for null values in specified columns, it is also possible to use should_acc to accumulate the nulls.

    Args:
        col_names : list of str
            The names of the columns to check for null values.
    """

    def __init__(
        self,
        col_names: List[str] = [],
        should_acc: bool | None = False,
        acc_col_name: str | None = "acc_col",
    ):
        """
        Initialize the transformer with column names.

        Args:
            col_names : list of str
                The names of the columns to check for null values.
        """
        self.col_names = col_names
        self.should_acc = should_acc
        self.acc_col_name = acc_col_name

    def fit(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> "FlagNullTransformer":
        """
        Fit the transformer. This transformer does not learn anything from the data.

        """
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the input data by adding binary indicator columns for null values.

        Args:
            X : pd.DataFrame
                The input data.

        Returns:
            X_transformed : pd.DataFrame
                The transformed data with added null indicator columns.
        """
        X = X.copy()  # To avoid changing the original DataFrame
        label_col = []
        for i in self.col_names:
            col_has_nulls = X[i].isnull().sum() != 0
            if col_has_nulls:
                X[f"{i}_isnull"] = X[i].isnull()
                label_col.append(f"{i}_isnull")

        if self.should_acc:
            X[self.acc_col_name] = X[label_col].sum(axis=1)

            X = X.drop(columns=label_col)

        return X

    def get_feature_names_out(self, input_features) -> list[str | None] | list[str]:
        """
        Get output feature names for transformation.

        Parameters
        ----------
        input_features : list of str or ndarray of str, optional
            Input feature names. If None, it defaults to the column names from the input DataFrame.

        Returns
        -------
        feature_names : list of str
            The list of feature names after transformation.
        """
        if input_features is None:
            input_features = self.col_names
        new_feature_names = [
            f"{col}_isnull" for col in self.col_names if col in input_features
        ]
        if self.should_acc:
            new_feature_names = [self.acc_col_name]

        return list(input_features) + new_feature_names



def encode_cat_to_bools_transformer(
    X: pd.DataFrame,
    target_col_names: list[str],
    pos_val: str = "Yes",
    neg_val: str = "No",
    should_rename_col: bool = False,
    new_col_name: str = 'new_col'

) -> pd.DataFrame:
    """
    Transforms specified categorical columns in a DataFrame to boolean values.

    This function converts the values in the specified columns to boolean based on the given positive and negative values.
    Any values that do not match the positive or negative values will be set to NaN. Optionally, the transformed column can be 
    renamed to a new column name, with the original column dropped from the DataFrame.

    Args:
        X (pd.DataFrame): The input DataFrame containing the data to be transformed.
        
        target_col_names (List[str]): The names of the columns to be transformed.
        
        pos_val (str, optional): The value to be considered as True. Default is "Yes".
        
        neg_val (str, optional): The value to be considered as False. Default is "No".
        
        should_rename_col (bool, optional): Whether to rename the transformed column to a new name. Default is False.
        
        new_col_name (str, optional): The new column name to use if should_rename_col is True. Default is 'new_col'.


    Returns:
        pd.DataFrame: A DataFrame with the specified columns transformed to boolean values.

    """
    X = X.copy()
    
    if should_rename_col:
        col_to_rename = target_col_names[0]
        encoded_col = X[col_to_rename].apply(
            lambda i: True if i == pos_val else False if i == neg_val else np.nan
        ).astype(pd.BooleanDtype())
        X[new_col_name] = encoded_col
        X = X.drop(columns = [col_to_rename])
        
    else:
        for i in target_col_names:
            encoded_col = X[i].apply(
                lambda i: True if i == pos_val else False if i == neg_val else np.nan
            ).astype(pd.BooleanDtype())

            X[i] = encoded_col

    return X

