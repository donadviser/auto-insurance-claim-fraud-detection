import sys
import numpy as np
import pandas as pd
from typing import Union
from datetime import datetime
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import PowerTransformer
from insurance import logging, CustomException

class OutlierTransformer(BaseEstimator, TransformerMixin):
    """Custom transformer for outlier removal based on various methods."""
    def __init__(self, factor=3.0, form='Z-score', k=1.5, fake=False):
        self.factor = factor
        self.form = form
        self.k = k
        self.fake = fake

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if self.fake:
            return X  # Skip outlier removal
        if self.form == 'Z-score':
            z_scores = np.abs((X - np.mean(X)) / np.std(X))
            return X[z_scores < self.factor]
        elif self.form == 'Turkey':
            lower_bound = np.percentile(X, 25) - self.k * (np.percentile(X, 75) - np.percentile(X, 25))
            upper_bound = np.percentile(X, 75) + self.k * (np.percentile(X, 75) - np.percentile(X, 25))
            return X[(X >= lower_bound) & (X <= upper_bound)]
        elif self.form == 'Modified Z-score':
            med = np.median(X)
            mad = np.median(np.abs(X - med))
            modified_z_scores = 0.6745 * (X - med) / mad
            return X[np.abs(modified_z_scores) < self.factor]
        elif self.form == 'IQR':
            Q1 = np.percentile(X, 25)
            Q3 = np.percentile(X, 75)
            IQR = Q3 - Q1
            return X[(X >= Q1 - 1.5 * IQR) & (X <= Q3 + 1.5 * IQR)]
        else:
            return X  # Default return if no valid method is specified
    def get_feature_names_out(self, input_features=None):
        # Simply return the input feature names since this transformer does not change them
        return input_features if input_features is not None else []


class OutlierHandler(BaseEstimator, TransformerMixin):
    def __init__(self, th: float = 0.99, strategy: str = 'iqr_clip', fill_value: float = None):
        self.quantiles = None
        self.th = th
        self.strategy = strategy
        self.fill_value = fill_value
        self.column_stats = {}

    def fit(self, X: pd.DataFrame, y=None):
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)

        # Compute the quantiles based on the threshold
        self.quantiles = X.quantile([1 - self.th, self.th])

        # Compute column stats for imputation strategies
        if self.strategy in ['iqr_mean', 'iqr_median']:
            if self.strategy == 'iqr_mean':
                self.column_stats = X.mean()
            elif self.strategy == 'iqr_median':
                self.column_stats = X.median()

        return self

    def fit_transform(self, X: pd.DataFrame, y=None):
        self.fit(X)
        return self.transform(X)

    def transform(self, X: pd.DataFrame, y=None):
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)

        for feature in X.columns:
            quantiles = self.quantiles[feature].values
            lower_bound = quantiles[0]
            upper_bound = quantiles[1]

            if self.strategy == 'iqr_clip':
                # Clip outliers to quantile limits
                lower_bound = int(lower_bound) if np.issubdtype(X[feature].dtype, np.integer) else lower_bound
                upper_bound = int(upper_bound) if np.issubdtype(X[feature].dtype, np.integer) else upper_bound
                X[feature] = X[feature].clip(lower=lower_bound, upper=upper_bound)

            elif self.strategy == 'remove':
                # Remove rows containing outliers
                X = X[(X[feature] >= lower_bound) & (X[feature] <= upper_bound)]

            elif self.strategy in ['iqr_mean', 'iqr_median']:
                # Replace outliers with mean or median
                fill_value = self.column_stats[feature]
                if np.issubdtype(X[feature].dtype, np.integer):
                    fill_value = int(fill_value)  # Ensure type compatibility with integer columns
                X.loc[X[feature] < lower_bound, feature] = fill_value
                X.loc[X[feature] > upper_bound, feature] = fill_value

            elif self.strategy == 'constant':
                # Replace outliers with a custom constant
                if self.fill_value is None:
                    raise ValueError("For 'constant' strategy, fill_value must be provided.")
                fill_value = self.fill_value
                if np.issubdtype(X[feature].dtype, np.integer):
                    fill_value = int(fill_value)  # Ensure type compatibility with integer columns
                X.loc[X[feature] < lower_bound, feature] = fill_value
                X.loc[X[feature] > upper_bound, feature] = fill_value

        return X

    def get_feature_names_out(self, input_features=None):
        # Simply return the input feature names since this transformer does not change them
        return input_features if input_features is not None else []

class OutlierDetector(BaseEstimator, TransformerMixin):
    """
    A class to detect and handle outliers in numerical features based on specified strategies.

    Attributes:
        method (str): The method to use for outlier detection ('z_score' or 'iqr').
        threshold (float): The threshold value for detecting outliers.
    """

    def __init__(self, method='iqr'):
        self.method = method

    def fit(self, X, y=None):
        # Fit method to identify outliers based on the chosen method
        if self.method == 'iqr':
            if isinstance(X, np.ndarray):
                X = pd.DataFrame(X)

            self.q1 = X.quantile(0.25)
            self.q3 = X.quantile(0.75)
            self.iqr = self.q3 - self.q1
            self.lower_bound = self.q1 - 1.5 * self.iqr
            self.upper_bound = self.q3 + 1.5 * self.iqr

        elif self.method == 'z-score':
            self.mean = X.mean()
            self.std = X.std()
            self.threshold = 3  # Typical threshold for Z-score
        return self

    def transform(self, X):
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)

        if self.method == 'iqr':
            mask = (X >= self.lower_bound) & (X <= self.upper_bound)
            output = X[mask]
        elif self.method == 'z-score':
            z_scores = (X - self.mean) / self.std
            mask = abs(z_scores) <= self.threshold
            output = X[mask]

        # Handle the case where output might be empty
        if output.shape[0] == 0:
            return np.empty((0, 1))

        # Ensure output is 2D
        output = output.values.reshape(-1, 1)
        return output


    def get_feature_names_out(self, input_features=None):
        # Simply return the input feature names since this transformer does not change them
        return input_features if input_features is not None else []


class LogTransformer(BaseEstimator, TransformerMixin):
    """Custom log transformation that handles negative, zero, and positive values."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Shift negative values to avoid NaN; add a constant (1 + abs(min_value))
        min_value = np.min(X)
        shift_value = 1 - min_value if min_value < 0 else 1

        # Apply the shift and log transformation
        adjusted_X = X + shift_value
        output = np.log1p(adjusted_X)

        return output

    def get_feature_names_out(self, input_features=None):
        # Simply return the input feature names since this transformer does not change them
        return input_features if input_features is not None else []


class PowerTransformerWrapper(BaseEstimator, TransformerMixin):
    """Wrapper for PowerTransformer from sklearn."""
    def __init__(self):
        self.transformer = PowerTransformer()

    def fit(self, X, y=None):
        self.transformer.fit(X)
        return self

    def transform(self, X):
        return self.transformer.transform(X)


# Custom transformers
class DropMissingThreshold(BaseEstimator, TransformerMixin):
    """
    Transformer to drop columns with missing values above a certain threshold.
    """
    def __init__(self, threshold: float=0.5):
        self.threshold = threshold

    def fit(self, X: pd.DataFrame, y: None = None) -> 'DropMissingThreshold':
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        try:
            return X.dropna(thresh=len(X) * self.threshold, axis=1)
        except Exception as e:
            print(e)
            #raise CustomException(e, f"Error in DropMissingThreshold")


class DropRedundantColumns(BaseEstimator, TransformerMixin):
    """
    Transformer to drop redundant columns from the dataset.
    """
    def __init__(self, redundant_cols: list):
        self.redundant_cols = redundant_cols

    def fit(self, X: pd.DataFrame, y: None = None) -> 'DropRedundantColumns':
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        try:
            # Filter redundant columns to include only those present in the DataFrame
            cols_to_drop = [col for col in self.redundant_cols if col in X.columns]

            # Drop the columns and return the modified DataFrame
            return X.drop(columns=cols_to_drop)
        except Exception as e:
            raise CustomException(e, sys)


class CreateNewFeature(BaseEstimator, TransformerMixin):
    """
    Transformer to engineer new features from the dataset.
    """
    def __init__(self, bins_hour: list, names_period: list):
        self.bins_hour = bins_hour
        self.names_period = names_period
        self.current_year = datetime.now().year

    def fit(self, X: pd.DataFrame, y: None = None) -> 'CreateNewFeature':
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        x_copy = X.copy()
        try:
            X_updated = (x_copy
                         # Calculate new features
                         .assign(
                             vehicle_age=self.current_year - x_copy['auto_year'],  # Calculate vehicle age
                             incident_period_of_day=pd.cut(x_copy['incident_hour_of_the_day'], bins=self.bins_hour, labels=self.names_period)  # Create period of day feature
                             )
                         )
            return X_updated
        except Exception as e:
            raise CustomException(e, sys)


class AlterCurrentFeature(BaseEstimator, TransformerMixin):
    """
    Transformer to engineer new features from the dataset.
    """
    def __init__(self, bins_hour: list, names_period: list):
        self.bins_hour = bins_hour
        self.names_period = names_period
        self.current_year = datetime.now().year

    def fit(self, X: pd.DataFrame, y: None = None) -> 'AlterCurrentFeature':
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        x_copy = X.copy()
        try:
            X_updated = (x_copy
                         # Calculate new features
                         .assign(
                             vehicle_age=self.current_year - x_copy['auto_year'],  # Calculate vehicle age
                             incident_period_of_day=pd.cut(x_copy['incident_hour_of_the_day'], bins=self.bins_hour, labels=self.names_period)  # Create period of day feature
                             )
                         )
            return X_updated
        except Exception as e:
            raise CustomException(e, sys)



class ReplaceClassTransformer(BaseEstimator, TransformerMixin):
    """
    A robust custom transformer to replace occurrences of a specified old value
    (e.g., '?') with a new value (e.g., 'Unknown') in selected columns of a DataFrame.

    Parameters
    ----------
    target_value : Union[str, int, float]
        The value to be replaced in the DataFrame (e.g., '?').
    replacement_value : Union[str, int, float], default='Unknown'
        The value that will replace the target_value in the DataFrame.
    columns_to_replace : list or None, default=None
        List of columns to apply the transformation on.
        If None, it will apply to all columns of type object.
    """

    def __init__(self, target_value: Union[str, int, float], replacement_value: Union[str, int, float] = 'Unknown', columns_to_replace: list = None):
        self.target_value = target_value  # The value that needs to be replaced
        self.replacement_value = replacement_value  # The value to replace the target value with
        self.columns_to_replace = columns_to_replace  # Columns where the replacement should be applied

    def fit(self, X: pd.DataFrame, y=None) -> 'ReplaceClassTransformer':
        """
        This transformer does not require fitting, so we return self.

        Parameters
        ----------
        X : pandas.DataFrame
            The input DataFrame to be transformed.

        y : ignored
            This parameter exists for pipeline compatibility.

        Returns
        -------
        self
        """
        # Validate input is a pandas DataFrame
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")

        # If specific columns are provided, check that they exist in the DataFrame
        if self.columns_to_replace is not None:
            missing_columns = set(self.columns_to_replace) - set(X.columns)
            if missing_columns:
                raise ValueError(f"Columns {missing_columns} are not found in the DataFrame")

        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """
        Transforms the DataFrame by replacing occurrences of target_value with replacement_value.

        Parameters
        ----------
        X : pandas.DataFrame
            The input DataFrame to be transformed.

        Returns
        -------
        pandas.DataFrame
            A transformed DataFrame with target_value replaced by replacement_value in the specified columns.
        """
        # Ensure input is a pandas DataFrame
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")

        # Create a copy to avoid mutating the original DataFrame
        X_copy = X.copy()

        # Determine which columns to apply the transformation to
        if self.columns_to_replace:
            columns_to_transform = self.columns_to_replace
        else:
            # Default to object type columns (categorical/string columns)
            columns_to_transform = X_copy.columns

        # Replace the target_value with the replacement_value in the selected columns
        X_copy[columns_to_transform] = X_copy[columns_to_transform].replace(self.target_value, self.replacement_value)

        return X_copy



class ReplaceValueTransformer(BaseEstimator, TransformerMixin):
    """
    Custom transformer to replace occurrences of a specified old value with a new value.
    """

    def __init__(self, old_value, new_value):
        """
        Initialize the transformer with the old and new values.

        Args:
            old_value: The value to be replaced.
            new_value: The new value to replace the old value with.
        """
        self.old_value = old_value
        self.new_value = new_value

    def fit(self, X, y=None):
        """
        Fit the transformer. This step is not necessary for this transformer.

        Args:
            X: The input data.
            y: The target labels (optional).

        Returns:
            self: The fitted transformer.
        """
        return self

    def transform(self, X):
        """
        Transform the input data by replacing the old value with the new value.

        Args:
            X: The input data.

        Returns:
            X_transformed: The transformed data.
        """
        X_transformed = X.copy()  # Create a copy to avoid modifying the original data
        X_transformed[X_transformed == self.old_value] = self.new_value
        return X_transformed

