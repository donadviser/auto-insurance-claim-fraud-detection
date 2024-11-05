from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import PowerTransformer
from datetime import datetime
import numpy as np
import pandas as pd

class MeanImputer(BaseEstimator, TransformerMixin):
    """Impute missing values with the mean of each specified column."""
    
    def __init__(self, columns=None):
        self.columns = columns
        self.mean_values = {}

    def fit(self, X, y=None):
        for col in self.columns:
            self.mean_values[col] = X[col].mean()
        return self
    
    def transform(self, X):
        X_copy = X.copy()
        for col, mean_val in self.mean_values.items():
            X_copy[col] = X_copy[col].fillna(mean_val)
        return X_copy


class ModeImputer(BaseEstimator, TransformerMixin):
    """Impute missing values with the mode of each specified column."""
    
    def __init__(self, columns=None):
        self.columns = columns
        self.mode_values = {}

    def fit(self, X, y=None):
        for col in self.columns:
            self.mode_values[col] = X[col].mode()[0]
        return self
    
    def transform(self, X):
        X_copy = X.copy()
        for col, mode_val in self.mode_values.items():
            X_copy[col] = X_copy[col].fillna(mode_val)
        return X_copy
    
class ConstantImputer(BaseEstimator, TransformerMixin):
    """Impute missing values in categorical columns with a constant."""
    
    def __init__(self, columns=None, constant_value="missing"):
        self.columns = columns
        self.constant_value = constant_value

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_copy = X.copy()
        for col in self.columns:
            X_copy[col] = X_copy[col].fillna(self.constant_value)
        return X_copy


class MedianImputer(BaseEstimator, TransformerMixin):
    """Impute missing values in numerical columns with the median."""
    
    def __init__(self, columns=None):
        self.columns = columns
        self.median_values = {}

    def fit(self, X, y=None):
        for col in self.columns:
            self.median_values[col] = X[col].median()
        return self
    
    def transform(self, X):
        X_copy = X.copy()
        for col, median_val in self.median_values.items():
            X_copy[col] = X_copy[col].fillna(median_val)
        return X_copy


class DropColumns(BaseEstimator, TransformerMixin):
    """Drop specified columns from the dataset."""
    
    def __init__(self, columns_to_drop=None):
        self.columns_to_drop = columns_to_drop

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X.drop(columns=self.columns_to_drop, errors="ignore")


class DomainFeatureCreator(BaseEstimator, TransformerMixin):
    """
    Transformer to engineer new features from the dataset.
    """
    def __init__(self, bins_hour: list, names_period: list):
        self.bins_hour = bins_hour
        self.names_period = names_period
        self.current_year = datetime.now().year
        
    def fit(self, X: pd.DataFrame, y: None = None):
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        x_copy = X.copy()
        
        X_updated = (x_copy
                        # Calculate new features
                        .assign(
                            vehicle_age=self.current_year - x_copy['auto_year'],  # Calculate vehicle age
                            incident_period_of_day=pd.cut(x_copy['incident_hour_of_the_day'], bins=self.bins_hour, labels=self.names_period)  # Create period of day feature
                            )
                        )
        return X_updated
        


class CustomLabelEncoder(BaseEstimator, TransformerMixin):
    """Encode categorical features as ordinal values based on frequency."""
    
    def __init__(self, columns=None):
        self.columns = columns
        self.label_mappings = {}

    def fit(self, X, y=None):
        for col in self.columns:
            sorted_values = X[col].value_counts().sort_values(ascending=True).index
            self.label_mappings[col] = {val: i for i, val in enumerate(sorted_values)}
        return self
    
    def transform(self, X):
        X_copy = X.copy()
        for col, mapping in self.label_mappings.items():
            X_copy[col] = X_copy[col].map(mapping)
        return X_copy


class LogTransformer(BaseEstimator, TransformerMixin):
    """Apply logarithmic transformation to specified columns."""
    
    def __init__(self, columns=None):
        self.columns = columns

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_copy = X.copy()
        for col in self.columns:
            # Shift negative values to avoid NaN; add a constant (1 + abs(min_value))
            min_value = np.min(X_copy[col])
            shift_value = 1 - min_value if min_value < 0 else 1
            
            # Apply the shift and log transformation
            X_copy[col] = np.log1p(X_copy[col] + shift_value)   
        return X_copy
    

class PowerTransformerWrapper(BaseEstimator, TransformerMixin):
    """Wrapper for PowerTransformer from sklearn.

     Attributes:
        transformer (PowerTransformer): The PowerTransformer instance used for transformation.
        transformed_columns (pandas.DataFrame): The DataFrame containing the transformed
            numerical columns.
        - Supporting pandas DataFrames directly (no need for manual column selection).
        - Transforming all numerical columns by default (configurable via `columns`).
        - Employing efficient vectorized operations for performance in production.
        - Returning a copy of the input DataFrame (avoids potential in-place modification).
        - Setting `copy=True` in `PowerTransformer` for production safety.
    """

    def __init__(self, columns=None, copy=True):
        """
        Parameters:
            columns (list-like of str, optional): List of column names to transform.
                If None, all numerical columns are transformed. Defaults to None.
            copy (bool, optional): Whether to create a copy of the input DataFrame during
                transformation. Defaults to True (recommended for production).
        """
        self.columns = columns
        self.transformer = PowerTransformer(copy=copy)  # Set copy=True for production

    def fit(self, X, y=None):
        """
        Fits the PowerTransformer to the numerical columns in X.

        Args:
            X (pandas.DataFrame): The input DataFrame.
            y (object, optional): Ignored.

        Returns:
            self (PowerTransformerWrapper): The fitted instance.
        """

        if not isinstance(X, pd.DataFrame):
            raise TypeError("During PowerTransformer, Input X must be a pandas.DataFrame")

        self.transformer.fit(X[self.columns])
        return self

    def transform(self, X, y=None):
        """
        Applies the fitted PowerTransformer to the numerical columns in X.

        Args:
            X (pandas.DataFrame): The input DataFrame.
            y (object, optional): Ignored.

        Returns:
            pandas.DataFrame: A new DataFrame with the transformed columns.
        """

        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input X must be a pandas.DataFrame")

        X_copy = X.copy()   
        X_copy[self.columns] = self.transformer.transform(X[self.columns])
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