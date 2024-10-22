import numpy as np
import pandas as pd
from typing import Union
from datetime import datetime
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import PowerTransformer


class LogTransformer(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def fit(self, X, y=None):
        # No fitting necessary for log transformation
        return self

    def transform(self, X):
        """Apply log transformation to the input data (X).
        
        Args:
            X (DataFrame or Array): Input data for transformation.

        Returns:
            Transformed data with log applied to all values.
        """
        X = X.copy()  # to avoid changing the original data
        return np.log(X)  # apply log to all columns directly
    
    
# Try out Log Transformation
class LogTransforms(BaseEstimator,TransformerMixin):
    def __init__(self,variables=None):
        self.variables = variables
    
    def fit(self,X,y=None) -> 'LogTransforms':
        return self
    
    def transform(self,X) ->pd.DataFrame:
        X = X.copy()
        for col in self.variables:
            X[col] = np.log(X[col])
        return X
    
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
            return X.drop(columns=self.redundant_cols)
        except Exception as e:
            print(e)
            #raise CustomException(e, f"Error in DropRedundantColumns")
            

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
            print(e)
            #raise CustomException(e, f"Error in CreateNewFeature")
            

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
            print(e)
            #raise CustomException(e, f"Error in AlterCurrentFeature")
            


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
            #columns_to_transform = X_copy.select_dtypes(include=['object']).columns
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

