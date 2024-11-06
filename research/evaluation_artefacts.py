import mlflow
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, roc_auc_score, roc_curve, PrecisionRecallDisplay
import os
import seaborn as sns
import numpy as np

def plot_confusion_matrix(model, X_test: pd.DataFrame, y_test: pd.Series, model_name: str) -> None:
    """Plot confusion matrix and log it as an artifact in MLflow."""
    disp = ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)
    plt.title(f"{model_name} - Confusion Matrix")
    plt.savefig(f"{model_name}_confusion_matrix.png")
    #mlflow.log_artifact(f"{model_name}_confusion_matrix.png")
    plt.close()  # Close the plot to free memory
    #os.remove(f"{model_name}_confusion_matrix.png")  # Clean up

def plot_roc_auc(model, X_test: pd.DataFrame, y_test: pd.Series, model_name: str) -> None:
    """Plot ROC-AUC curve and log it as an artifact in MLflow."""
    RocCurveDisplay.from_estimator(model, X_test, y_test)
    plt.title(f"{model_name} - ROC-AUC Curve")
    plt.savefig(f"{model_name}_roc_auc.png")
    #mlflow.log_artifact(f"{model_name}_roc_auc.png")
    plt.close()
    #os.remove(f"{model_name}_roc_auc.png")

def plot_feature_importance(model, feature_names: list, model_name: str) -> None:
    """Plot feature importance for tree-based models and log it as an artifact in MLflow."""
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        sorted_indices = importances.argsort()
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(sorted_indices)), importances[sorted_indices], align='center')
        plt.yticks(range(len(sorted_indices)), [feature_names[i] for i in sorted_indices])
        plt.title(f"{model_name} - Feature Importance")
        plt.xlabel("Importance")
        plt.ylabel("Feature")
        plt.savefig(f"{model_name}_feature_importance.png")
        #mlflow.log_artifact(f"{model_name}_feature_importance.png")
        plt.close()
        #os.remove(f"{model_name}_feature_importance.png")
        
def plot_feature_importance_main(model, feature_names=None, model_name='classifier', top_n=20):
    """
    Plot the feature importance of a fitted tree-based model.

    Parameters:
    -----------
    model : estimator
        A fitted tree-based model with `feature_importances_` attribute (e.g., RandomForest, GradientBoosting).
    
    feature_names : list or None
        List of feature names. If None, numerical indices are used as feature names.
    
    top_n : int, default=20
        The number of top features to plot. If `None`, all features will be plotted.
    """
    # Check if the model has feature_importances_ attribute
    try:
        if not hasattr(model, 'feature_importances_'):
            importance_values = model.coef_[0]
        else:
            # Get feature importance values and sort them in descending order
            importance_values = model.feature_importances_
    except Exception as e:
        raise ValueError(f"The model does not have `feature_importances_` attribute.: {e}")

    
    
    # Create a DataFrame for better manipulation and sorting
    if feature_names is None:
        feature_names = [f'Feature {i}' for i in range(len(importance_values))]
    
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importance_values})
    
    # Sort features by importance
    importance_df = importance_df.sort_values(by='Importance', ascending=False)
    
    # If top_n is specified, select the top_n features
    if top_n is not None and top_n < len(importance_df):
        importance_df = importance_df.head(top_n)
    
    # Plot the feature importance
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=importance_df, hue='Feature', palette='viridis', dodge=False, legend=False)
    plt.title(f'{model_name} - Top Feature Importance')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
    #plt.show()
    plt.savefig(f"{model_name}_feature_importance.png")
    plt.close()
    
def plot_feature_importance_best(model, feature_names, model_name="classifier", n_top=10):
    """
    This function takes in a dictionary of models, the dataset X, y, and the feature names.
    It fits each model, extracts feature importances (if available), 
    and plots the top n features.

    Parameters:
    models (dict): A dictionary containing model names and their respective model objects.
    X (np.ndarray): Feature dataset.
    y (pd.Series): Target variable.
    feature_names (list): List of feature names after transformations.
    n_top (int): Number of top features to display. Default is 10.
    
    Returns:
    None
    """
     
    print(f"Feature ranking for model: {model_name}")
    try:
        # Check if the model has the attribute `feature_importances_`
        if hasattr(model, 'feature_importances_'):
            # Get feature importances
            importance_scores = model.feature_importances_
            
        else:                 
            print(f"{model_name} does not support feature importances.")
            importance_scores = model.coef_[0]
            
        # Create a DataFrame from feature names and importances
        data = {'Feature': feature_names, 'Score': importance_scores}
        df = pd.DataFrame(data)
        
        
        # Take the absolute value of the score
        df['Abs_Score'] = np.abs(df['Score'])
        
        df_sorted = df.sort_values(by="Abs_Score", ascending=False)
        if n_top:
            # Sort by absolute value of score in descending order (top 10)
            df_sorted = df_sorted.head(n_top)
        
        # Define a color palette based on score values (positive = green, negative = red)
        colors = ["green" if score > 0 else "red" for score in df_sorted["Score"]]
        plt.figure(figsize=(12, 8))
        # Create the bar chart with Seaborn
        sns.barplot(x="Feature", y="Score", hue="Feature", legend=False, data=df_sorted, palette=colors)
        
        # Customize the plot for better visual appeal
        plt.xlabel("Feature")
        plt.ylabel("Feature Importance Score")
        plt.title(f"{model_name} - Feature Importance")
        plt.xticks(rotation=45, ha="right")  # Rotate x-axis labels for better readability
        plt.tight_layout()  # Adjust spacing between elements
    
        # Display the plot
        #plt.show()
        plt.savefig(f"{model_name}_feature_importance.png")
        plt.close()          
        
            
    except Exception as e:
        print(f"Error occurred while extracting feature importances: {str(e)}")
        
        
def plot_roc_auc_curve_seaborn(model, X_test: pd.DataFrame, y_test: pd.Series, model_name: str='classifier', y_pred_proba=None):
  """
  Plots the ROC curve and calculates the AUC score for a given model using Seaborn.

  Args:
      model: The trained model to evaluate.
      X_test: The test data features.
      y_test: The test data labels.
      title: Title for the plot (default: "ROC Curve").
  """
  figsize=(8, 6)

  # Calculate ROC curve and AUC score
  if y_pred_proba is None:
      y_pred_proba = model.predict_proba(X_test)[:, 1]

  fpr, tpr, thr = roc_curve(y_test, y_pred_proba)
  auc = roc_auc_score(y_test, y_pred_proba)

  # Create the plot
  plt.figure(figsize=figsize)

  # Use Seaborn for ROC curve
  sns.lineplot(x=fpr, y=tpr, label=f'ROC curve (AUC = {auc:.3f})', color='violet', linewidth=2)

  # Plot baseline performance line
  plt.plot([0, 1], [0, 1], linestyle='--', label='Baseline performance', color='gray')

  # Set axis labels and title
  plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=14)
  plt.ylabel('True Positive Rate (Sensitivity)', fontsize=14)
   
  # Add legend
  plt.legend(loc=4)
  
  plt.title(f"{model_name} - ROC-AUC Curve", fontsize=16)
  plt.savefig(f"{model_name}_roc_auc.png")

  # Display the plot
  #plt.show()
  plt.close() 
  #return {"auc": auc}
  
  # Function to log Precision-Recall Curve
def log_precision_recall_curve(model, X_test, y_test, model_name="model"):
    PrecisionRecallDisplay.from_estimator(model, X_test, y_test)
    plt.title("Precision-Recall Curve")
    plt.savefig(f"{model_name}_precision_recall_curve.png")
    #mlflow.log_artifact(f"{model_name}_precision_recall_curve.png")
    plt.close()
    

def log_cumulative_gains_chart(model, X_test, y_test, model_name="model"):
    # Predict probabilities
    y_proba = model.predict_proba(X_test)[:, 1]
    # Sort by probability
    sorted_indices = np.argsort(y_proba)[::-1]
    sorted_y_test = np.array(y_test)[sorted_indices]
    
    # Cumulative gain
    cumulative_gains = np.cumsum(sorted_y_test) / sum(y_test)
    gains_x = np.arange(1, len(y_test) + 1) / len(y_test)
    
    # Plotting
    plt.figure(figsize=(8, 6))
    plt.plot(gains_x, cumulative_gains, label="Cumulative Gain", color="b")
    plt.plot([0, 1], [0, 1], label="Baseline", linestyle="--", color="r")
    plt.xlabel("Percentage of Samples")
    plt.ylabel("Percentage of Positive Cases")
    plt.title("Cumulative Gains Chart (Lift Chart)")
    plt.legend()
    plt.savefig(f"{model_name}_cumulative_gains_chart.png")
    #mlflow.log_artifact(f"{model_name}_cumulative_gains_chart.png")
    plt.close()
    
import shap
import copy
from imblearn.pipeline import Pipeline 
from sklearn.compose import ColumnTransformer 

def log_shap_plot(model: Pipeline, X_test: pd.DataFrame, feature_names=None, model_name="model"):
    # Deepcopy the pipeline to separate preprocessor
    preprocessor = copy.deepcopy(model)
    preprocessor.steps = [(name, step) for name, step in preprocessor.steps if name != 'model']
    
    # Transform X_test using the fitted preprocessor
    X_test_transformed = preprocessor.transform(X_test)
    
    # Get transformed feature names if the preprocessor is a ColumnTransformer
    if feature_names is None and isinstance(preprocessor.named_steps['preprocessor'], ColumnTransformer):
        feature_names = list(preprocessor.named_steps['preprocessor'].get_feature_names_out())
    
    # Convert transformed data to a DataFrame with feature names
    X_test_transformed = pd.DataFrame(X_test_transformed, columns=feature_names)
    
    # Extract the final model
    final_model = model.named_steps['model']
    
    # Initialize SHAP Explainer
    explainer = shap.Explainer(final_model, X_test_transformed)
    shap_values = explainer(X_test_transformed)

    # Ensure shap_values is in the right format (numpy array)
    if isinstance(shap_values, list):
        shap_values = shap_values[0]  # Ensure we only get the first output if it's a list

    # Summary plot (beeswarm)
    plt.figure()
    shap.summary_plot(shap_values, X_test_transformed, show=False, feature_names=feature_names)
    plt.title("SHAP Summary Plot")
    plt.savefig(f"{model_name}_shap_summary_plot.png")
    mlflow.log_artifact(f"{model_name}_shap_summary_plot.png")
    plt.close()
    
    # Individual force plot (example for first prediction)
    plt.figure()
    shap.force_plot(explainer.expected_value, shap_values[0, :], X_test_transformed.iloc[0], matplotlib=True)
    plt.savefig(f"{model_name}_shap_force_plot.png")
    mlflow.log_artifact(f"{model_name}_shap_force_plot.png")
    plt.close()
    
    
# Example usage
def log_classification_artefacts(model, X_test: pd.DataFrame, y_test: pd.Series, feature_names: list, model_name: str) -> None:
    """Log confusion matrix, ROC-AUC, and feature importance for a model."""
    plot_confusion_matrix(model, X_test, y_test, model_name)
    #plot_roc_auc(model, X_test, y_test, model_name)
    #plot_feature_importance(model, feature_names, model_name)
    #plot_feature_importance_main(model.named_steps['model'], feature_names, model_name)
    plot_feature_importance_best(model.named_steps['model'], feature_names, model_name)
    plot_roc_auc_curve_seaborn(model, X_test, y_test, model_name)
    log_precision_recall_curve(model, X_test, y_test, model_name)
    log_cumulative_gains_chart(model, X_test, y_test, model_name) 
    log_shap_plot(model, X_test, feature_names=feature_names, model_name=model_name)

