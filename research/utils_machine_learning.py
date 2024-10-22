import re
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import logging
from typing import List, Tuple
import pandas.api.types as ptypes


# Metric
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    roc_auc_score, roc_curve, auc,
    fbeta_score, make_scorer
)

# file and data management
import urllib.request
import zipfile

def rename_columns_to_snake_case(df: pd.DataFrame) -> pd.DataFrame:
    """
    Renames the columns of a DataFrame to snake_case, handling camel case, acronyms, Pascal case, hyphens, 
    multiple spaces, and already snake_case columns.
    
    Args:
        df: The DataFrame to rename.
    
    Returns:
        A new DataFrame with the columns renamed to snake_case.
    """
    def to_snake_case(col_name):
        # Replace hyphens or multiple spaces with an underscore
        col_name = re.sub(r'[-\s]+', '_', col_name)
        # Handle acronyms and split camel case / Pascal case
        col_name = re.sub(r'(?<!^)(?=[A-Z])', '_', col_name).lower()
        # Replace multiple underscores with a single underscore
        return re.sub(r'_+', '_', col_name)

    # Apply the snake_case function to all column names
    return df.rename(columns=lambda col: to_snake_case(col))


def extract_zip(src: str, dst: str, member_name: str) -> pd.DataFrame:
    """Function to extract a member file from a zip file and read it into a pandas 
    DataFrame.

    Parameters:
        src (str): URL of the zip file to be downloaded and extracted.
        dst (str): Local file path where the zip file will be written.
        member_name (str): Name of the member file inside the zip file 
            to be read into a DataFrame.

    Returns:
        pandas.DataFrame: DataFrame containing the contents of the 
            member file.

    usage:
        raw = extract_zip(url, fname, member_name)

    example:
        url = 'https://github.com/mattharrison/datasets/raw/master/data/kaggle-survey-2018.zip'
        fname = 'kaggle-survey-2018.zip'
        member_name = 'multipleChoiceResponses.csv'
    """    
    url = src
    fname = dst
    fin = urllib.request.urlopen(url)
    data = fin.read()
    with open(dst, mode='wb') as fout:
        fout.write(data)
    with zipfile.ZipFile(dst) as z:
        kag = pd.read_csv(z.open(member_name))
        kag_questions = kag.iloc[0]
        raw = kag.iloc[1:]
        return raw

def missing_data(data: pd.DataFrame, name: str, width: int, height: int):
    ''' Display missing data
        Usage: missing_data(df, "marketing campign data", 15, 8)
    '''
    
    total = data.isnull().sum().sort_values(ascending = False)
    percent = (data.isnull().sum()/data.isnull().count()*100).sort_values(ascending = False)
    temp = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    df_missing = temp[temp["Percent"] > 0]
    features_completed = temp[temp["Percent"] == 0]
    
    f,ax =plt.subplots(figsize=(width, height))
    plt.xticks(rotation='90')
    fig=sns.barplot(df_missing.index, df_missing["Percent"])
    plt.ylabel('% Missing values', fontsize=15)
    plt.title('Percentage of missing values in: '+name, fontsize=22, fontweight='bold')
    
    return features_completed, plt.show()


def handle_outliers(df: pd.DataFrame, cols: list[str]):
    """Handles outliers in multiple columns using the IQR method."""
    q1 = df[cols].quantile(0.25)
    q3 = df[cols].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    return df.loc[~((df[cols] < lower_bound) | (df[cols] > upper_bound)).any(axis=1)]


def plot_stats(feature, target, data, label_rotation=False, horizontal_layout=True):
    temp = data[feature].value_counts()
    df1 = pd.DataFrame({feature: temp.index,'Number of Count': temp.values})

    # Calculate the percentage of target=1 per category value
    cat_perc = data[[feature, target]].groupby([feature],as_index=False).mean()
    cat_perc[target] = cat_perc[target]*100
    cat_perc.sort_values(by=target, ascending=False, inplace=True)
    
    if(horizontal_layout):
        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12,6))
    else:
        fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(12,14))
        
    # 1. Count plot of categorical column     
    s = sns.countplot(ax=ax1, x = feature, hue=target, data=data, palette=['r','g'])
    ax1.legend(data[target].unique().tolist())
    ax1.set_title(feature+'\n', fontdict={'fontsize' : 15, 'fontweight' : 'bold'}) 
    if(label_rotation):
        s.set_xticklabels(s.get_xticklabels(),rotation=90)
    
    # 2. Subplot 2: Percentage of defaulters within the categorical column
    s = sns.barplot(ax=ax2, x = feature, y=target, order=cat_perc[feature], data=cat_perc, palette="rocket")
    plt.ylabel('Percent of target with value 1 [%]', fontsize=15)
    plt.tick_params(axis='both', which='major', labelsize=10)
    ax2.set_title(feature+" (% Default)"+'\n', fontdict={'fontsize' : 15, 'fontweight' : 'bold'})
    if(label_rotation):
        s.set_xticklabels(s.get_xticklabels(),rotation=90)
            
    return plt.show(), display(df1)


def grid_cv_result(estimator):
    """make a fancy df from estimator.cv_result"""
    
    res = estimator.cv_results_
    res = pd.DataFrame(res)
    cols = [i for i in res.columns if "split" not in i]
    res = res[cols]
    res = res.sort_values("rank_test_score")
    display(res)
    
    return res

def score_classifier(y_test, y_pred, fit_time=None, debug_info=None):
    """
    Compute and print train score, test score, and fitting time.
    
    Args:
      estimator: The model estimator to be evaluated.
      X_train: The training data features.
      X_test: The test data features.
      y_train: The training data labels.
      y_test: The test data labels.
      fit_time: The time taken to fit the estimator (optional).
    
    tr_score = round(estimator.score(X_train, y_train), 4)  # Accuracy score
    te_score = round(estimator.score(X_test, y_test), 4)
    
    print(type(estimator).__name__)
    print(f"score train: {tr_score}, score test: {te_score}")

    """

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    test_scores = {        
        "Accuracy Score": accuracy,
        "Precision Score": precision,
        "Recall Score": recall,
        "F1-Score": f1
    }
    if debug_info is not None:
        for key, value in test_scores.items():
            print(key, value)
   
    if fit_time:
        print(f"time: {fit_time}")

    return test_scores


def plot_confusion_matrix(confusion_mat, title="Confusion Matrix"):
    """
    Creates and displays a confusion matrix using Seaborn's heatmap.
    
    Args:
      y_true: Ground truth labels for the data.
      y_predicted: Predicted labels by the model.
      title: Title for the confusion matrix plot (default: "Confusion Matrix").
      figsize: Size of the figure (default: (4, 4)).
      cmap: Colormap used for the heatmap (default: "RdPu").
      annot: Boolean indicating whether to annotate values inside the heatmap (default: True).
      fmt: Format string for the annotations (default: "d" for integer).
      annot_kws: Keyword arguments for the annotation text (default: {"size": 10}).
    """

    figsize=(4, 4) 
    cmap="RdPu"
    annot=True
    fmt="d" 
    annot_kws={"size": 10}
    
    # To return to values
    confusion_matrix_clf = confusion_mat.tolist()
    mat_values_dict = {"true_pos" : confusion_matrix_clf[1][1],
                        "true_neg" : confusion_matrix_clf[0][0],
                        "false_pos" : confusion_matrix_clf[0][1],
                        "false_neg" : confusion_matrix_clf[1][0]
                    }
    
    # Create pandas dataframe for easier manipulation
    mat = pd.DataFrame(confusion_mat)
    
    # Set column names with "pred_" prefix
    mat.columns = [f"pred_{i}" for i in mat.columns]
    
    # Set row names with "test_" prefix
    mat.index = [f"test_{i}" for i in mat.index]
    
    # Create the figure and subplot
    plt.figure(figsize=figsize)
    ax = plt.subplot()
    
    # Generate heatmap using Seaborn
    sns.heatmap(mat, annot=annot, fmt=fmt, annot_kws=annot_kws, ax=ax, cmap=cmap)
    
    # Set title and display the plot
    plt.title(title, fontsize=10, fontweight="bold")
    plt.show()

    return mat_values_dict




def plot_roc_auc_curve_seaborn(y_test, y_pred_proba, title="ROC Curve"):
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
  plt.title(title, fontsize=16)

  # Add legend
  plt.legend(loc=4)

  # Display the plot
  plt.show()
  return {"auc": auc}


def plot_roc_auc_curve(model, X_test, y_test, title="ROC Curve", figsize=(8, 6)):
    """
    Plots the ROC curve and calculates the AUC score for a given model using Matplotlib.

    Args:
        model: The trained model to evaluate.
        X_test: The test data features.
        y_test: The test data labels.
        title: Title for the plot (default: "ROC Curve").
        figsize: Size of the figure (default: (8, 6)).
    """
    
    # Get predicted probabilities
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Calculate ROC curve and AUC score
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    auc = roc_auc_score(y_test, y_pred_proba)

    # Create the plot
    plt.figure(figsize=figsize)

    # Plot ROC curve using Matplotlib
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc:.3f})', color='violet', linewidth=2)

    # Plot baseline performance line
    plt.plot([0, 1], [0, 1], linestyle='--', label='Baseline performance', color='gray')

    # Set axis labels and title
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=14)
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=14)
    plt.title(title, fontsize=16)

    # Add legend
    plt.legend(loc=4)

    # Display the plot
    plt.show()
    
    return {"auc": auc}


def get_feature_importance(feature_names, importance_scores, n_top, estimator_name):
                         
        """
        Gets feature importances for the best model obtained during training.
        
        This implementation assumes a tree-based model is used within the FeatureClassifier.
        
        Args:
          X_test (pd.DataFrame): Testing data features (unused in this implementation).
        
        Returns:
          dict: Dictionary of feature names and their importance scores.
        """     
       
        # Create a DataFrame from feature names and importances
        data = {'Feature': feature_names, 'Score': importance_scores}
        df = pd.DataFrame(data)
        
        
        # Take the absolute value of the score
        df['Abs_Score'] = np.abs(df['Score'])
        
        df_sorted = df.sort_values(by="Abs_Score", ascending=False)
        if n_top:
            # Sort by absolute value of score in descending order (top 10)
            df_sorted = df_sorted.head(10)
        
        # Define a color palette based on score values (positive = green, negative = red)
        colors = ["green" if score > 0 else "red" for score in df_sorted["Score"]]
        
        # Create the bar chart with Seaborn
        sns.barplot(x="Feature", y="Score", hue="Feature", legend=False, data=df_sorted, palette=colors)
        
        # Customize the plot for better visual appeal
        plt.xlabel("Feature")
        plt.ylabel("Feature Importance Score")
        plt.title(f"Feature Importance in {estimator_name} Supervised Classification")
        plt.xticks(rotation=45, ha="right")  # Rotate x-axis labels for better readability
        plt.tight_layout()  # Adjust spacing between elements
        
        # Display the plot
        plt.show()


def get_feature_importance_scores(best_estimator, estimator_name, 
                                  feature_names, n_top_features=None):
    # Get the Feature Importance
    try:
        importance = best_estimator.named_steps[estimator_name]
        # Check if model type is LogisticRegression before accessing feature_importances_
        if hasattr(importance, 'feature_importances_'):
            importance_scores = importance.feature_importances_
        else:
            # Handle models other than LogisticRegression (assuming coef_ approach)
            importance_scores = importance.coef_[0]
    except AttributeError:
        print("Unexpected error while retrieving feature importance scores.")
        importance_scores = None  # Or set importance_scores to an empty list/placeholder
    else:
        # Call get_feature_importance only if importance_scores retrieved
        if importance_scores is not None:
            get_feature_importance(feature_names=feature_names, importance_scores=importance_scores, n_top=n_top_features, estimator_name=estimator_name)
    

# Configure logging
#logging.basicConfig(level=print, format='%(asctime)s %(lineno)s - %(levelname)s - %(message)s')

def check_df(dataframe: pd.DataFrame, head: int = 5, check_na: bool = True) -> None:
    """
    Provides a detailed overview of a DataFrame, including its shape, data types, head/tail rows, 
    missing values, and quantiles.

    Parameters:
    ----------
    dataframe : pd.DataFrame
        The DataFrame to be analyzed.
    head : int, optional
        Number of rows to display from the head and tail of the DataFrame (default is 5).
    check_na : bool, optional
        Whether to display missing values information (default is True).

    Returns:
    -------
    None
    """

    try:
        print("### Shape ###")
        print(f"Rows: {dataframe.shape[0]}, Columns: {dataframe.shape[1]}")

        print("### Data Types ###")
        print(f"\n{dataframe.dtypes}")

        print(f"### Head ({head} rows) ###")
        print(f"\n{dataframe.head(head)}")

        print(f"### Tail ({head} rows) ###")
        print(f"\n{dataframe.tail(head)}")

        if check_na:
            print("### Missing Values (NA) ###")
            missing_values = dataframe.isnull().sum()
            if missing_values.sum() == 0:
                print("No missing values.")
            else:
                print(f"\n{missing_values}")

        print("### Quantiles ###")
        quantiles = dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T
        print(f"\n{quantiles}")

    except Exception as e:
        print(f"An error occurred while checking the DataFrame: {e}")


def grab_col_names(dataframe: pd.DataFrame, 
                   categorical_threshold: int = 10, 
                   cardinal_threshold: int = 20) -> Tuple[List[str], List[str], List[str], List[str]]:
    """
    Retrieves the names of categorical, numerical, and cardinal variables from the given DataFrame.

    Parameters
    ----------
    dataframe : pd.DataFrame
        The DataFrame from which variable names are to be extracted.
    categorical_threshold : int, optional
        Threshold for numeric variables to be considered as categorical (default is 10 unique values).
    cardinal_threshold : int, optional
        Threshold for categorical variables to be considered as cardinal (default is 20 unique values).

    Returns
    -------
    categorical_cols : List[str]
        List of categorical variable names.
    numerical_cols : List[str]
        List of numerical variable names.
    cardinal_cols : List[str]
        List of cardinal variable names.
    nominal_cols : List[str]
        List of nominal variable names.
    """

    try:
        # Identify categorical columns using pandas type checking
        categorical_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
        
        # Identify nominal columns (numerical variables with fewer unique values than the categorical_threshold)
        nominal_cols = [col for col in dataframe.columns if
                    dataframe[col].nunique() < categorical_threshold and dataframe[col].dtypes != "O"]

        # Identify cardinal columns (categorical variables with more unique values than the cardinal_threshold)
        cardinal_cols = [col for col in dataframe.columns if
                     dataframe[col].nunique() > cardinal_threshold and dataframe[col].dtypes == "O"]

        # Combine categorical and nominal columns, excluding cardinal columns
        categorical_cols += nominal_cols
        categorical_cols = [col for col in categorical_cols if col not in cardinal_cols]

        # Identify numerical columns, excluding the ones already marked as categorical
        numerical_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
        numerical_cols = [col for col in numerical_cols if col not in categorical_cols]

        # Log the results
        print(f"Observations: {dataframe.shape[0]}")
        print(f"Variables: {dataframe.shape[1]}")
        print(f"Categorical Columns: {len(categorical_cols)} -> {categorical_cols}")
        print(f"Numerical Columns: {len(numerical_cols)} -> {numerical_cols}")
        print(f"Cardinal Columns: {len(cardinal_cols)} -> {cardinal_cols}")
        print(f"Nominal Columns: {len(nominal_cols)} -> {nominal_cols}")

        return categorical_cols, numerical_cols, cardinal_cols, nominal_cols

    except Exception as e:
        logging.error(f"An error occurred in grab_col_names: {e}")
        return [], [], [], []
    

# Observe categorical variable distributions
def cat_summary(dataframe, col_name, plot=False):
    if dataframe[col_name].dtypes == "bool":
        dataframe[col_name] = dataframe[col_name].astype(int)

    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")

    if plot:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
        
        # Create a color palette with unique colors for each category
        length_cat = len(dataframe[col_name].unique())
        palette = sns.color_palette('tab10', length_cat)  # Define a palette with unique colours

        # Create the countplot with custom colors
        sns.countplot(x=col_name, data=dataframe, hue=col_name, legend=False, palette=palette, ax=ax1)
        ax1.set_title(f"Frequency of {col_name}")
        ax1.set_ylabel("TARGET_COUNT")
        ax1.tick_params(axis="x", rotation=45)
        
        # RATIO (Pie Chart) with matching colours
        values = dataframe[col_name].value_counts()
        ax2.pie(x=values, labels=values.index, autopct="%1.1f%%", startangle=90, colors=palette)  # Use same palette
        ax2.set_title(f"RATIO by {col_name}")
        ax2.legend(labels=[f"{index} - {value/sum(values)*100:.2f}%" for index, value in zip(values.index, values)],
                   loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
        
        plt.tight_layout()
        plt.show()




import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def plot_numerical_summary(dataframe: pd.DataFrame, numerical_col: str, hist_bins: int = 20) -> None:
    """
    Plots the histogram, boxplot, KDE, and QQ plot for the given numerical column.
    
    Parameters
    ----------
    dataframe : pd.DataFrame
        The DataFrame containing the numerical column.
    numerical_col : str
        The name of the numerical column to be plotted.
    hist_bins : int, optional
        Number of bins for the histogram (default is 20).
    """
    try:
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        
        # Histogram
        plt.subplot(2, 2, 1)
        dataframe[numerical_col].hist(bins=hist_bins)
        plt.xlabel(numerical_col)
        plt.title(f"{numerical_col} Distribution")
        
        # Boxplot
        plt.subplot(2, 2, 2)
        sns.boxplot(y=numerical_col, data=dataframe)
        plt.title(f"Boxplot of {numerical_col}")
        plt.xticks(rotation=90)
        
        # Density Plot (KDE)
        plt.subplot(2, 2, 3)
        sns.kdeplot(dataframe[numerical_col], fill=True)
        plt.xlabel(numerical_col)
        plt.title(f"{numerical_col} KDE")
        
        # QQ Plot
        plt.subplot(2, 2, 4)
        stats.probplot(dataframe[numerical_col], dist="norm", plot=plt)
        plt.title(f"{numerical_col} QQ Plot")
        
        plt.tight_layout()
        plt.show(block=True)

    except Exception as e:
        print(f"Error while plotting {numerical_col}: {e}")

def num_summary(dataframe: pd.DataFrame, numerical_col: str, plot: bool = False, hist_bins: int = 20):
    """
    Summarizes the statistics of a numerical column, including descriptive statistics and optional plotting.
    
    Parameters
    ----------
    dataframe : pd.DataFrame
        The DataFrame containing the numerical column.
    numerical_col : str
        The name of the numerical column to be summarized.
    plot : bool, optional
        Whether or not to plot the column's distribution (default is False).
    hist_bins : int, optional
        Number of bins for the histogram if plot is True (default is 20).
    """
    try:
        # Set custom quantiles for detailed summary
        quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]

        print(f"Numerical Column: {numerical_col}")
        print("##########################################")
        summary_stats = dataframe[numerical_col].describe(percentiles=quantiles).T
        print(f"{summary_stats}")
        print("##########################################")
        
        # Optional: Plotting the column's distribution if requested
        if plot:
            plot_numerical_summary(dataframe, numerical_col, hist_bins)
    
    except KeyError:
        print(f"Column {numerical_col} does not exist in the dataframe.")
    except Exception as e:
        print(f"An error occurred while summarizing {numerical_col}: {e}")



def target_summary_with_categorical_data(dataframe, target, categorical_col):
    """
    It gives the summary of specified categorical column name according to target column.

    Args:
        dataframe (dataframe): The dataframe from which variables names are to be retrieved.
        target (string): The target column name are to be retrieved. 
        categorical_col (string): The categorical column names are to be retrieved.
    """
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean()}), end="\n\n\n")



def target_summary_with_num(dataframe, target, numerical_col, bins=10):
    df = dataframe.copy()
    
    summary_df = df.groupby(target).agg({numerical_col: "mean"})
    
    df["binned"] = pd.cut(df[numerical_col], bins=bins)
    binned_summary_df = df.groupby("binned").agg({target: "mean"})

    return binned_summary_df, numerical_col



def correlation_matrix(df, cols):
    """
    It gives the correlation of numerical variables with each other.

    Args:
        df (dataframe): The dataframe from which variables names are to be retrieved.
        cols (list): The column name list are to be retrieved.
    """
    fig = plt.gcf()
    fig.set_size_inches(10,8)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize = 10)
    fig = sns.heatmap(df[cols].corr(), annot=True, linewidths=0.5,
                      annot_kws={"size":12},linecolor="w",
                      cmap="RdBu")
    plt.savefig("Correlation Matrix.png")
    plt.show(block=True)


def high_correlated_cols(dataframe, plot=False, corr_th=0.88):
    corr = dataframe.corr(numeric_only=True)
    cor_matrix = corr.abs()
    upper_triangle_matrix = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(bool))
    drop_list = [col for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col] > corr_th)]

    # Optionally, plot a heatmap
    if plot:
        import seaborn as sns
        import matplotlib.pyplot as plt
        sns.set(rc={'figure.figsize': (18, 13)})
        sns.heatmap(corr, cmap='magma', annot=True, fmt='.2f', annot_kws={"size": 7})
        plt.show()

    return drop_list


def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    """
    Return the lower and upper limit of each columns. 

    Args:
        dataframe (dataframe): The dataframe from which variables names are to be retrieved.
        col_name (string): The column names from which features names are to be retrieved
        q1 (float, optional): _description_. Defaults to 0.25.
        q3 (float, optional): _description_. Defaults to 0.75.

    Returns:
        low_limit (float): Returns the lower limit by using quartile1 and interquantile range values for each columns.
        up_limit (float) : Returns the upper limit by using quartile3 and interquantile range values for each columns.
    """
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit



def replace_with_thresholds(dataframe, col_name):
    """
    Replacing values ​​in Outlier with lower and upper bounds

    Args:
        dataframe (dataframe): The dataframe from which variables names are to be retrieved.
        col_name (string): The column name with the outlier values are to be retrieved.
    """
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    dataframe.loc[(dataframe[col_name] < low_limit), col_name] = low_limit
    dataframe.loc[(dataframe[col_name] > up_limit), col_name] = up_limit


def check_outlier(dataframe, col_name, q1=0.05, q3=0.95):
    """
    Checking the outliers for each columns

    Args:
        dataframe (dataframe): The dataframe from which variables names are to be retrieved.
        col_name (string): The column names from which features names are to be retrieved
        q1 (float, optional): Quartile 1 Value of specified column. Defaults to 0.25.
        q3 (float, optional): Quartile 3 Value of specified column.. Defaults to 0.75.

    Returns:
        boolean: If the specified column has outlier values, return True. If not, return False.
    """
    low_limit, up_limit = outlier_thresholds(dataframe, col_name, q1, q3)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False
    
    
def check_df_with_commentries(dataframe: pd.DataFrame, head: int = 5, check_na: bool = True, target_col='y') -> None:
    """
    Provides a detailed overview of a DataFrame, including its shape, data types, head/tail rows, 
    missing values, and quantiles.

    Parameters:
    ----------
    dataframe : pd.DataFrame
        The DataFrame to be analyzed.
    head : int, optional
        Number of rows to display from the head and tail of the DataFrame (default is 5).
    check_na : bool, optional
        Whether to display missing values information (default is True).

    Returns:
    -------
    None
    """

    try:
        print(f"============= OBSERVATIONS ==============")
        
        dtype_summary = ", ".join([f"{dtype}: {count}" for dtype, count in dataframe.dtypes.value_counts().items()])
        print(f">> The shape of the dataset is {dataframe.shape}, comprising of {dataframe.shape[0]} rows and {dataframe.shape[1]} columns")
        print(f">> The data types have {dtype_summary}")
        print(f">> There are {dataframe.duplicated().sum()} duplicate rows. These duplicate rows have to be deleted")
        print(f">> There are {dataframe.isnull().sum().sum()} null values")
        print(f">> The target column {target_col} has a value of {dataframe[target_col].unique()}")

    except Exception as e:
        print(f"An error occurred while checking the DataFrame: {e}")

