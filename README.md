# Auto Insurance Claims Fraud Detection Project

## 1. **Preamble**

Fraudulent insurance claims are a significant problem in the Auto Insurance industry, costing companies millions annually. This project aims to address this issue by developing a machine learning model that can effectively predict whether an insurance claim is fraudulent. Using a dataset of past claims provided by an insurance company, this project will explore several machine learning algorithms to find the most suitable approach for fraud detection. By automating the detection process, the model can significantly reduce fraudulent payouts and improve the overall efficiency of claim handling.

## 2. **Objective**

The primary goal of this project is to build a predictive model that classifies insurance claims as either fraudulent or non-fraudulent. The project also aims to perform a comparative analysis of different machine learning algorithms, ensuring the best algorithm is chosen based on key performance metrics.

Specific objectives include:
- Developing a binary classification model for fraudulent claims prediction.
- Handling class imbalance using appropriate techniques.
- Tuning hyperparameters to enhance the performance of models.
- Comparing various machine learning models based on evaluation metrics such as precision, recall, F1-score, and AUC-ROC.

## 3. **Purpose and Motivation**

Insurance fraud detection is a critical problem faced by the Auto Insurance industry, often resulting in financial losses. Manual identification of fraudulent claims is both challenging and resource-intensive. Machine learning provides an opportunity to automate this process by identifying patterns in the data that indicate fraud. If successful, this project can result in significant cost savings and operational efficiencies for insurance companies, making it a highly valuable real-world application of machine learning.

## 4. **Dataset and Features**

**Dataset Information:**
- **Source**: Proprietary dataset provided by the insurance company [or public repository if applicable].
- **Size**: The dataset contains [number of instances] and 40 features.

**Target Variable**:
- `fraud_reported`: A binary variable indicating whether the claim is fraudulent (Yes) or not (No).

**Feature Overview**:
- **Numerical Features**: `months_as_customer`, `policy_deductable`, `total_claim_amount`, `capital_gains`, etc.
- **Categorical Features**: `policy_state`, `incident_type`, `authorities_contacted`, `auto_make`, etc.

## 5. **Data Preprocessing**

**Handling Missing Values**:
- Missing values in categorical and numerical features were handled using imputation strategies. For categorical features, the most frequent value was used, while for numerical features, median imputation was applied.

**Feature Engineering**:
- **Date Transformation**: Features such as `policy_bind_date` and `incident_date` were transformed into numerical features representing days since a particular event.
- **Interaction Terms**: New features representing interactions between `incident_type` and `incident_severity` were created to capture important patterns.
- **Skewness Correction**: Features like `capital_gains` were transformed using the Yeo-Johnson transformation to correct skewness.

**Data Standardization**:
- **Scaling**: Numerical features were standardised using `StandardScaler`.
- **One-Hot Encoding**: Applied to categorical variables such as `policy_state`, `incident_city`.
- **Ordinal Encoding**: Used for ordered categorical features such as `incident_severity`.

**Class Imbalance Handling**:
- The dataset was imbalanced, with fraudulent claims being the minority class. To address this, the **SMOTEENN** technique (a combination of oversampling and undersampling) was used to resample the dataset.

## 6. **Model Selection and Training**

**Algorithms Considered**:
The following machine learning models were considered:
- Random Forest
- Decision Tree
- Gradient Boosting
- Logistic Regression
- K-Nearest Neighbors
- XGBoost
- CatBoost
- Support Vector Classifier
- AdaBoost

**Hyperparameter Tuning**:
- Hyperparameters for models such as Random Forest, XGBoost, and CatBoost were tuned using **RandomizedSearchCV** to optimise their performance.
- Parameters tuned include the number of trees, maximum depth, learning rate, and the number of features considered for splits.

**Evaluation Metrics**:
To evaluate model performance, the following metrics were used:
- **Accuracy**: Measures the overall correctness of the model.
- **Precision**: Focuses on the accuracy of the positive predictions (fraudulent claims).
- **Recall**: Measures the ability of the model to identify fraudulent claims.
- **F1-Score**: A balance between precision and recall.
- **ROC AUC**: Evaluates the model's ability to distinguish between classes.

## 7. **Comparison Study**

**Criteria**:
The algorithms were compared based on the following criteria:
- Performance on imbalanced data.
- Sensitivity to hyperparameter tuning.
- Computational efficiency (training time).
- Overall performance on evaluation metrics, particularly precision and recall due to the imbalanced nature of the dataset.

**Results**:
A summary of the model performances is presented, where models like **XGBoost** and **Random Forest** showed superior performance in terms of precision and recall. Logistic Regression and KNN performed less favourably due to their inability to capture the complex patterns of fraud in the dataset.

## 8. **Reproducibility Steps**

To ensure the reproducibility of this project, the following instructions are provided:

**Data Acquisition**:
- Obtain the dataset from the provided source (e.g., proprietary insurance dataset or a public repository). Ensure that the data is correctly formatted and follows the structure provided in the feature list above.

**Data Preprocessing**:
- Follow the data preprocessing steps outlined in the project. This includes handling missing values, feature engineering, scaling, and handling class imbalance.
- Ensure that the `ColumnTransformer` and `Pipeline` setup is used for consistent preprocessing.

**Model Training**:
- Train the models using the preprocessed dataset. The models listed should be trained with the default parameters first, followed by hyperparameter tuning using **RandomizedSearchCV** for optimal results.

**Evaluation**:
- Evaluate the models using the defined evaluation metrics. Store the results and compare them to ensure that the models generalise well to unseen data.

By following these steps, the project can be easily reproduced, and the models can be re-trained and fine-tuned for further experimentation.

## 9. **Conclusion**

This project provides a comprehensive solution to the problem of auto insurance fraud detection using machine learning. By leveraging advanced models like XGBoost, Random Forest, and CatBoost, and addressing class imbalance with SMOTEENN, the model can reliably predict fraudulent claims. The results of this project can lead to significant cost savings for the insurance company and provide a blueprint for similar fraud detection problems across other industries.

# Technical Documentation

##  Resources
- Pyenv: https://github.com/pyenv/pyenv
- Vs code: https://code.visualstudio.com/download
- Git: https://git-scm.com/
- Flowchart: https://whimsical.com/
- MLOPs Tool: https://www.evidentlyai.com/
- Data link: https://www.kaggle.com/datasets/moro23/easyvisa-dataset


## Git commands

```bash
git add .

git commit -m "Updated"

git push origin main

# To restore a file from a previous commit
git checkout <commit-hash> -- path/to/file
# example
git checkout a1b2c3d4 -- important_file.txt
```


## How to run?

```bash
pyenv virtualenv 3.11 venv311_insure
```

```bash
pyenv activate venv311_insure
```
OR

```bash
pyenv pyenv local venv311_insure
```

```bash
pip install -r requirements.txt
```

## Workflows

1. Update config.yaml
2. Update secrets.yaml [Optional]
3. Update params.yaml
4. Update the entity
5. Update the configuration manager in src config
6. Update the components
7. Update the pipeline
8. Update the main.py


## MLflow

```bash
mlflow server --host 0.0.0.0 --port <PORT>
# mlflow server --host 127.0.0.1 --port 5001 # to be actively runing on another terminal
mlflow.set_tracking_uri("http://127.0.0.1:5001")
```


## AWS-CICD-Deployment-with-Github-Actions

### 1. Login to AWS console.

### 2. Create IAM user for deployment

	#with specific access

	1. EC2 access : It is virtual machine

	2. ECR: Elastic Container registry to save your docker image in aws


	#Description: About the deployment

	1. Build docker image of the source code

	2. Push your docker image to ECR

	3. Launch Your EC2

	4. Pull Your image from ECR in EC2

	5. Lauch your docker image in EC2

	#Policy:

	1. AmazonEC2ContainerRegistryFullAccess

	2. AmazonEC2FullAccess


### 3. Create ECR repo to store/save docker image
    - Save the URI: 439197624230.dkr.ecr.us-east-1.amazonaws.com/insurance


### 4. Create EC2 machine (Ubuntu)

### 5. Open EC2 and Install docker in EC2 Machine:


	#optinal

	sudo apt-get update -y

	sudo apt-get upgrade

	#required

	curl -fsSL https://get.docker.com -o get-docker.sh

	sudo sh get-docker.sh

	sudo usermod -aG docker ubuntu

	newgrp docker

### 6. Configure EC2 as self-hosted runner:
    setting>actions>runner>new self hosted runner> choose os> then run command one by one


### 7. Setup github secrets:

   - AWS_ACCESS_KEY_ID
   - AWS_SECRET_ACCESS_KEY
   - AWS_REGION: us-east-1
   - AWS_ECR_LOGIN_URI
   - ECR_REPOSITORY_NAME: insurance
