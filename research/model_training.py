# Let's create the `trainer.py` and `tuner.py` modules by refactoring core functionalities from `model_trainer.py`.

# Step 1: Define `trainer.py` with a `Trainer` class to handle the model training process and `tuner.py` for tuning.


import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class Trainer:
    def __init__(self, model, data, labels, preprocess_pipeline=None):
        self.model = model
        self.data = data
        self.labels = labels
        self.preprocess_pipeline = preprocess_pipeline

    def train(self):
        """Train the model with the provided data."""
        if self.preprocess_pipeline:
            self.data = self.preprocess_pipeline.fit_transform(self.data)
        self.model.fit(self.data, self.labels)
        return self.model

    def evaluate(self, test_data, test_labels):
        """Evaluate the model using standard metrics."""
        if self.preprocess_pipeline:
            test_data = self.preprocess_pipeline.transform(test_data)
        predictions = self.model.predict(test_data)
        metrics = {
            "accuracy": accuracy_score(test_labels, predictions),
            "precision": precision_score(test_labels, predictions, average='weighted'),
            "recall": recall_score(test_labels, predictions, average='weighted'),
            "f1": f1_score(test_labels, predictions, average='weighted')
        }
        return metrics

    def save_model(self, path):
        """Save the trained model to the specified path."""
        joblib.dump(self.model, path)

    def load_model(self, path):
        """Load a saved model from the specified path."""
        self.model = joblib.load(path)
        return self.model

from sklearn.model_selection import GridSearchCV

class Tuner:
    def __init__(self, model, param_grid, cv=5, scoring='accuracy'):
        self.model = model
        self.param_grid = param_grid
        self.cv = cv
        self.scoring = scoring

    def tune(self, data, labels):
        """Performs grid search over hyperparameters."""
        grid_search = GridSearchCV(self.model, self.param_grid, cv=self.cv, scoring=self.scoring)
        grid_search.fit(data, labels)
        return grid_search.best_estimator_, grid_search.best_params_