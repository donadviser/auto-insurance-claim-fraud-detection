class ModelTrainer:
    def __init__(
        self,
        data_ingestion_artefacts: DataIngestionArtefacts,
        data_transformation_artefact: DataTransformationArtefacts,
        model_trainer_config: ModelTrainerConfig,
    ):
        self.data_ingestion_artefacts = data_ingestion_artefacts
        self.data_transformation_artefact = data_transformation_artefact
        self.model_trainer_config = model_trainer_config

        # Load model parameters and create directories
        self.model_config = self.model_trainer_config.UTILS.read_yaml_file(filename=MODEL_CONFIG_FILE)
        self.model_trainer_artefacts_dir = self.model_trainer_config.MODEL_TRAINER_ARTEFACTS_DIR
        self.best_model_artefacts_dir = self.model_trainer_config.BEST_MODEL_ARTEFACTS_DIR
        os.makedirs(self.best_model_artefacts_dir, exist_ok=True)
        logging.info(f"Created Best Model Artefacts Directory: {self.best_model_artefacts_dir}")

        # Load datasets
        self.load_datasets()
        logging.info("Completed loading train and test datasets.")

        # Get schema configuration
        self.get_schema_config()
        logging.info("Completed reading the schema config file.")

        # Prepare target and feature variables
        self.prepare_data()

    def load_datasets(self):
        self.train_set = pd.read_csv(self.data_ingestion_artefacts.train_data_file_path)
        self.test_set = pd.read_csv(self.data_ingestion_artefacts.test_data_file_path)
        logging.info(f"Loaded train_set dataset from the path: {self.data_ingestion_artefacts.train_data_file_path}")
        logging.info(f"Loaded test_set dataset from the path: {self.data_ingestion_artefacts.test_data_file_path}")
        logging.info(f"Shape of train_set: {self.train_set.shape}, Shape of test_set: {self.test_set.shape}")

    def get_schema_config(self):
        schema_config = self.model_trainer_config.SCHEMA_CONFIG
        self.numerical_features = schema_config['numerical_features']
        self.onehot_features = schema_config['onehot_features']
        self.ordinal_features = schema_config['ordinal_features']
        self.transform_features = schema_config['transform_features']
        self.bins_hour = schema_config['incident_hour_time_bins']['bins_hour']
        self.names_period = schema_config['incident_hour_time_bins']['names_period']
        self.drop_columns = schema_config['drop_columns']
        self.yes_no_map = schema_config['yes_no_map']
        self.target_columns = schema_config['target_column']

    def prepare_data(self):
        X_train, y_train = self.train_set.drop(columns=[self.target_columns]), self.train_set[self.target_columns]
        X_test, y_test = self.test_set.drop(columns=[self.target_columns]), self.test_set[self.target_columns]

        self.X_train = X_train.copy()
        self.y_train = y_train.map(self.yes_no_map)  # Map target labels
        self.X_test = X_test.copy()
        self.y_test = y_test.map(self.yes_no_map)  # Map target labels
        logging.info("Completed setting the Train and Test X and y")

    def get_pipeline_model_and_params(self, classifier_name, trial, model_hyperparams=None):
        preprocessing_pipeline = self.create_preprocessing_pipeline(trial)

        pipeline_manager = PipelineManager(pipeline_type='ImbPipeline')
        self.add_pipeline_steps(pipeline_manager, preprocessing_pipeline, trial)

        model_factory = ModelFactory(classifier_name, model_hyperparams)
        model_obj = model_factory.get_model_instance()
        pipeline_manager.add_step('model', model_obj, position=7)

        return pipeline_manager.get_pipeline()

    def create_preprocessing_pipeline(self, trial):
        return PreprocessingPipeline(
            bins_hour=self.bins_hour,
            names_period=self.names_period,
            drop_columns=self.drop_columns,
            numerical_features=self.numerical_features,
            onehot_features=self.onehot_features,
            ordinal_features=self.ordinal_features,
            transform_features=self.transform_features,
            trial=trial
        )

    def add_pipeline_steps(self, pipeline_manager, preprocessing_pipeline, trial):
        steps = [
            ('create_new_features', 0),
            ('replace_class', 1),
            ('drop_cols', 2),
            ('column_transformer', 3),
            ('resampler', 4),
            ('scaler', 5),
            ('dim_reduction', 6)
        ]

        for step_name, position in steps:
            pipeline_manager.add_step(step_name, preprocessing_pipeline.build(step_name=step_name, trial=None), position)

        # Add resampler and scaler
        resampler_selector = ResamplerSelector(trial=trial)
        scaler_selector = ScalerSelector(trial=trial)
        dim_red_selector = DimensionalityReductionSelector(trial=trial)

        pipeline_manager.add_step('resampler', resampler_selector.get_resampler(), position=4)
        pipeline_manager.add_step('scaler', scaler_selector.get_scaler(), position=5)
        pipeline_manager.add_step('dim_reduction', dim_red_selector.get_dimensionality_reduction(), position=6)

    def get_classification_metrics(self, y_true, y_pred, y_pred_proba=None):
        accuracy = accuracy_score(y_true, y_pred)  # Calculate Accuracy
        f1 = f1_score(y_true, y_pred)  # Calculate F1-score
        precision = precision_score(y_true, y_pred)  # Calculate Precision
        recall = recall_score(y_true, y_pred)  # Calculate Recall
        roc_auc = roc_auc_score(y_true, y_pred_proba) if y_pred_proba is not None else roc_auc_score(y_true, y_pred)  # Calculate Roc
        detailed_report = classification_report(y_true, y_pred, output_dict=True)  # Detailed report

        return {
            'f1': f1,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'roc_auc': roc_auc,
            "classification_report": detailed_report
        }

    def objective(self, classifier_name: str, trial: optuna.Trial = None, scoring='f1') -> float:
        hyperparameter_tuner = HyperparameterTuner()
        model_hyperparams = hyperparameter_tuner.get_params(trial, classifier_name)
        pipeline = self.get_pipeline_model_and_params(classifier_name=classifier_name, trial=trial, model_hyperparams=model_hyperparams)

        # Cross-validation
        kfold = StratifiedKFold(n_splits=10)
        score = cross_val_score(pipeline, self.X_train, self.y_train, scoring=scoring, n_jobs=-1, cv=kfold, verbose=0, error_score='raise')
        score_training = score.mean()

        pipeline.fit(self.X_train, self.y_train)
        y_pred = pipeline.predict(self.X_test)
        y_pred_proba = pipeline.predict_proba(self.X_test)[:, 1]

        classification_metrics = self.get_classification_metrics(self.y_test, y_pred, y_pred_proba)
        classification_metrics['classifier_name'] = classifier_name
        classification_metrics['training_score'] = score_training

        return classification_metrics[scoring]

    def detailed_objective(self, classifier_name: str, trial: optuna.Trial = None, scoring='f1') -> ImbPipeline:
        hyperparameter_tuner = HyperparameterTuner()
        model_hyperparams = hyperparameter_tuner.get_params(trial, classifier_name)

        pipeline = self.get_pipeline_model_and_params(classifier_name, trial, model_hyperparams=model_hyperparams)
        logging.info(f"Fitted pipeline with the best parameters: {classifier_name}")

        return pipeline

    def run_optimization(self, config_path: str, n_trials: int = 100, scoring: str = 'f1') -> None:
        best_model_score = -1
        best_model = None
        best_params = None
        best_of_models = []
        best_pipeline = {}

        all_models = ["RandomForest", "DecisionTree", "XGBoost", "LGBM", "GradientBoosting"]

        for classifier_name in all_models:
            logging.info(f"Optimizing model: {classifier_name}")
            study = optuna.create_study(direction="maximize", sampler=TPESampler())
            study.optimize(lambda trial: self.objective(classifier_name, trial, scoring), n_trials=n_trials)

            best_trial_obj = study.best_trial
            score_training = best_trial_obj.value

            pipeline = self.detailed_objective(classifier_name=classifier_name, trial=study.best_trial, scoring=scoring)
            pipeline.fit(self.X_train, self.y_train)

            y_pred = pipeline.predict(self.X_test)
            y_pred_proba = pipeline.predict_proba(self.X_test)[:, 1]

            classification_metrics = self.get_classification_metrics(self.y_test, y_pred, y_pred_proba)
            classification_metrics['training_score'] = score_training 

            current_test_score = classification_metrics[scoring]
            trained_model_filename = f'{classifier_name}_pipeline{MODEL_SAVE_FORMAT}'
            trained_model_saved_path = os.path.join(self.model_trainer_artefacts_dir, trained_model_filename)
            joblib.dump((pipeline, classification_metrics), trained_model_saved_path)
            print(f'Serialized {classifier_name} pipeline and test metrics to {trained_model_saved_path}')

            logging.info(f"Model: {classifier_name}, Current Score: {current_test_score} | Best Model: {best_model}, Best Score: {best_model_score}")

            if current_test_score > best_model_score:
                best_model_score = current_test_score
                best_model = classifier_name
                best_params = study.best_trial.params
                best_pipeline = pipeline

        # Save the best model and its configuration
        best_model_filename = f'{best_model}_pipeline{MODEL_SAVE_FORMAT}'
        best_model_path = os.path.join(self.best_model_artefacts_dir, best_model_filename)
        joblib.dump((best_pipeline, best_model_score, best_params), best_model_path)
        print(f'Serialized Best Model: {best_model} to {best_model_path}')
