from sklearn.base import BaseEstimator, TransformerMixin
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import r2_score, mean_squared_error
import pandas as pd
import logging
import warnings
import mlflow
from mlflow.tracking import MlflowClient
import optuna
from optuna.samplers import TPESampler
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import random

class ANNRegressor(BaseEstimator, TransformerMixin):
    def __init__(self, neurons_layer1=128, neurons_layer2=64, neurons_layer3=32,
                 learning_rate=0.001, batch_size=64, epochs=100,
                 l1_reg=0.0, l2_reg=0.0, activation='relu',
                 dropout_rate=0.2, early_stopping_patience=5,
                 device=None, verbose=0):

        # Architecture params
        self.neurons_layer1 = neurons_layer1
        self.neurons_layer2 = neurons_layer2
        self.neurons_layer3 = neurons_layer3
        self.dropout_rate = dropout_rate
        self.activation = activation

        # Training params
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.early_stopping_patience = early_stopping_patience

        # Regularization
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg

        # System
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.verbose = verbose
        self.model = None

    class _ANN(nn.Module):
        def __init__(self, input_size, cfg):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(input_size, cfg['neurons_layer1']),
                self._get_activation(cfg['activation']),
                nn.Dropout(cfg['dropout_rate']),

                nn.Linear(cfg['neurons_layer1'], cfg['neurons_layer2']),
                self._get_activation(cfg['activation']),
                nn.Dropout(cfg['dropout_rate']),

                nn.Linear(cfg['neurons_layer2'], cfg['neurons_layer3']),
                self._get_activation(cfg['activation']),
                nn.Dropout(cfg['dropout_rate']),

                nn.Linear(cfg['neurons_layer3'], 1)
            )

        def _get_activation(self, activation):
            return nn.ReLU() if activation == 'relu' else nn.Sigmoid()

        def forward(self, x):
            return self.layers(x).squeeze()

    def fit(self, X, y):
        # Convert and scale data
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device)

        # Create model config
        model_cfg = {
            'neurons_layer1': self.neurons_layer1,
            'neurons_layer2': self.neurons_layer2,
            'neurons_layer3': self.neurons_layer3,
            'dropout_rate': self.dropout_rate,
            'activation': self.activation
        }

        # Initialize model
        self.model = self._ANN(X.shape[1], model_cfg).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        loss_fn = nn.MSELoss()

        # Early stopping setup
        best_loss = float('inf')
        patience_counter = 0

        # Training loop
        for epoch in range(self.epochs):
            self.model.train()
            epoch_loss = 0

            # Mini-batch training
            for i in range(0, len(X), self.batch_size):
                batch_X = X_tensor[i:i+self.batch_size]
                batch_y = y_tensor[i:i+self.batch_size]

                optimizer.zero_grad()
                predictions = self.model(batch_X)
                loss = loss_fn(predictions, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / (len(X) / self.batch_size)

            # Early stopping check
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.early_stopping_patience:
                    if self.verbose:
                        print(f"Early stopping at epoch {epoch}")
                    break

        return self

    def predict(self, X):
        X_tensor = torch.FloatTensor(X).to(self.device)

        self.model.eval()
        with torch.no_grad():
            return self.model(X_tensor).cpu().numpy()

class PricePredictionExperiment:
    def __init__(self, model_name, model_params, experiment_name,
                 data_paths=None, random_state=42, feature_set=None, embedding=None):
        """
        Initialize the price prediction experiment with fixed feature_set and embedding.
        
        Args:
            model_name (str): Name of the model ('XGBRegressor', 'RandomForestRegressor', or 'ANNRegressor')
            model_params (dict): Dictionary of model hyperparameters for Optuna
            experiment_name (str): Name for the MLflow experiment
            data_paths (dict): Paths to data files (default uses Colab paths)
            random_state (int): Random seed for reproducibility
            feature_set (str): Fixed feature set type ('text-only', 'categorical-only', or 'mixed')
            embedding (str): Fixed embedding type ('tfidf', 'w2v', or 'rubert')
        """
        self.model_name = model_name
        self.model_params = self._validate_model_params(model_params)
        self.experiment_name = experiment_name
        self.random_state = random_state
        self.feature_set = feature_set
        self.embedding = embedding
        self.best_params = None

        # Validate feature_set and embedding
        if feature_set not in ['text-only', 'categorical-only', 'mixed']:
            raise ValueError("feature_set must be one of: 'text-only', 'categorical-only', 'mixed'")
        if feature_set != 'categorical-only' and embedding not in ['tfidf', 'w2v', 'rubert']:
            raise ValueError("embedding must be one of: 'tfidf', 'w2v', 'rubert' when not using categorical-only")

        # Set default data paths if not provided
        self.data_paths = data_paths or {
            'full': '/content/drive/MyDrive/thesis/df_full_filtered_onehot.csv',
            'part': '/content/drive/MyDrive/thesis/df_part_filtered_onehot.csv'
        }

        # Model mapping
        self.model_classes = {
            'XGBRegressor': XGBRegressor,
            'RandomForestRegressor': RandomForestRegressor,
            'LinearSVR': LinearSVR,
            'ANNRegressor': ANNRegressor
        }

        if model_name not in self.model_classes:
            raise ValueError(f"Unsupported model: {model_name}. Supported models: {list(self.model_classes.keys())}")

        # Set random seeds
        np.random.seed(random_state)
        torch.manual_seed(random_state)

    def _validate_model_params(self, params):
        """Validate and adjust model parameters based on sklearn version"""
        validated_params = params.copy()

        # Handle max_features compatibility
        if 'model__max_features' in validated_params:
            if 'auto' in validated_params['model__max_features']:
                validated_params['model__max_features'] = [
                    x for x in validated_params['model__max_features'] if x != 'auto'
                ] + ['sqrt']  # Replace 'auto' with 'sqrt'

        return validated_params

    def load_and_merge_data(self, sample_size=None):
        """Load and prepare the data"""
        logger.info("Loading data...")
        df_full = pd.read_csv(self.data_paths['full']).set_index('id').head(10000)
        df_part = pd.read_csv(self.data_paths['part']).set_index('id').head(10000)

        if sample_size:
            df_full = df_full.head(sample_size)
            df_part = df_part.head(sample_size)

        # Split features
        X_num_cat = df_full.drop(columns=['price', 'description'])
        X_text_full = df_full['description']
        X_text_part = df_part['description']
        y = df_full['price']

        return X_num_cat, X_text_full, X_text_part, y

    def calculate_metrics(self, y_true, y_pred):
        """Calculate evaluation metrics"""
        metrics = {
            'r2': r2_score(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'smape': 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-8))
        }
        return metrics

    def create_text_transformer(self):
        """Create appropriate text transformer based on fixed embedding parameter"""
        if self.embedding == 'tfidf':
            return TfidfTransformer(
                max_features=self.best_params.get('tfidf__max_features', 300),
                ngram_range=(1, self.best_params.get('tfidf__ngram_range', 1)),
                min_df=self.best_params.get('tfidf__min_df', 5),
                max_df=self.best_params.get('tfidf__max_df', 0.85)
            )
        elif self.embedding == 'w2v':
            return Word2VecTransformer(
                vector_size=self.best_params.get('w2v__vector_size', 200),
                window=self.best_params.get('w2v__window', 10),
                min_count=self.best_params.get('w2v__min_count', 5),
                sg=self.best_params.get('w2v__sg', 0)
            )
        else:  # rubert
            return RuBertTiny2Embedder(
                max_length=self.best_params.get('bert__max_len', 256),
                batch_size=self.best_params.get('bert__batch_size', 64),
                num_epochs=self.best_params.get('bert__num_epochs', 3),
                learning_rate=self.best_params.get('bert__learning_rate', 3e-5),
                use_cv=False
            )

    def create_final_pipeline(self, params, X_num_cat):
        """Create pipeline with proper column handling using fixed feature_set and embedding"""
        # Create text transformer if needed
        text_transformer = None
        if self.feature_set != 'categorical-only':
            text_transformer = self.create_text_transformer()

        # Create feature transformers
        transformers = []
        if self.feature_set in ['categorical-only', 'mixed']:
            transformers.append(('num_cat', 'passthrough', list(X_num_cat.columns)))

        if self.feature_set in ['text-only', 'mixed']:
            transformers.append(('text', text_transformer, 'text'))

        # Create model with parameter processing
        model_class = self.model_classes[self.model_name]
        model_params = {k.replace('model__', ''): v for k, v in params.items()
                       if k.startswith('model__')}

        # Special handling for RandomForest parameters
        if self.model_name == 'RandomForestRegressor':
            if 'max_features' in model_params and model_params['max_features'] == 'auto':
                model_params['max_features'] = 'sqrt'

        # Create pipeline
        pipeline = Pipeline([
            ('features', ColumnTransformer(transformers)),
            ('model', model_class(**model_params))
        ])

        return pipeline

    def suggest_parameter(self, trial, param_name, param_values):
        """Suggest parameter value with proper type handling"""
        if isinstance(param_values, list):
            if all(isinstance(x, bool) for x in param_values):
                return trial.suggest_categorical(param_name, param_values)
            elif all(isinstance(x, int) for x in param_values):
                return trial.suggest_int(param_name, min(param_values), max(param_values))
            elif all(isinstance(x, float) for x in param_values):
                return trial.suggest_float(param_name, min(param_values), max(param_values))
            else:
                return trial.suggest_categorical(param_name, param_values)
        return param_values

    def objective(self, trial, X_train, X_text_full_train, X_text_part_train, y_train, X_num_cat):
        """Optimization objective with fixed feature_set and embedding"""
        params = {
            'feature_set': self.feature_set,
            'embedding': self.embedding
        }

        # Add model-specific parameters
        for param_name, param_values in self.model_params.items():
            params[param_name] = self.suggest_parameter(trial, param_name, param_values)

        # Add embedding parameters if needed
        if self.feature_set != 'categorical-only':
            params.update(self.suggest_embedding_params(trial))

        try:
            return self.run_cross_validation(trial, params, X_train,
                                           X_text_full_train, X_text_part_train,
                                           y_train, X_num_cat)
        except Exception as e:
            logger.error(f"Trial {trial.number} failed: {str(e)}")
            return float('-inf')

    def suggest_embedding_params(self, trial):
        """Suggest parameters for the fixed embedding method"""
        params = {}
        
        if self.embedding == 'tfidf':
            params.update({
                'tfidf__max_features': trial.suggest_int('tfidf__max_features', 200, 500),
                'tfidf__ngram_range': trial.suggest_int('tfidf__ngram_range', 1, 2),
                'tfidf__min_df': trial.suggest_int('tfidf__min_df', 5, 20),
                'tfidf__max_df': trial.suggest_float('tfidf__max_df', 0.7, 0.95)
            })
        elif self.embedding == 'w2v':
            params.update({
                'w2v__vector_size': trial.suggest_int('w2v__vector_size', 100, 300),
                'w2v__window': trial.suggest_int('w2v__window', 5, 15),
                'w2v__min_count': trial.suggest_int('w2v__min_count', 1, 10),
                'w2v__sg': trial.suggest_int('w2v__sg', 0, 1)
            })
        else:  # rubert
            params.update({
                'bert__learning_rate': trial.suggest_float('bert__learning_rate', 1e-5, 1e-4, log=True),
                'bert__num_epochs': trial.suggest_int('bert__num_epochs', 1, 5),
                'bert__batch_size': trial.suggest_categorical('bert__batch_size', [16, 64]),
                'bert__max_len': trial.suggest_int('bert__max_len', 256, 512)
            })

        return params

    def run_cross_validation(self, trial, params, X_train, X_text_full_train,
                            X_text_part_train, y_train, X_num_cat):
        """Run cross-validation for a trial with fixed feature_set and embedding"""
        # Prepare data based on fixed feature set
        X_trial = self.prepare_trial_data(X_train, X_text_full_train, X_text_part_train)

        kf = KFold(n_splits=5, shuffle=True, random_state=self.random_state)
        metrics = {'r2': [], 'rmse': [], 'smape': []}

        for fold, (train_idx, val_idx) in enumerate(kf.split(X_trial)):
            X_train_fold, X_val_fold = X_trial.iloc[train_idx], X_trial.iloc[val_idx]
            y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]

            fold_metrics = self.run_fold(params, X_num_cat, X_train_fold,
                                       y_train_fold, X_val_fold, y_val_fold)

            # Log fold metrics
            self.log_fold_metrics(trial, fold, params, fold_metrics)

            # Store metrics
            for k in metrics:
                metrics[k].append(fold_metrics[k])

        # Calculate and log average metrics
        avg_metrics = {k: np.mean(v) for k, v in metrics.items()}
        trial.set_user_attr("avg_metrics", avg_metrics)

        logger.info(
            f"Trial {trial.number} Averages - "
            f"R²: {avg_metrics['r2']:.4f}, "
            f"RMSE: {avg_metrics['rmse']:.4f}, "
            f"SMAPE: {avg_metrics['smape']:.4f}%"
        )

        return avg_metrics['r2']

    def prepare_trial_data(self, X_train, X_text_full_train, X_text_part_train):
        """Prepare data based on fixed feature_set and embedding"""
        if self.feature_set == 'text-only':
            return pd.DataFrame({
                'text': X_text_full_train if self.embedding in ['tfidf', 'w2v']
                      else X_text_part_train
            })
        elif self.feature_set == 'categorical-only':
            return X_train.copy()
        else:  # mixed
            X_trial = X_train.copy()
            X_trial['text'] = X_text_full_train if self.embedding in ['tfidf', 'w2v'] else X_text_part_train
            return X_trial

    def run_fold(self, params, X_num_cat, X_train_fold, y_train_fold, X_val_fold, y_val_fold):
        """Run a single fold of cross-validation"""
        pipeline = self.create_final_pipeline(params, X_num_cat)
        pipeline.fit(X_train_fold, y_train_fold)

        y_pred = pipeline.predict(X_val_fold)
        if np.isnan(y_pred).any():
            raise ValueError("NaN values in predictions")

        return self.calculate_metrics(y_val_fold, y_pred)

    def log_fold_metrics(self, trial, fold, params, metrics):
        """Log metrics for a single fold"""
        with mlflow.start_run(nested=True):
            mlflow.log_params(params)
            for metric_name, value in metrics.items():
                mlflow.log_metric(metric_name, value)
            mlflow.log_metric("fold", fold)

        logger.info(
            f"Trial {trial.number} Fold {fold} - "
            f"R²: {metrics['r2']:.4f}, "
            f"RMSE: {metrics['rmse']:.4f}, "
            f"SMAPE: {metrics['smape']:.4f}%"
        )

    def run(self, n_trials=50, sample_size=None):
        """Run the complete experiment with fixed feature_set and embedding"""
        mlflow.set_experiment(self.experiment_name)
        logger.info(f"Starting experiment: {self.experiment_name}")
        logger.info(f"Fixed parameters - feature_set: {self.feature_set}, embedding: {self.embedding}")

        # Load and split data
        X_num_cat, X_text_full, X_text_part, y = self.load_and_merge_data(sample_size)
        X_train, X_test, X_text_full_train, X_text_full_test, X_text_part_train, X_text_part_test, y_train, y_test = train_test_split(
            X_num_cat, X_text_full, X_text_part, y,
            test_size=0.2, random_state=self.random_state
        )

        # Setup Optuna study
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=self.random_state),
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=5)
        )

        with mlflow.start_run():
            self.log_experiment_metadata(X_train, X_test)
            study.optimize(
                lambda trial: self.objective(trial, X_train, X_text_full_train,
                                           X_text_part_train, y_train, X_num_cat),
                n_trials=n_trials,
                show_progress_bar=True
            )

            if study.best_trial is None:
                raise RuntimeError("No successful trials completed. Check logs for errors.")

            self.process_best_trial(study, X_num_cat, X_train, X_test,
                                   X_text_full_train, X_text_full_test,
                                   X_text_part_train, X_text_part_test,
                                   y_train, y_test)

        return self.best_params

    def log_experiment_metadata(self, X_train, X_test):
        """Log basic experiment metadata including fixed parameters"""
        mlflow.log_params({
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'model': self.model_name,
            'random_state': self.random_state,
            'fixed_feature_set': self.feature_set,
            'fixed_embedding': self.embedding
        })

    def process_best_trial(self, study, X_num_cat, X_train, X_test,
                          X_text_full_train, X_text_full_test,
                          X_text_part_train, X_text_part_test,
                          y_train, y_test):
        """Process and log the best trial results"""
        self.best_params = study.best_params
        mlflow.log_params(self.best_params)
        mlflow.log_metric("best_r2", study.best_value)

        logger.info("Training final model...")
        final_pipeline = self.train_final_model(X_num_cat, X_train, X_test,
                                              X_text_full_train, X_text_full_test,
                                              X_text_part_train, X_text_part_test,
                                              y_train, y_test)

        # Evaluate on test set
        X_test_final = self.prepare_test_data(X_test, X_text_full_test, X_text_part_test)
        y_pred = final_pipeline.predict(X_test_final)
        metrics = self.calculate_metrics(y_test, y_pred)

        logger.info(f"Final Test R²: {metrics['r2']:.4f}")
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(final_pipeline, "best_model")

    def train_final_model(self, X_num_cat, X_train, X_test,
                         X_text_full_train, X_text_full_test,
                         X_text_part_train, X_text_part_test,
                         y_train, y_test):
        """Train the final model on all available data with fixed feature_set and embedding"""
        final_pipeline = self.create_final_pipeline(self.best_params, X_num_cat)

        # Prepare final training data based on fixed feature_set
        if self.feature_set == 'text-only':
            X_final = pd.DataFrame({
                'text': pd.concat([X_text_full_train, X_text_full_test])
                if self.embedding in ['tfidf', 'w2v']
                else pd.concat([X_text_part_train, X_text_part_test])
            })
        elif self.feature_set == 'categorical-only':
            X_final = pd.concat([X_train, X_test])
        else:  # mixed
            X_final = pd.concat([X_train, X_test])
            X_final['text'] = (pd.concat([X_text_full_train, X_text_full_test])
                             if self.embedding in ['tfidf', 'w2v']
                             else pd.concat([X_text_part_train, X_text_part_test]))

        final_pipeline.fit(X_final, pd.concat([y_train, y_test]))
        return final_pipeline

    def prepare_test_data(self, X_test, X_text_full_test, X_text_part_test):
        """Prepare test data based on fixed feature_set and embedding"""
        if self.feature_set == 'text-only':
            return pd.DataFrame({
                'text': X_text_full_test if self.embedding in ['tfidf', 'w2v']
                         else X_text_part_test
            })
        elif self.feature_set == 'categorical-only':
            return X_test.copy()
        else:  # mixed
            X_test_final = X_test.copy()
            X_test_final['text'] = (X_text_full_test
                                   if self.embedding in ['tfidf', 'w2v']
                                   else X_text_part_test)
            return X_test_final
