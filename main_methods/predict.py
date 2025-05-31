import logging
import os
import time
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import optuna
from optuna.samplers import TPESampler
from tqdm import tqdm
import mlflow
import mlflow.sklearn
import warnings
from transformers import AutoModel, AutoTokenizer, AdamW
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
from sklearn.svm import LinearSVR
from sklearn.base import BaseEstimator, TransformerMixin
import random
from mlflow.tracking import MlflowClient
from datetime import datetime
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.sparse import issparse
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.base import BaseEstimator, TransformerMixin
import torch
import torch.nn as nn
import os
import joblib
import transformers
import json
import joblib
from datetime import datetime
from sklearn.decomposition import PCA
import time
import sys
import platform
import psutil
import threading

try:
    import pynvml
    pynvml.nvmlInit()
    HAS_NVML = True
except Exception:
    HAS_NVML = False


import os
import tempfile
import pandas as pd
import numpy as np
from mlflow.tracking import MlflowClient
from typing import Optional, Dict, Any, Union, List

class PricePredictorMLflow:
    """
    A comprehensive price prediction model wrapper that integrates with MLflow for model management,
    providing prediction capabilities, evaluation metrics, and model comparison functionality.
    """
    
    def __init__(
        self,
        experiment_name: str,
        param_filters: Optional[Dict[str, Any]],
        model_class,
        local_dir: Optional[str] = None,
        use_run_id: bool = False,
        run_id: Optional[str] = None,
        verbose: bool = True
    ):
        """
        Initialize the PricePredictorMLflow by loading a trained model from MLflow.
        
        Args:
            experiment_name: Name of the MLflow experiment containing the model
            param_filters: Dictionary of parameters to filter model runs
            model_class: The model class used for loading the trained model
            local_dir: Local directory for caching artifacts
            use_run_id: Whether to load a specific run by ID
            run_id: Specific MLflow run ID to load
            verbose: Whether to print progress information
        """
        self.client = MlflowClient()
        self.verbose = verbose
        self.manual_text_params_path = None  # Initialize as None

        # 1) Load runinfo and bundle
        if use_run_id:
            if not run_id:
                raise ValueError("If use_run_id=True, you must provide run_id")
            self.run_id = run_id
            bundle = model_class.load_everything_from_mlflow(run_id)
            self.params = bundle.get('params', {})

        else:
            bundle, runinfo = model_class.find_and_load_full_model(
                experiment_name,
                param_filters or {},
                local_dir,
                return_runinfo=True
            )
            self.run_id = runinfo['run_id']
            self.params = runinfo['params']
        # 2) Extract components
        self.pipeline = bundle.get('pipeline')
        self.transformer = bundle.get('transformer')
        self.tokenizer = bundle.get('tokenizer')
        self.meta = bundle.get('preprocessing', {}) or {}
        self.scaler = self.meta.get('scaler')
        self.lat_long_scaler = self.meta.get('lat_long_scaler')
        self.outlier_bounds = self.meta.get('outlier_bounds')
        self.hex_stats = self.meta.get('hex_stats')
        self.global_median_ppsm = self.meta.get('global_median_ppsm')
        self.global_median_ppland = self.meta.get('global_median_ppland')
        self.global_median_price = self.meta.get('global_median_price')
        self.feature_set = self.params.get('feature_set', None)
        self.embedding = self.params.get('embedding')
        self.manual_text_params = self.params.get('manual_text_params', False)

        self.params = self._convert_params_types(runinfo['params'] if not use_run_id else bundle.get('params', {}))

        print(f"Loaded manual_text_params from params: {self.params.get('manual_text_params')}")
        print(f"Converted to bool: {self.manual_text_params}")


        # 3) Only download manual text features if they exist and are needed
        if self.manual_text_params:
            try:
                tmp_dir = local_dir or tempfile.mkdtemp(prefix="manual_text_")
                local_dir_art = self.client.download_artifacts(
                    run_id=self.run_id,
                    path="manual_text_features",
                    dst_path=tmp_dir
                )
                # Find the JSON file
                for fn in os.listdir(local_dir_art):
                    if fn.endswith(".json"):
                        self.manual_text_params_path = os.path.join(local_dir_art, fn)
                        if self.verbose:
                            print(f"[INFO] Found manual text features at: {self.manual_text_params_path}")
                        break
                else:
                    if self.verbose:
                        print("[WARNING] Manual text params enabled but no JSON file found in artifacts")
            except Exception as e:
                if self.verbose:
                    print(f"[WARNING] Failed to download manual text features: {str(e)}")
                self.manual_text_params_path = None

        if self.manual_text_params == 'True':
            self.manual_text_params = True
        else:
            self.manual_text_params = False

        if self.verbose:
            print(f"[INFO] Model loaded successfully - feature_set: {self.feature_set}, "
                  f"embedding: {self.embedding}, "
                  f"manual_text_params: {self.manual_text_params}")

    def _convert_params_types(self, params: Dict[str, str]) -> Dict[str, Any]:
        """
        Internal method to convert parameter types from strings to appropriate Python types.
        
        Args:
            params: Dictionary of parameters with string values
            
        Returns:
            Dictionary with converted parameter values
        """
        converted = {}
        for key, value in params.items():
            if value is None or value == 'None':
                converted[key] = None
            elif value == 'True':
                converted[key] = True
            elif value == 'False':
                converted[key] = False
            else:
                try:
                    converted[key] = int(value)
                except ValueError:
                    try:
                        converted[key] = float(value)
                    except ValueError:
                        converted[key] = value
        return converted


    def _preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Perform base and full preprocessing using DataProcessingPipeline.
        
        Args:
            df: Input DataFrame to preprocess
            
        Returns:
            Preprocessed DataFrame ready for model input
        """
        # Step 1 - basic transformations

        p1 = DataProcessingPipeline(
            df.copy(),
            log_needed=self.params.get("log_needed", True),
            norm_needed=self.params.get("norm_needed", True),
            one_hot_only=self.params.get("one_hot_only", True),
            use_hex_features=self.params.get("use_hex_features", True),
            hex_resolution=self.params.get("hex_resolution", 10),
            scaler=self.scaler,
            lat_long_scaler=self.lat_long_scaler,
            outlier_bounds=self.outlier_bounds,
            train=False,
            manual_text_params=self.manual_text_params,
            manual_text_params_path=self.manual_text_params_path
        )

        df_base = p1.preprocess_base()

        # Step 2 - full preparation for model
        p2 = DataProcessingPipeline(
            df_base,
            log_needed=self.params.get("log_needed", True),
            norm_needed=self.params.get("norm_needed", True),
            one_hot_only=self.params.get("one_hot_only", True),
            use_hex_features=self.params.get("use_hex_features", True),
            hex_resolution=self.params.get("hex_resolution", 10),
            scaler=self.scaler,
            lat_long_scaler=self.lat_long_scaler,
            outlier_bounds=self.outlier_bounds,
            hex_stats=self.hex_stats,
            global_median_ppsm=self.global_median_ppsm,
            global_median_ppland=self.global_median_ppland,
            global_median_price=self.global_median_price,
            train=False
        )
        return p2.prepare_for_model()

    def _prepare_model_input(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare data for prediction with proper column ordering and text handling.
        
        Args:
            df: DataFrame to prepare for model input
            
        Returns:
            DataFrame properly formatted for model prediction
        """
        try:
            # 1. Prepare base DataFrame with text conversion
            if self.feature_set == 'text-only':
                text_col = 'description_raw' if self.embedding == 'rubert' else 'description'
                X = pd.DataFrame({
                    'text': df[text_col].astype(str)  # Force conversion to string
                })
                return X  # For text-only, no need for column ordering

            elif self.feature_set == 'categorical-only':
                drop_cols = [c for c in ['description', 'description_raw', 'text'] if c in df.columns]
                X = df.drop(columns=drop_cols)
            else:  # mixed features
                X = df.copy()
                text_col = 'description_raw' if self.embedding == 'rubert' else 'description'
                X['text'] = df[text_col].astype(str)  # Force conversion to string

            # 2. Handle missing values in text columns
            if 'text' in X.columns:
                X['text'] = X['text'].fillna('')  # Replace NaN with empty string

            # 3. Get the correct column order from the pipeline
            if self.pipeline and 'features' in self.pipeline.named_steps:
                # Get the transformer names and columns from the ColumnTransformer
                ct = self.pipeline.named_steps['features']

                # Get the expected column order from the transformers
                expected_columns = []
                for name, transformer, columns in ct.transformers_:
                    if name == 'text':
                        # For text column, we just need to ensure it exists
                        if 'text' not in X.columns:
                            X['text'] = ''
                        expected_columns.append('text')
                    else:
                        # For other transformers, add all their columns
                        if isinstance(columns, list):
                            expected_columns.extend(columns)
                        else:
                            expected_columns.append(columns)

                # Ensure all expected columns exist
                missing_cols = set(expected_columns) - set(X.columns)
                for col in missing_cols:
                    X[col] = 0  # Fill missing with 0

                # Remove extra columns and order as expected
                X = X[expected_columns]

            self.last_prepared_input = X.copy()
    #       self.last_preparation_success = True

            return X

        except Exception as e:
            if self.verbose:
                print(f"[ERROR] Data preparation failed: {str(e)}")
            raise

    def _reverse_transform(self, preds: np.ndarray) -> np.ndarray:
        """
        Reverse normalization and log transformation of predictions.
        
        Args:
            preds: Array of model predictions
            
        Returns:
            Array of predictions in original scale
        """
        arr = preds.copy()
        if self.meta.get('norm_needed', True) and self.scaler is not None:
            tmp = np.zeros((len(arr), self.scaler.n_features_in_))
            tmp[:, -1] = arr
            arr = self.scaler.inverse_transform(tmp)[:, -1]
        if self.meta.get('log_needed', True):
            arr = np.expm1(arr)
        return arr

    def predict(
        self,
        raw: Union[str, pd.DataFrame],
        max_rows: Optional[int] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Make predictions on new data.
        
        Args:
            raw: Either a path to CSV file or a DataFrame
            max_rows: Maximum number of rows to process
            
        Returns:
            Dictionary containing:
            - 'result': Preprocessed data with predicted_price
            - 'predictions': Original columns plus predicted_price column
        """
        df = pd.read_csv(raw) if isinstance(raw, str) else raw.copy()
        if max_rows:
            df = df.head(max_rows)

        df_pre = self._preprocess(df)
        X = self._prepare_model_input(df_pre)
        y_pred = self.pipeline.predict(X)
        y_final = self._reverse_transform(y_pred)

        df_out = df_pre.reset_index(drop=False).rename(columns={'index':'_orig_idx'})
        df_out['predicted_price'] = y_final

        preds = df.merge(
            df_out[['id','predicted_price']],
            on='id', how='left', validate='1:1'
        )
        return {'result': df_out, 'predictions': preds}

    def analyze_text(self, texts: List[str], top_n: int = 20) -> Any:
        """
        Get token/phrase importance from text transformer (if supported).
        
        Args:
            texts: List of text strings to analyze
            top_n: Number of top features to return
            
        Returns:
            Feature importance analysis if supported by transformer
        """
        if hasattr(self.transformer, 'get_feature_importance'):
            return self.transformer.get_feature_importance(top_n=top_n)
        raise ValueError("Current transformer doesn't support text analysis")

    def evaluate_metrics(self, predictions_df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate prediction quality metrics including R², RMSE, SMAPE, and MedAPE.
        
        Args:
            predictions_df: DataFrame with 'price' and 'predicted_price' columns
            
        Returns:
            Dictionary of calculated metrics
        """
        y_true = predictions_df['price'].values
        y_pred = predictions_df['predicted_price'].values

        # Remove rows with missing values
        mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
        y_true = y_true[mask]
        y_pred = y_pred[mask]

        if len(y_true) == 0:
            raise ValueError("No data for evaluation (all price or predicted_price values missing)")

        metrics = {}

        # R²
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        metrics['r2'] = 1 - (ss_res / ss_tot) if ss_tot != 0 else float('nan')

        # RMSE
        metrics['rmse'] = np.sqrt(np.mean((y_true - y_pred) ** 2))

        # SMAPE
        denominator = (np.abs(y_true) + np.abs(y_pred))
        smape = np.where(denominator == 0, 0, 200 * np.abs(y_pred - y_true) / denominator)
        metrics['smape'] = np.mean(smape)

        # MedAPE
        ape = np.where(y_true == 0, np.abs(y_pred - y_true), 100 * np.abs(y_pred - y_true) / y_true)
        metrics['medape'] = np.median(ape)

        return metrics

    def evaluate_metrics_bootstrap(
        self,
        predictions_df: pd.DataFrame,
        n_iterations: int = 10000,
        confidence_level: float = 0.95,
        random_seed: Optional[int] = None,
        metrics_to_calculate: List[str] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate metrics with bootstrap confidence intervals for robust estimation.
        
        Args:
            predictions_df: DataFrame with 'price' and 'predicted_price' columns
            n_iterations: Number of bootstrap iterations
            confidence_level: Confidence level for intervals
            random_seed: Optional random seed for reproducibility
            metrics_to_calculate: List of metrics to compute
            
        Returns:
            Dictionary with metrics and their confidence intervals
        """
        # Set random seed if provided
        if random_seed is not None:
            np.random.seed(random_seed)

        # Get original data
        y_true = predictions_df['price'].values
        y_pred = predictions_df['predicted_price'].values

        # Remove rows with missing values
        mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
        y_true = y_true[mask]
        y_pred = y_pred[mask]

        if len(y_true) == 0:
            raise ValueError("No valid data for evaluation (all price or predicted_price values are missing)")

        n_samples = len(y_true)

        # Default metrics to calculate
        if metrics_to_calculate is None:
            metrics_to_calculate = ['r2', 'rmse', 'smape', 'medape']

        bootstrap_metrics = {metric: [] for metric in metrics_to_calculate}

        # Progress bar with tqdm
        pbar = tqdm(range(n_iterations), desc="Bootstrapping", disable=not self.verbose)

        # Bootstrap iterations
        for _ in pbar:
            # Resample with replacement
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            y_true_boot = y_true[indices]
            y_pred_boot = y_pred[indices]

            # Calculate metrics
            metrics = {}

            if 'r2' in metrics_to_calculate:
                ss_res = np.sum((y_true_boot - y_pred_boot) ** 2)
                ss_tot = np.sum((y_true_boot - np.mean(y_true_boot)) ** 2)
                metrics['r2'] = 1 - (ss_res / ss_tot) if ss_tot != 0 else float('nan')

            if 'rmse' in metrics_to_calculate:
                metrics['rmse'] = np.sqrt(np.mean((y_true_boot - y_pred_boot) ** 2))

            if 'smape' in metrics_to_calculate:
                denominator = (np.abs(y_true_boot) + np.abs(y_pred_boot))
                smape = np.where(denominator == 0, 0, 200 * np.abs(y_pred_boot - y_true_boot) / denominator)
                metrics['smape'] = np.mean(smape)

            if 'medape' in metrics_to_calculate:
                ape = np.where(y_true_boot == 0,
                            np.abs(y_pred_boot - y_true_boot),
                            100 * np.abs(y_pred_boot - y_true_boot) / y_true_boot)
                metrics['medape'] = np.median(ape)

            # Store metrics
            for metric in metrics_to_calculate:
                bootstrap_metrics[metric].append(metrics.get(metric))

        # Calculate confidence intervals
        alpha = (1 - confidence_level) / 2
        results = {}

        for metric_name, values in bootstrap_metrics.items():
            values = np.array(values)
            values = values[~np.isnan(values)]  # Remove NaN values

            if len(values) == 0:
                results[metric_name] = {
                    'mean': float('nan'),
                    'lower': float('nan'),
                    'upper': float('nan'),
                    'std': float('nan')
                }
                continue

            lower = np.percentile(values, 100 * alpha)
            upper = np.percentile(values, 100 * (1 - alpha))

            results[metric_name] = {
                'mean': np.mean(values),
                'lower': lower,
                'upper': upper,
                'std': np.std(values),
                'n_valid_samples': len(values)
            }

        return results

    def compare_models_bootstrap(
        self,
        other: Union['PricePredictorMLflow', pd.DataFrame, str],
        ground_truth_df: pd.DataFrame,
        other_pred_col: str = 'predicted_price',
        n_iterations: int = 10000,
        confidence_level: float = 0.95,
        random_seed: Optional[int] = None,
        metrics_to_compare: List[str] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare two models using bootstrap with flexible input types.
        
        Args:
            other: Either PricePredictorMLflow instance, DataFrame with predictions, or path to CSV
            ground_truth_df: DataFrame with 'price' and 'id' columns (actual values)
            other_pred_col: Column name with predictions if 'other' is DataFrame
            n_iterations: Number of bootstrap iterations
            confidence_level: Confidence level for intervals
            random_seed: Optional random seed
            metrics_to_compare: List of metrics to compare
            
        Returns:
            Dictionary with metric differences and confidence intervals
        """
        # Get predictions from current model
        preds1 = self.predict(ground_truth_df)['predictions']

        # Get predictions from the other model/source
        if isinstance(other, PricePredictorMLflow):
            preds2 = other.predict(ground_truth_df)['predictions']
            other_pred_col = 'predicted_price'  # Standard column name
        elif isinstance(other, pd.DataFrame):
            preds2 = other
        elif isinstance(other, str):
            try:
                preds2 = pd.read_csv(other)
            except Exception as e:
                raise ValueError(f"Failed to read predictions from {other}: {str(e)}")
        else:
            raise TypeError("'other' must be either PricePredictorMLflow, DataFrame, or path string")

        # Validate prediction DataFrame
        required_cols = ['id', other_pred_col]
        missing_cols = [c for c in required_cols if c not in preds2.columns]
        if missing_cols:
            raise ValueError(f"Prediction DataFrame missing required columns: {missing_cols}")

        # Align predictions
        merged = preds1[['id', 'predicted_price']].merge(
            preds2[['id', other_pred_col]].rename(columns={other_pred_col: 'predicted_price_model2'}),
            on='id',
            how='inner'  # Only compare samples present in both
        ).merge(
            ground_truth_df[['id', 'price']],
            on='id',
            how='inner'
        )

        # Extract values
        y_true = merged['price'].values
        y_pred1 = merged['predicted_price'].values
        y_pred2 = merged['predicted_price_model2'].values

        # Remove missing values
        mask = (~np.isnan(y_true) & ~np.isnan(y_pred1) & ~np.isnan(y_pred2))
        y_true = y_true[mask]
        y_pred1 = y_pred1[mask]
        y_pred2 = y_pred2[mask]

        if len(y_true) == 0:
            raise ValueError("No valid overlapping samples for comparison")

        n_samples = len(y_true)

        # Default metrics to compare
        if metrics_to_compare is None:
            metrics_to_compare = ['r2', 'rmse', 'smape', 'medape']

        bootstrap_diffs = {f"{metric}_diff": [] for metric in metrics_to_compare}

        # Set random seed if provided
        if random_seed is not None:
            np.random.seed(random_seed)

        # Progress bar
        pbar = tqdm(range(n_iterations), desc="Comparing models", disable=not self.verbose)

        # Bootstrap iterations
        for _ in pbar:
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            y_true_boot = y_true[indices]
            y_pred1_boot = y_pred1[indices]
            y_pred2_boot = y_pred2[indices]

            # Calculate metrics for both models
            def calculate_metrics(y_true, y_pred):
                metrics = {}

                if 'r2' in metrics_to_compare:
                    ss_res = np.sum((y_true - y_pred) ** 2)
                    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
                    metrics['r2'] = 1 - (ss_res / ss_tot) if ss_tot != 0 else float('nan')

                if 'rmse' in metrics_to_compare:
                    metrics['rmse'] = np.sqrt(np.mean((y_true - y_pred) ** 2))

                if 'smape' in metrics_to_compare:
                    denominator = (np.abs(y_true) + np.abs(y_pred))
                    smape = np.where(denominator == 0, 0, 200 * np.abs(y_pred - y_true) / denominator)
                    metrics['smape'] = np.mean(smape)

                if 'medape' in metrics_to_compare:
                    ape = np.where(y_true == 0,
                                np.abs(y_pred - y_true),
                                100 * np.abs(y_pred - y_true) / y_true)
                    metrics['medape'] = np.median(ape)

                return metrics

            metrics1 = calculate_metrics(y_true_boot, y_pred1_boot)
            metrics2 = calculate_metrics(y_true_boot, y_pred2_boot)

            # Store differences
            for metric in metrics_to_compare:
                diff = metrics1.get(metric, float('nan')) - metrics2.get(metric, float('nan'))
                bootstrap_diffs[f"{metric}_diff"].append(diff)

        # Calculate confidence intervals and p-values
        alpha = (1 - confidence_level) / 2
        results = {}

        for metric_name, diffs in bootstrap_diffs.items():
            diffs = np.array(diffs)
            diffs = diffs[~np.isnan(diffs)]  # Remove NaN values

            if len(diffs) == 0:
                results[metric_name] = {
                    'mean': float('nan'),
                    'lower': float('nan'),
                    'upper': float('nan'),
                    'p_value': float('nan'),
                    'std': float('nan'),
                    'n_valid_samples': 0
                }
                continue

            mean_diff = np.mean(diffs)
            lower = np.percentile(diffs, 100 * alpha)
            upper = np.percentile(diffs, 100 * (1 - alpha))

            # Calculate p-value (proportion of differences crossing zero)
            if 'r2_diff' in metric_name or 'smape_diff' in metric_name or 'medape_diff' in metric_name:
                # For these metrics, positive difference means model1 is better
                p_value = np.mean(np.array(diffs) <= 0)
            elif 'rmse_diff' in metric_name:
                # For RMSE, negative difference means model1 is better
                p_value = np.mean(np.array(diffs) >= 0)
            else:
                p_value = float('nan')

            results[metric_name] = {
                'mean': mean_diff,
                'lower': lower,
                'upper': upper,
                'p_value': p_value,
                'std': np.std(diffs),
                'n_valid_samples': len(diffs)
            }

        return results
