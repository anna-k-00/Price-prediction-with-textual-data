import logging
import os
import numpy as np
import pandas as pd
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


# Import your custom transformers
# from content.drive.MyDrive.thesis.Transformers.tfidf_transformer import TfidfTransformer
# from content.drive.MyDrive.thesis.Transformers.w2v_ztransformer import  Word2VecTransformer
# from content.drive.MyDrive.thesis.Transformers.rubert_transformer import  RuBertTiny2Embedder
from tfidf_transformer import TfidfTransformer
from w2v_transformer import Word2VecTransformer
from rubert_transformer import RuBertTiny2Embedder


tuples = [('text-only','tfidf'),('text-only','w2v'),('text-only','rubert'),('mixed','tfidf'),('mixed','w2v'),('mixed','rubert'),('categorical-only','tfidf')]

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('experiment.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


for feaure_set,embedding in tuples:
  exp_name=f"XGBR_{feaure_set}_{embedding}_{random.randint(1_000_000, 9_999_999_999)}"
  if __name__ == "__main__":
    print(exp_name)
  # Define data paths
    data_paths = {
        'full': '/content/drive/MyDrive/thesis datasets/df_full_filtered.csv',
        'part': '/content/drive/MyDrive/thesis datasets/df_part_filtered.csv'
    }

    # Define hyperparameters for Random Forest
    xgbr_params = {
        'model__n_estimators': [500, 1000],
        'model__max_depth': [4, 6],
        'model__learning_rate': [0.005, 0.01, 0.02, 0.05],
        'model__subsample': [0.8, 1.0]
    }

    # Create and run experiment with fixed feature_set and embedding
    experiment = PricePredictionExperiment(
        model_name='XGBRegressor',
        data_paths=data_paths,
        model_params=xgbr_params,
        experiment_name=exp_name,
        feature_set=feaure_set,  # Fixed parameter
        embedding=embedding,    # Fixed parameter
        random_state=42
    )

    best_params = experiment.run(n_trials=50, sample_size=10000)
    logger.info("\nBest parameters found:")
    logger.info(best_params)

    try:

      client = MlflowClient()
      experiments = client.search_experiments()  # This is the correct method name

      for exp in experiments:
        if exp.name == exp_name:
          experiment_id = exp.experiment_id
          break

      runs = client.search_runs(experiment_id)

      # Prepare data for DataFrame
      data = []
      for run in runs:
          row = {
              'run_id': run.info.run_id,
              'status': run.info.status,
              'start_time': pd.to_datetime(run.info.start_time, unit='ms'),
              **run.data.params,  # Unpack all parameters
              **run.data.metrics   # Unpack all metrics
          }
          data.append(row)

      # Create DataFrame
      results_df = pd.DataFrame(data)
      results_df[results_df.status == 'FINISHED' ].to_csv(f'/content/drive/MyDrive/thesis datasets/Results/{exp_name}.csv')
      latest_run = runs[0]

      # Get metrics
      final_metrics = latest_run.data.metrics
      print("Final Model Performance:")
      print(f"R²: {final_metrics['r2']:.4f}")
      print(f"RMSE: {final_metrics['rmse']:.4f}")
      print(f"SMAPE: {final_metrics['smape']:.4f}%")

      # Get parameters
      final_params = latest_run.data.params
      print("\nModel Parameters:")
      for k, v in final_params.items():
          print(f"{k}: {v}")
    except Exception:
      continue
    print('-'*50)
