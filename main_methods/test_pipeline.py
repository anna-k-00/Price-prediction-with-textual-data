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
from resource_monitor import ResourceMonitor


try:
    import pynvml
    pynvml.nvmlInit()
    HAS_NVML = True
except Exception:
    HAS_NVML = False


from typing import Optional
from typing import Dict, List


class PricePredictionExperiment:
    def __init__(self, model_name, model_params, experiment_name,
                data_path=None, random_state=42, results_dir=None,
                verbose=True, use_hex_features=False, hex_resolution=7,
                use_pca=None, pca_components=100, use_gate=None,
                manual_text_params: bool = False,
                manual_text_params_path: Optional[str] = None,
                test_path: Optional[str] = None,
                profile_json_path: Optional[str] = None):
        self.verbose = verbose
        self.model_name = model_name
        self.model_params = model_params
        self.experiment_name = experiment_name
        self.random_state = random_state
        self.results_dir = results_dir or 'results'
        os.makedirs(self.results_dir, exist_ok=True)

# Note: this section is described uniformly.
        self.manual_text_params = manual_text_params
        self.manual_text_params_path = manual_text_params_path
        if self.manual_text_params:
            if not self.manual_text_params_path:
                raise ValueError("manual_text_params_path must be provided when manual_text_params=True")
            import json

            with open(self.manual_text_params_path, 'r', encoding='utf-8') as f:
                self.manual_text_features_dict = json.load(f)
        else:
            self.manual_text_features_dict = {}

# Note: this section is described uniformly.
        self.best_params = None
        self.fixed_feature_set = None
        self.fixed_embedding = None
        self.data_path = data_path or 'df_full_filtered_onehot.csv'
        self.use_hex_features = use_hex_features
        self.hex_resolution = hex_resolution
        self.use_pca = use_pca
        self.pca_components = pca_components
        self.use_gate = use_gate
        self.profile_json_path = profile_json_path
        self.load_ = False  # Note: this section is described uniformly.
        self.test_path = test_path

        self.model_classes = {
            'ANNRegressor': ANNRegressor,
            'XGBRegressor': XGBRegressor,
            'RandomForestRegressor': RandomForestRegressor,
            'LinearSVR': LinearSVR
        }
        if model_name not in self.model_classes:
            raise ValueError(f"Unsupported model: {model_name}")

        self._setup_logging()
        self.resource_monitor = ResourceMonitor(logger=self.logger, interval=30)
        np.random.seed(random_state)
        random.seed(random_state)
        torch.manual_seed(random_state)

    def _check_tag(self, parsed_word, tags):
        """Проверяет, содержит ли parsed_word любой из указанных тегов"""
        for tag in tags:
            if tag in str(parsed_word.tag):
                return True
        return False


    @staticmethod
    def stratified_subsample(df, target_col, size, n_bins=10, random_state=42):
        """
        Возвращает стратифицированную подвыборку size из df по binned target_col.
        Индексы исходные, порядок не меняется.
        Не оставляет никаких временных колонок в df!
        """
        df = df.copy()
        bins = pd.qcut(df[target_col], q=n_bins, duplicates='drop')
        df['__bins__'] = bins.cat.codes
# Note: this section is described uniformly.
        stratified = (
            df[df['__bins__'] != -1]
            .groupby('__bins__', group_keys=False, observed=True)
            .apply(lambda x: x.sample(int(np.ceil(size / n_bins)), random_state=random_state) if len(x) > 0 else x)
        )
        stratified = stratified.iloc[:size]
        stratified = stratified.drop(columns=['__bins__'])
        return stratified

    def _make_run_name(self) -> str:
        """
        Возвращает run_name для mlflow.start_run на основе:
        {experiment_name}_{model_name}_{feature_set}_{embedding}_pca_{use_pca}_gate_{use_gate}
        """
        return (
            f"{self.experiment_name}_{self.model_name}_"
            f"{self.fixed_feature_set}_{self.fixed_embedding or 'none'}_"
            f"pca_{self.use_pca}_gate_{self.use_gate}_mtf_{self.manual_text_params}_"
        )

    def _log_cv_stats_to_mlflow(self, cv_metrics):
        """
        Логирует в MLflow все summary-статистики финальной CV:
        mean, std, conf_interval, min, max для r2, rmse, smape, medape, отдельно по log/orig.
        """
        for scale in ['log', 'orig']:
            for metric in ['r2', 'rmse', 'smape', 'medape']:
                m = cv_metrics[f"{scale}_metrics"][metric]
                mlflow.log_metric(f"cv_{scale}_{metric}_mean", m['mean'])
                mlflow.log_metric(f"cv_{scale}_{metric}_std", m['std'])
                mlflow.log_metric(f"cv_{scale}_{metric}_conf_interval", m['conf_interval'])
                mlflow.log_metric(f"cv_{scale}_{metric}_min", m['min'])
                mlflow.log_metric(f"cv_{scale}_{metric}_max", m['max'])

    def _suggest_parameter(self, trial, param_name, param_values):
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

    def _log_time_start(self, label='total'):
        if not hasattr(self, '_time_stamps'):
            self._time_stamps = {}
        self._time_stamps[label] = time.time()
        self.logger.info(f"Timing started for: {label}")

    def _log_time_end(self, label='total', log_to_mlflow=True):
        if hasattr(self, '_time_stamps') and label in self._time_stamps:
            elapsed = time.time() - self._time_stamps[label]
            self.logger.info(f"Elapsed time for {label}: {elapsed:.2f} seconds")
            if log_to_mlflow:
                mlflow.log_metric(f"time_{label}_seconds", elapsed)
            return elapsed
        else:
            self.logger.warning(f"Timing for {label} not started.")
            return None

    def _log_environment(self):
        env = {
            "python_version": sys.version,
            "platform": platform.platform(),
            "sklearn_version": sklearn.__version__,
            "pandas_version": pd.__version__,
            "numpy_version": np.__version__,
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
            "run_time": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()),
            "run_hostname": platform.node()
        }
        for k, v in env.items():
            mlflow.log_param(f'env_{k}', v)
            self.logger.info(f"ENV: {k}: {v}")

        mlflow.set_tag("python_version", sys.version.split()[0])
        mlflow.set_tag("sklearn_version", sklearn.__version__)
        mlflow.set_tag("pandas_version", pd.__version__)
        mlflow.set_tag("numpy_version", np.__version__)
        mlflow.set_tag("torch_version", torch.__version__)
        mlflow.set_tag("cuda", str(torch.cuda.is_available()))
        mlflow.set_tag("hostname", platform.node())
        mlflow.set_tag("run_time", time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))

        return env


        HAS_NVML = False

    def validate_strict(self, df):
        """
        Проверяет df на соответствие профилю, сохранённому в self.profile_json_path.
        Вызывает ValueError/TypeError с понятной ошибкой при любом несоответствии.
        """
        import json
        profile_path = self.profile_json_path
        if not profile_path:
            self.logger.warning("profile_json_path не задан — пропуск валидации данных.")
            return
        with open(profile_path, 'r', encoding='utf-8') as f:
            profile = json.load(f)

        object_uniques = profile['object_uniques']
        dtypes_ref = profile['dtypes']

# Note: this section is described uniformly.
        for col, dtype in dtypes_ref.items():
            if col not in df.columns:
                raise ValueError(f"Критическая ошибка: колонка '{col}' отсутствует в новых данных!")
            actual = str(df[col].dtype)
            if actual != dtype:
                raise TypeError(f"Критическая ошибка: для '{col}' ожидался тип '{dtype}', получен '{actual}'")

# Note: this section is described uniformly.
        for col, allowed_values in object_uniques.items():
            if col not in df.columns:
                raise ValueError(f"Критическая ошибка: колонка '{col}' отсутствует в новых данных!")
            actual_uniques = set(df[col].dropna().unique())
            allowed_set = set(allowed_values)
            new_unexpected = actual_uniques - allowed_set
            if new_unexpected:
                raise ValueError(f"Критическая ошибка: в колонке '{col}' найдены неожиданные значения: {new_unexpected}")

# Note: this section is described uniformly.
        required_cols = ['id', 'Площадь участка', 'Площадь дома',
                        'price', 'category', 'lat', 'lon']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Критическая ошибка: обязательная колонка '{col}' отсутствует в новых данных!")
            n_empty = df[col].isnull().sum()
            if n_empty > 0:
                raise ValueError(f"Критическая ошибка: в колонке '{col}' обнаружено {n_empty} пустых значений!")

        self.logger.info("Данные прошли строгую валидацию профиля.")


    def _suggest_embedding_params(self, trial):
        params = {}
        if self.fixed_feature_set == 'categorical-only':
            params['use_gate'] = False
            return params

        if self.fixed_embedding == 'tfidf':
            params.update({
                'tfidf__max_features': trial.suggest_int('tfidf__max_features', 300, 500),
                'tfidf__ngram_range_min': 1,
                'tfidf__ngram_range_max': trial.suggest_int('tfidf__ngram_range_max', 1, 3),
                'tfidf__min_df': trial.suggest_int('tfidf__min_df', 1, 10),
                'tfidf__max_df': trial.suggest_float('tfidf__max_df', 0.7, 1.0)
            })
        elif self.fixed_embedding == 'w2v':
            params.update({
                'w2v__vector_size': trial.suggest_int('w2v__vector_size', 100, 300, step=50),
                'w2v__window': trial.suggest_int('w2v__window', 5, 15),
                'w2v__min_count': trial.suggest_int('w2v__min_count', 1, 10),
                'w2v__sg': trial.suggest_categorical('w2v__sg', [0, 1])
            })
        elif self.fixed_embedding == 'rubert':
            params.update({
                'bert__max_len': trial.suggest_int('bert__max_len', 256, 512, step=128),
                'bert__batch_size': trial.suggest_categorical('bert__batch_size', [8, 16]),
                'bert__pooling_type': trial.suggest_categorical('bert__pooling_type', ["cls", "mean", "max", "weighted"]),
                'bert__lr': trial.suggest_float('bert__lr', 1e-5, 6e-5, log=True)
            })

# Note: this section is described uniformly.
        params['use_gate'] = self.use_gate
        if self.use_gate:
            params['gate__threshold'] = 0.5
            params['gate__mode'] = 'adaptive'
            params['gate__epochs'] = 20
            params['gate__lr'] = 0.0005
            params['gate__l1_reg'] = 1e-5

        return params

    def _create_text_transformer(self, params):
        """Создание текстового трансформера с учетом параметров"""
        base_transformer = self._create_base_transformer(params)

# Note: this section is described uniformly.
        if params.get('feature_set', self.fixed_feature_set) == 'categorical-only':
            return base_transformer

# Note: this section is described uniformly.
        if params.get('use_gate', False) and params.get('embedding') == 'rubert':
            return GatedTransformerWithTokenImportance(
                text_transformer=base_transformer,
                gate_threshold=params.get('gate__threshold', 0.5),
                use_gate=True,
                hidden_dim=256,
                ngram_range=(
                    params.get('tfidf__ngram_range_min', 1),
                    params.get('tfidf__ngram_range_max', 1)
                ),
                tokenizer_params={
                    'max_length': params.get('bert__max_len', 512),
                    'truncation': True,
                    'padding': 'max_length'
                },
                bert_aggregation='mean',
                gate_mode=params.get('gate__mode', 'adaptive'),
                gate_epochs=params.get('gate__epochs', 20),
                gate_lr=params.get('gate__lr', 0.0005),
                l1_reg=params.get('gate__l1_reg', 1e-5),
                device='cuda' if torch.cuda.is_available() else 'cpu'
            )

        return base_transformer


    def _create_base_transformer(self, params):
        emb = params['embedding']
        use_pca = params.get('use_pca', False)
        pca_components = params.get('pca_components', 100)

        if emb == 'tfidf':
            base = TfidfTransformer(
                max_features=params.get('tfidf__max_features', 300),
                ngram_range=(
                    params.get('tfidf__ngram_range_min', 1),
                    params.get('tfidf__ngram_range_max', 1)
                ),
                min_df=params.get('tfidf__min_df', 1),
                max_df=params.get('tfidf__max_df', 1.0)
            )
            if use_pca:
# Note: this section is described uniformly.
                effective_components = min(pca_components,
                                        params.get('tfidf__max_features', 300))
                return Pipeline([
                    ('tfidf', base),
                    ('pca', PCA(n_components=effective_components))
                ])
            return base

        elif emb == 'w2v':
            base = Word2VecTransformer(
                vector_size=params.get('w2v__vector_size', 100),
                window=params.get('w2v__window', 5),
                min_count=params.get('w2v__min_count', 1),
                sg=params.get('w2v__sg', 0)
            )
            if use_pca:
# Note: this section is described uniformly.
                effective_components = min(pca_components,
                                        params.get('w2v__vector_size', 100))
                return Pipeline([
                    ('w2v', base),
                    ('pca', PCA(n_components=effective_components))
                ])
            return base

        elif emb == 'rubert':
            base = RuBertTiny2Embedder(
                max_length=params.get('bert__max_len', 512),
                batch_size=params.get('bert__batch_size', 128),
                learning_rate=params.get('bert__lr', 3e-4),
                pooling_type=params.get('bert__pooling_type', 'weighted')
            )
            if use_pca:
# Note: this section is described uniformly.
# Note: this section is described uniformly.
                try:
                    output_dim = base.model.bert.config.hidden_size
                    effective_components = min(pca_components, output_dim)
                    return Pipeline([
                        ('bert', base),
                        ('pca', PCA(n_components=effective_components))
                    ])
                except Exception as e:
                    print(f"Couldn't determine BERT output dim, using default PCA: {e}")
                    return Pipeline([
                        ('bert', base),
                        ('pca', PCA(n_components=min(pca_components, 100)))
                    ])
            return base
        else:
            raise ValueError(f"Unknown embedding: {emb}")


    def _prepare_data(self, params, X_train, X_text_train, X_text_raw_train):
        """Подготовка данных с учетом feature_set"""
        if params['feature_set'] == 'text-only':
            text_data = X_text_raw_train if params['embedding'] == 'rubert' else X_text_train
            return pd.DataFrame({'text': text_data})
        elif params['feature_set'] == 'categorical-only':
            return X_train.copy()
        else:  # Note: this section is described uniformly.
            X_data = X_train.copy()
            text_col = X_text_raw_train if params['embedding'] == 'rubert' else X_text_train
            X_data['text'] = text_col
            return X_data

    def _create_pipeline(self, params, feature_columns):
        """Создание пайплайна с учетом feature_set"""
        transformers = []

        if params['feature_set'] in ['categorical-only', 'mixed']:
            transformers.append(('num_cat', 'passthrough', list(feature_columns)))

        if params['feature_set'] in ['text-only', 'mixed']:
            text_transformer = self._create_text_transformer(params)
            transformers.append(('text', text_transformer, 'text'))

        model_class = self.model_classes[self.model_name]
        model_params = {k.replace('model__', ''): v for k, v in params.items()
                      if k.startswith('model__')}

        return Pipeline([
            ('features', ColumnTransformer(transformers)),
            ('model', model_class(**model_params))
        ])


    def _cross_validate(self,
                        params,
                        X_train,
                        X_text_train,
                        X_text_raw_train,
                        y_train,
                        feature_columns,
                        n_splits: int = 3):
        """
        Stratified CV для Optuna — стратификация по qcut на numpy-массиве y.
        """
# Note: this section is described uniformly.
        X_trial = self._prepare_data(
            params,
            X_train,
            X_text_train,
            X_text_raw_train
        )

# Note: this section is described uniformly.
        if isinstance(y_train, pd.Series):
            y_arr = y_train.values
        else:
            y_arr = np.asarray(y_train)

# Note: this section is described uniformly.
        cats = pd.qcut(y_arr, q=n_splits, duplicates='drop')
        codes = cats.codes      # Note: this section is described uniformly.
        valid = codes != -1     # Note: this section is described uniformly.

        X_valid      = X_trial.iloc[valid]
        y_valid      = y_arr[valid]
        codes_valid  = codes[valid]

        print(f"CV stratified bins counts: {np.bincount(codes_valid)}")
        print(f"Running {n_splits}-fold CV on {len(y_valid)} samples")

# Note: this section is described uniformly.
        skf = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=self.random_state
        )

        metrics = {
            'log_r2':   [], 'log_rmse': [], 'log_smape': [],
            'orig_r2':  [], 'orig_rmse': [], 'orig_smape': []
        }

        for fold, (tr_idx, vl_idx) in enumerate(skf.split(X_valid, codes_valid)):
            print(f" Fold {fold}: train_size={len(tr_idx)} val_size={len(vl_idx)}")

            X_tr = X_valid.iloc[tr_idx]
            X_vl = X_valid.iloc[vl_idx]
            y_tr = y_valid[tr_idx]
            y_vl = y_valid[vl_idx]

            pipeline = self._create_pipeline(params, feature_columns)
            pipeline.fit(X_tr, y_tr)

            y_pred = pipeline.predict(X_vl)

            lm = self._calculate_metrics(y_vl, y_pred)
            om = self._calculate_original_scale_metrics(y_vl, y_pred)

            metrics['log_r2'].append(lm['r2'])
            metrics['log_rmse'].append(lm['rmse'])
            metrics['log_smape'].append(lm['smape'])
            metrics['orig_r2'].append(om['r2'])
            metrics['orig_rmse'].append(om['rmse'])
            metrics['orig_smape'].append(om['smape'])

            self._log_fold_results(fold, lm, om)

        return np.mean(metrics['log_r2']), np.mean(metrics['orig_r2'])



    def _cross_validate_final_model(self,
                                    feature_columns,
                                    X_full,
                                    X_text_full,
                                    X_text_raw_full,
                                    y_full,
                                    n_splits: int = 5):
        """
        Финальное StratifiedKFold CV на всём train:
        1) pd.qcut на numpy-массиве → pandas.Categorical → .codes
        2) StratifiedKFold по этим кодам
        3) Возвращаем статистики
        """
# Note: this section is described uniformly.
        X_trial = self._prepare_data(
            self.best_params,
            X_full,
            X_text_full,
            X_text_raw_full
        )

# Note: this section is described uniformly.
        if isinstance(y_full, pd.Series):
            y_arr = y_full.values
        else:
            y_arr = np.asarray(y_full)

# Note: this section is described uniformly.
        cats = pd.qcut(y_arr, q=n_splits, duplicates='drop')
        codes = cats.codes
        valid = codes != -1

        X_valid     = X_trial.iloc[valid]
        y_valid     = y_arr[valid]
        codes_valid = codes[valid]

        print(f"Final CV stratified bins counts: {np.bincount(codes_valid)}")
        print(f"Running final {n_splits}-fold CV on {len(y_valid)} samples")

# Note: this section is described uniformly.
        skf = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=self.random_state
        )

        log_mets  = {'r2': [], 'rmse': [], 'smape': [], 'medape': []}
        orig_mets = {'r2': [], 'rmse': [], 'smape': [], 'medape': []}
        times     = []

        for fold, (tr_idx, vl_idx) in enumerate(skf.split(X_valid, codes_valid)):
            start = time.time()
            print(f" FinalCV Fold {fold}: train_size={len(tr_idx)} val_size={len(vl_idx)}")

            X_tr = X_valid.iloc[tr_idx]
            X_vl = X_valid.iloc[vl_idx]
            y_tr = y_valid[tr_idx]
            y_vl = y_valid[vl_idx]

            pipeline = self._create_pipeline(self.best_params, feature_columns)
            pipeline.fit(X_tr, y_tr)

            y_pred = pipeline.predict(X_vl)

            lm = self._calculate_metrics(y_vl, y_pred)
            om = self._calculate_original_scale_metrics(y_vl, y_pred)

            log_mets ['r2'].append(lm['r2'])
            log_mets ['rmse'].append(lm['rmse'])
            log_mets ['smape'].append(lm['smape'])
            log_mets ['medape'].append(lm['medape'])

            orig_mets['r2'].append(om['r2'])
            orig_mets['rmse'].append(om['rmse'])
            orig_mets['smape'].append(om['smape'])
            orig_mets['medape'].append(om['medape'])

            times.append(time.time() - start)
            self.logger.info(
                f" Fold {fold+1}/{n_splits} | Log R2={lm['r2']:.4f} | Orig R2={om['r2']:.4f}"
            )

        def _stats(arr):
            a = np.array(arr)
            return {
                'mean': a.mean(),
                'std':  a.std(),
                'values': a.tolist(),
                'conf_interval': 1.96 * a.std() / np.sqrt(len(a)),
                'min':  a.min(),
                'max':  a.max()
            }

        return {
            'log_metrics':  {k: _stats(v) for k, v in log_mets.items()},
            'orig_metrics': {k: _stats(v) for k, v in orig_mets.items()},
            'time_stats':   _stats(times),
            'n_splits':     n_splits
        }


    def get_token_importance(self, texts, top_n=20):
        """
        Получение важности токенов/фраз для текущей модели
        Возвращает:
        - Для TF-IDF/W2V: словарь {фраза: важность}
        - Для BERT: список кортежей (токен, важность) для каждого текста
        """
        if not hasattr(self, 'best_model'):
            raise ValueError("Model not trained yet")

        transformer = self._get_text_transformer()

        if isinstance(transformer, GatedTransformerWithTokenImportance):
            if isinstance(transformer.text_transformer, RuBertTiny2Embedder):
                return transformer.get_bert_token_importance(texts)
            else:
                return transformer.get_feature_importance(top_n)
        else:
            raise ValueError("Current model doesn't support token importance")

    def _get_text_transformer(self):
        """Извлечение текстового трансформера из пайплайна"""
        if 'features' in self.best_model.named_steps:
            return self.best_model.named_steps['features'].named_transformers_['text']
        return self.best_model.named_steps['text']

    def visualize_token_importance(self, texts, save_path=None):
        """Визуализация важности токенов"""
        importance = self.get_token_importance(texts)

        if isinstance(importance, dict):  # Note: this section is described uniformly.
            import matplotlib.pyplot as plt
            phrases, scores = zip(*sorted(importance.items(), key=lambda x: -abs(x[1])))
            plt.figure(figsize=(10, 6))
            plt.barh(phrases[:20], scores[:20])
            plt.title("Top Important Phrases")
            if save_path:
                plt.savefig(save_path)
            plt.show()
        else:  # Note: this section is described uniformly.
            from IPython.display import HTML
            html_output = ""
            for i, text_importance in enumerate(importance):
                tokens, scores = zip(*text_importance)
                html_output += f"<h3>Text {i+1}</h3><div style='border:1px solid #ccc; padding:10px;'>" # Note: this section is described uniformly.
                for token, score in text_importance:
                    color = "green" if score > 0.5 else ("red" if score < 0.3 else "black")
                    html_output += f"<span style='color:{color}; font-weight:{score*100}'> {token} </span>"
                html_output += "</div><br>"
            return HTML(html_output)


    def _setup_logging(self):
        self.logger = logging.getLogger(f"{self.__class__.__name__}_{self.model_name}")
        self.logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        log_file = os.path.join(self.results_dir, f"{self.model_name}_experiment.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)

# Note: this section is described uniformly.
        if self.logger.hasHandlers():
            self.logger.handlers.clear()

        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def _log_best_params(self, best_params):
        """Логирует лучшие параметры в файл"""
        self.logger.info("\nBest parameters found:")
        for param, value in best_params.items():
            self.logger.info(f"{param}: {value}")

    def _save_evaluation_artifacts(self, cv_metrics, test_metrics, y_test, y_pred, X_test, experiment_suffix):
        """
        Сохраняет артефакты только оценки модели:
          - cv_metrics.json (финальная кросс-валидация)
          - test_metrics.json (метрики на тесте)
          - predictions_test.csv (предсказания на тесте)
          - summary_metrics.csv (табличка по метрикам)
        """
        import json
        import pandas as pd
        import os

        metrics_dir = os.path.join(self.results_dir, "metrics")
        os.makedirs(metrics_dir, exist_ok=True)

# Note: this section is described uniformly.
        cv_metrics_path = os.path.join(metrics_dir, f"cv_metrics_{experiment_suffix}.json")
        with open(cv_metrics_path, "w") as f:
            json.dump(cv_metrics, f, indent=2)

# Note: this section is described uniformly.
        test_metrics_path = os.path.join(metrics_dir, f"test_metrics_{experiment_suffix}.json")
        with open(test_metrics_path, "w") as f:
            json.dump(test_metrics, f, indent=2)

# Note: this section is described uniformly.
# Note: this section is described uniformly.
        pred_df = X_test.reset_index()
        pred_df['true_price_log'] = y_test.values if isinstance(y_test, pd.Series) else y_test
        pred_df['predicted_price_log'] = y_pred
# Note: this section is described uniformly.
        if hasattr(self, "_reverse_transformations"):
            temp_df = pd.DataFrame({'price': pred_df['true_price_log'], 'predicted': pred_df['predicted_price_log']})
            temp_df = self._reverse_transformations(temp_df)
            pred_df['true_price'] = temp_df['price']
            pred_df['predicted_price'] = temp_df['predicted']
        pred_path = os.path.join(metrics_dir, f"predictions_test_{experiment_suffix}.csv")
        pred_df.to_csv(pred_path, index=False)

# Note: this section is described uniformly.
        summary_rows = []
        for scale in ['log','orig']:
            for m in ['r2','rmse','smape','medape']:
# Note: this section is described uniformly.
                stats = cv_metrics[f'{scale}_metrics'][m]
                summary_rows.append({
                    'type': f'cv_{scale}',
                    'metric': m,
                    'mean': stats['mean'],
                    'std': stats['std'],
                    'conf_interval': stats['conf_interval']
                })
# Note: this section is described uniformly.
                value = test_metrics[f'{scale}_metrics'][m]
                summary_rows.append({
                    'type': f'test_{scale}',
                    'metric': m,
                    'value': value
                })
        summary_path = os.path.join(metrics_dir, f"summary_metrics_{experiment_suffix}.csv")
        pd.DataFrame(summary_rows).to_csv(summary_path, index=False)

        mlflow.log_artifact(cv_metrics_path)
        mlflow.log_artifact(test_metrics_path)
        mlflow.log_artifact(pred_path)
        mlflow.log_artifact(summary_path)
        self.logger.info(f"Evaluation artifacts saved to {metrics_dir}")

    def _finalize_experiment(
        self,
        study,
        feature_columns,
        X_train, X_test,
        X_text_train, X_text_test,
        X_text_raw_train, X_text_raw_test,
        y_train, y_test
    ):
        """
        1) Лог best_params,
        2) Финальная CV → self._last_cv_metrics,
        3) Итоговая модель → тестовая оценка → self._last_test_metrics,
        4) Сохраняет только результаты оценки (метрики/предсказания/таблицы), модель НЕ сохраняем!
        """

# Note: this section is described uniformly.
        self.best_params = study.best_params.copy()
        self.best_params.update({
            "feature_set":    self.fixed_feature_set,
            "embedding":      self.fixed_embedding,
            "use_pca":        self.use_pca,
            "pca_components": self.pca_components,
            "use_gate":       self.use_gate
        })
        mlflow.log_params(self.best_params)

# Note: this section is described uniformly.
        cv_metrics = self._cross_validate_final_model(
            feature_columns,
            X_train, X_text_train, X_text_raw_train,
            y_train,
            n_splits=5
        )
        self._last_cv_metrics = cv_metrics  # Note: this section is described uniformly.

# Note: this section is described uniformly.
        final_model = self._train_final_model(
            feature_columns,
            X_train, X_text_train, X_text_raw_train,
            y_train
        )
        test_metrics, y_pred = self._evaluate_final_model(
            final_model,
            X_test, X_text_test, X_text_raw_test,
            y_test
        )
        self._last_test_metrics = test_metrics  # Note: this section is described uniformly.

# Note: this section is described uniformly.
        experiment_suffix = self._make_run_name()
        self._save_evaluation_artifacts(
            cv_metrics=cv_metrics,
            test_metrics=test_metrics,
            y_test=y_test,
            y_pred=y_pred,
            X_test=X_test,
            experiment_suffix=experiment_suffix
        )

# Note: this section is described uniformly.
        study.trials_dataframe().to_csv(
            os.path.join(self.results_dir, "metrics", f"optuna_trials_{experiment_suffix}.csv"),
            index=False
        )

    def _load_data(self, sample_size=None):
        """Загрузка и начальная предобработка данных (с sample_size)."""
        self.logger.info("Loading and preprocessing data...")
        df = pd.read_csv(self.data_path[0] if isinstance(self.data_path, (list, tuple)) else self.data_path)
        df['description'] = df['description'].astype(str)

        if sample_size:
            self.logger.info(f"Applying sample_size: {sample_size}")
            df = df.iloc[:sample_size].copy()

        if self.profile_json_path:
            self.validate_strict(df)

        one_hot_only = self.model_name in ['RandomForestRegressor', 'XGBRegressor']
        process = DataProcessingPipeline(
            df,
            log_needed=True,
            norm_needed=True,
            one_hot_only=one_hot_only,
            train=True,
            use_hex_features=self.use_hex_features,
            hex_resolution=self.hex_resolution,
            manual_text_params=self.manual_text_params,
            manual_text_params_path=self.manual_text_params_path,
        )
        df = process.preprocess_base()


        return df



    def _calculate_metrics(self, y_true, y_pred):
      return {
          'r2': r2_score(y_true, y_pred),
          'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
          'smape': 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-8)),
          'medape': self._calculate_medape(y_true, y_pred)
      }


    def run(
         self,
          n_trials: int = 30,
          sample_size: int = None,
          subsample: int = None,
          fixed_feature_set: str = None,
          fixed_embedding: str = None,
          use_pca: bool = None,
          pca_components: int = None,
          save_full: bool = False,
          df_for_full: pd.DataFrame = None,
          combo_name: str = None
    ) -> dict:
        """
        Единый запуск Optuna + финальная CV + тестовая оценка
        как один MLflow-run с run_name, возвращает best_params.
        Если save_full=True — сохранит полную модель в эту же run.
        """
# Note: this section is described uniformly.
        self.fixed_feature_set = fixed_feature_set
        self.fixed_embedding   = fixed_embedding
        self.use_pca           = use_pca
        self.pca_components    = pca_components

        try:
# Note: this section is described uniformly.
            df = self._load_data(sample_size)
            (X_train, X_test,
            X_text_train, X_text_test,
            X_text_raw_train, X_text_raw_test,
            y_train, y_test) = self._prepare_train_test_data(df)



# Note: this section is described uniformly.
            if subsample:
                df_sub = pd.concat([X_train, y_train.rename("price")], axis=1)
                sub_df = self.stratified_subsample(df_sub, "price", subsample)
                idx = sub_df.index
                X_opt, Xtxt_opt, Xraw_opt, y_opt = (
                    X_train.loc[idx],
                    X_text_train.loc[idx],
                    X_text_raw_train.loc[idx],
                    y_train.loc[idx]
                )
            else:
                X_opt, Xtxt_opt, Xraw_opt, y_opt = (
                    X_train, X_text_train, X_text_raw_train, y_train
                )

# Note: this section is described uniformly.
            study = optuna.create_study(
                direction="maximize",
                sampler=optuna.samplers.TPESampler(seed=self.random_state),
                pruner=optuna.pruners.MedianPruner(n_warmup_steps=5)
            )

# Note: this section is described uniformly.
            run_name = self._make_run_name()
            mlflow.set_experiment(self.experiment_name)
            with mlflow.start_run(run_name=run_name):
# Note: this section is described uniformly.
                mlflow.set_tag("experiment_name", self.experiment_name)
                mlflow.set_tag("model_name", self.model_name)
                mlflow.set_tag("embedding", self.fixed_embedding or "none")
                mlflow.set_tag("feature_set", self.fixed_feature_set or "none")
                mlflow.set_tag("run_group", "final" if self.use_gate else "baseline")
                mlflow.set_tag("author", "Фамилия Имя")  # Note: this section is described uniformly.
                mlflow.set_tag("started_at", time.strftime('%Y-%m-%d %H:%M:%S'))
                mlflow.set_tag("hardware", "GPU" if torch.cuda.is_available() else "CPU")
                mlflow.set_tag("is_final", str(bool(self.use_gate and self.use_pca)))
# Note: this section is described uniformly.

                self._log_time_start('total')
                self._log_environment()
                self._log_time_start('optuna')

# Note: this section is described uniformly.
                mlflow.log_param("feature_set",   self.fixed_feature_set)
                mlflow.log_param("embedding",     self.fixed_embedding)
                mlflow.log_param("use_pca",       self.use_pca)
                mlflow.log_param("manual_text_params",       self.manual_text_params)
                if self.manual_text_params:
                    mlflow.log_param("manual_text_params_path", self.manual_text_params_path)
                    mlflow.log_artifact(self.manual_text_params_path, artifact_path="manual_text_features")
                if self.use_pca:
                    mlflow.log_param("pca_components", self.pca_components)
                mlflow.log_param("use_gate",      self.use_gate)
                mlflow.log_param("n_trials",      n_trials)
                mlflow.log_param("random_state",  self.random_state)


                n_trials_effective = n_trials
                if self.use_gate or (fixed_feature_set and 'gate' in fixed_feature_set.lower()):
                    n_trials = max(n_trials, 50)  # Note: this section is described uniformly.

# Note: this section is described uniformly.
                study.optimize(
                    lambda trial: self._objective(
                        trial,
                        X_opt, Xtxt_opt, Xraw_opt, y_opt,
                        feature_columns=X_train.columns
                    ),
                    n_trials=n_trials
                )
                self._log_time_end('optuna')

                self._log_time_start('final_cv')
# Note: this section is described uniformly.
                self._finalize_experiment(
                    study,
                    feature_columns=X_train.columns,
                    X_train=X_train, X_test=X_test,
                    X_text_train=X_text_train, X_text_test=X_text_test,
                    X_text_raw_train=X_text_raw_train, X_text_raw_test=X_text_raw_test,
                    y_train=y_train, y_test=y_test
                )
                self._log_time_end('final_cv')
# Note: this section is described uniformly.
                cv = self._last_cv_metrics
                tm = self._last_test_metrics

                self._log_cv_stats_to_mlflow(cv)

                mlflow.set_tag("final_test_r2", tm['orig_metrics']['r2'])
                mlflow.set_tag("final_test_medape", tm['orig_metrics']['medape'])
                mlflow.set_tag("artifact_path", os.path.join(self.results_dir, "metrics"))
                mlflow.set_tag("data_path", self.data_path)
                mlflow.set_tag("experiment_status", "finished")
                mlflow.set_tag("cv_splits", str(n_trials))


                self._log_time_start('final_model_teaching')
                if save_full:
# Note: this section is described uniformly.
                    df_full = df_for_full if df_for_full is not None else df
                    combo_name_effective = combo_name if combo_name else self._make_run_name()
                    self.fit_and_save_full_model(
                        df=df_full,
                        combo_name=combo_name_effective,
                        feature_set=self.fixed_feature_set,
                        embedding=self.fixed_embedding,
                        use_gate=self.use_gate,
                        use_pca=self.use_pca,
                        log_to_mlflow=True
                    )
                self._log_time_end('final_model_teaching')
                self._log_time_end('total')
            return study.best_params

        except Exception as e:
            self.logger.error(f"Сбой run(): {e}", exc_info=True)
            raise


    def run_for_combinations(
        self,
        combinations: list,
        n_trials: int = 30,
        sample_size: int = None,
        subsample: int = None,
        save_full: bool = False
    ) -> dict:
        """
        Для каждой комбинации (feature_set, embedding, use_pca, use_gate, manual_text_params[, save_full])
        вызывает run() как отдельный MLflow-run в **одном** эксперименте.
        Если save_full=True — полный бандл сохраняется туда же.
        """
        results = {}
# Note: this section is described uniformly.
        df = pd.read_csv(self.data_path if isinstance(self.data_path, str) else self.data_path[0])

        for combo in combinations:
# Note: this section is described uniformly.
            if len(combo) == 2:
                feature_set, embedding = combo
                use_pca = None
                use_gate = None
                manual_text = False  # Note: this section is described uniformly.
                combo_save_full = save_full
            elif len(combo) == 4:
                feature_set, embedding, use_pca, use_gate = combo
                manual_text = False  # Note: this section is described uniformly.
                combo_save_full = save_full
            elif len(combo) == 5:
                feature_set, embedding, use_pca, use_gate, manual_text = combo
                combo_save_full = save_full
            elif len(combo) == 6:
                feature_set, embedding, use_pca, use_gate, manual_text, combo_save_full = combo
            else:
                raise ValueError("Комбинация параметров некорректна по длине")

            combo_name = self._make_run_name()
            self.logger.info(f"→ Запуск конфигурации: {combo_name}")

# Note: this section is described uniformly.
            self.fixed_feature_set = feature_set
            self.fixed_embedding = embedding
            self.use_pca = use_pca
            self.use_gate = use_gate
            self.manual_text_params = manual_text  # Note: this section is described uniformly.

            try:
                best = self.run(
                    n_trials=n_trials,
                    sample_size=sample_size,
                    subsample=subsample,
                    fixed_feature_set=feature_set,
                    fixed_embedding=embedding,
                    use_pca=use_pca,
                    pca_components=self.pca_components,
                    save_full=combo_save_full,
                    df_for_full=df,                # Note: this section is described uniformly.
                    combo_name=combo_name
                )
                results[combo_name] = best

            except Exception as e:
                self.logger.error(f"Не удалось запустить {combo_name}: {e}", exc_info=True)

        return results


    def _train_final_model(self, feature_columns,
                     X_train, X_text_train, X_text_raw_train,
                     y_train):
        """Обучает модель только на тренировочных данных."""
        feature_set = self.best_params.get('feature_set', self.fixed_feature_set)
        embedding = self.best_params.get('embedding', self.fixed_embedding)

        if feature_set == 'text-only':
            text_data = X_text_raw_train if embedding == 'rubert' else X_text_train
            X_final = pd.DataFrame({'text': text_data})
        else:
            X_final = X_train.copy()
            if feature_set == 'mixed':
                text_col = X_text_raw_train if embedding == 'rubert' else X_text_train
                X_final['text'] = text_col

        pipeline = self._create_pipeline(self.best_params, feature_columns)
        pipeline.fit(X_final, y_train.values if isinstance(y_train, pd.Series) else y_train)

        return pipeline

    def _prepare_test_data(self, X_test, X_text_test, X_text_raw_test):
        if self.best_params['feature_set'] == 'text-only':
            text_data = X_text_raw_test if self.best_params['embedding'] == 'rubert' else X_text_test
            return pd.DataFrame({'text': text_data})
        elif self.best_params['feature_set'] == 'categorical-only':
            return X_test.copy()
        else:
            X_test_final = X_test.copy()
            text_col = X_text_raw_test if self.best_params['embedding'] == 'rubert' else X_text_test
            X_test_final['text'] = text_col
            return X_test_final


    def _save_artifacts(self, model, metrics):
        """Сохраняет модель, трансформер, метрики и графики с учётом флагов в имени."""
        timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
        fs          = self.best_params.get('feature_set', 'unknown')
        emb         = self.best_params.get('embedding', 'none')
        use_pca     = self.best_params.get('use_pca', self.use_pca)
        use_gate    = self.best_params.get('use_gate', self.use_gate)
        base_name   = f"{self.model_name}_{fs}_{emb}_pca_{use_pca}_gate_{use_gate}_{timestamp}"

# Note: this section is described uniformly.
        model_path = os.path.join(self.results_dir, f"model_{base_name}")
        mlflow.sklearn.save_model(model, model_path)

# Note: this section is described uniformly.
        if fs in ['text-only', 'mixed']:
# Note: this section is described uniformly.
            if 'features' in model.named_steps:
                transformer = model.named_steps['features'].named_transformers_['text']
            else:
                transformer = model.named_steps['text']

            tr_path = os.path.join(self.results_dir, f"transformer_{base_name}")
            if hasattr(transformer, 'save'):
                transformer.save(tr_path)
            else:
                joblib.dump(transformer, tr_path + '.pkl')

# Note: this section is described uniformly.
        rows = []
        for p, v in self.best_params.items():
            rows.append({'type':'best_param','metric':p,'value':str(v)})

        for scale in ['log','orig']:
            for m in ['r2','rmse','smape','medape']:
                stats = metrics['cv_metrics'][f'{scale}_metrics'][m]
                rows.append({
                    'type':f'cv_{scale}',
                    'metric':m,
                    'mean':stats['mean'],
                    'std':stats['std'],
                    'conf_interval':stats['conf_interval'],
                    **{f'fold_{i}':val for i,val in enumerate(stats['values'])}
                })

        for scale in ['log','orig']:
            for m in ['r2','rmse','smape','medape']:
                rows.append({
                    'type':f'test_{scale}',
                    'metric':m,
                    'value':metrics['test_metrics'][f'{scale}_metrics'][m]
                })

        metrics_path = os.path.join(self.results_dir, f"metrics_{base_name}.csv")
        pd.DataFrame(rows).to_csv(metrics_path, index=False)

# Note: this section is described uniformly.
        self._save_metrics_plots(metrics['cv_metrics'], timestamp, fs, emb)


    def _save_metrics_plots(self, cv_metrics, timestamp, feature_set, embedding):
        """Сохраняет графики CV-метрик с учётом флагов в имени."""
        metrics = ['r2','rmse','smape','medape']
        scales  = ['log','orig']

        for m in metrics:
            plt.figure(figsize=(12,6))
            for s in scales:
                vals = cv_metrics[f'{s}_metrics'][m]['values']
                mean = cv_metrics[f'{s}_metrics'][m]['mean']
                ci   = cv_metrics[f'{s}_metrics'][m]['conf_interval']
                plt.axhline(mean, linestyle='--',
                            label=f'{s} mean {mean:.3f}±{ci:.3f}')
                plt.plot(range(cv_metrics['n_splits']), vals, 'o-',
                        label=f'{s} {m}')
            plt.title(f'CV {m.upper()} — {self.model_name} ({feature_set},{embedding})')
            plt.xlabel('Fold'); plt.ylabel(m.upper()); plt.legend(); plt.grid(True)

            fname = (
                f"cv_{m}_plot_{self.model_name}_{feature_set}_{embedding}"
                f"_pca_{self.use_pca}_gate_{self.use_gate}_{timestamp}.png"
            )
            plt.savefig(os.path.join(self.results_dir, fname))
            plt.close()


    def create_results_table(self, experiment_names):
        """Собирает результаты из MLflow для списка experiment_names."""
        client = MlflowClient()
        all_runs = []

        for exp in experiment_names:
            e = client.get_experiment_by_name(exp)
            if not e:
                continue

            try:
                runs = client.search_runs(e.experiment_id, "attributes.status='FINISHED'")
            except Exception:
                continue

            for r in runs:
                if r.info.status != 'FINISHED':
                    continue

                params   = r.data.params
                use_pca  = params.get('use_pca',  'NA')
                use_gate = params.get('use_gate', 'NA')

                metrics = {
                    'cv_log_r2_mean':  r.data.metrics.get('cv_log_r2_mean'),
                    'cv_orig_r2_mean': r.data.metrics.get('cv_orig_r2_mean'),
                    'test_orig_r2':    r.data.metrics.get('test_orig_r2'),
# Note: this section is described uniformly.
                }

                row = {
                    'experiment': exp,
                    'run_id':     r.info.run_id,
                    'start_time': pd.to_datetime(r.info.start_time, unit='ms'),
                    'use_pca':    use_pca,
                    'use_gate':   use_gate,
                    **params,
                    **metrics
                }
                all_runs.append(row)

        df = pd.DataFrame(all_runs)
        if 'test_orig_r2' in df.columns:
            return df.sort_values('test_orig_r2', ascending=False)
        return df


    def _evaluate_final_model(self, model, X_test, X_text_test, X_text_raw_test, y_test):
      X_test_final = self._prepare_test_data(X_test, X_text_test, X_text_raw_test)
      y_pred = model.predict(X_test_final)

# Note: this section is described uniformly.
      if isinstance(y_test, pd.Series):
          y_test = y_test.values
      if isinstance(X_test_final, pd.DataFrame):
          y_test = y_test[X_test_final.index]  # Note: this section is described uniformly.

      log_metrics = self._calculate_metrics(y_test, y_pred)
      orig_metrics = self._calculate_original_scale_metrics(y_test, y_pred)

      return {'log_metrics': log_metrics, 'orig_metrics': orig_metrics}, y_pred


    def _save_predictions_per_house(self, X_test, y_true, y_pred, experiment_suffix):
      """Сохраняет предсказания и метрики для каждого объекта с ID"""
# Note: this section is described uniformly.
      pred_df = X_test.reset_index() if isinstance(X_test, pd.DataFrame) else pd.DataFrame()

# Note: this section is described uniformly.
      if isinstance(y_true, pd.Series):
          y_true = y_true.values
      if isinstance(y_pred, pd.Series):
          y_pred = y_pred.values

# Note: this section is described uniformly.
      pred_df['true_price_log'] = y_true
      pred_df['predicted_price_log'] = y_pred

# Note: this section is described uniformly.
      temp_df = pd.DataFrame({
          'price': y_true,
          'predicted': y_pred
      })
      temp_df = self._reverse_transformations(temp_df)

# Note: this section is described uniformly.
      pred_df['true_price'] = temp_df['price']
      pred_df['predicted_price'] = temp_df['predicted']
      pred_df['absolute_error'] = np.abs(pred_df['true_price'] - pred_df['predicted_price'])
      pred_df['relative_error'] = pred_df['absolute_error'] / (pred_df['true_price'] + 1e-8)

# Note: this section is described uniformly.
      columns_to_save = ['id', 'true_price_log', 'predicted_price_log',
                        'true_price', 'predicted_price',
                        'absolute_error', 'relative_error']

# Note: this section is described uniformly.
      columns_to_save = list(dict.fromkeys(columns_to_save))

# Note: this section is described uniformly.
      existing_columns = [col for col in columns_to_save if col in pred_df.columns]

# Note: this section is described uniformly.
      pred_path = os.path.join(
          self.results_dir,
          f"predictions_{self.model_name.lower()}_{experiment_suffix}.csv"
      )
      pred_df[existing_columns].to_csv(pred_path, index=False)

      self.logger.info(f"Saved per-house predictions to {pred_path}")

    def _reverse_transformations(self, df):
      """Корректное обратное преобразование цен"""
      result = df.copy()

# Note: this section is described uniformly.
      if hasattr(self, 'norm_needed') and self.norm_needed and hasattr(self, 'scaler'):
# Note: this section is described uniformly.
          temp_price = np.zeros((len(df), len(self.scaler.feature_names_in_)))
          temp_price[:, -1] = df['price']
          result['price'] = self.scaler.inverse_transform(temp_price)[:, -1]

# Note: this section is described uniformly.
          temp_pred = np.zeros((len(df), len(self.scaler.feature_names_in_)))
          temp_pred[:, -1] = df['predicted']
          result['predicted'] = self.scaler.inverse_transform(temp_pred)[:, -1]

# Note: this section is described uniformly.
      if hasattr(self, 'log_needed') and self.log_needed:
          result['price'] = np.expm1(result['price'])
          result['predicted'] = np.expm1(result['predicted'])

      return result

    def _prepare_train_test_data(self, df, test_size=0.2, n_bins=10):
        """
        Подготовка train/test данных с сохранением параметров предобработки
        и стратификацией по бинам цены (без создания лишних колонок).
        """
        df = df.copy()
# Note: this section is described uniformly.
        price_bins = pd.qcut(df['price'], q=n_bins, duplicates='drop')
# Note: this section is described uniformly.
        train_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=self.random_state,
            stratify=price_bins
        )

# Note: this section is described uniformly.

# Note: this section is described uniformly.
        use_hex_features = self.use_hex_features and 'latitude' in train_df.columns and 'longitude' in train_df.columns

# Note: this section is described uniformly.
        train_process = DataProcessingPipeline(
            train_df,
            log_needed=True,
            norm_needed=True,
            one_hot_only=self.model_name in ['RandomForestRegressor', 'XGBRegressor'],
            train=True,
            use_hex_features=use_hex_features,
            hex_resolution=self.hex_resolution,
            manual_text_params=self.manual_text_params,  # Note: this section is described uniformly.
            manual_text_params_path=self.manual_text_params_path,
        )
        train_result = train_process.prepare_for_model()
        train_processed = train_result['processed_df']

# Note: this section is described uniformly.
        self.log_needed = True
        self.norm_needed = True
        self.scaler = train_result['scaler']
        self.lat_long_scaler = train_result['lat_long_scaler']
        self.outlier_bounds = train_result['outlier_bounds']
        self.hex_stats = train_result.get('hex_stats', None)
        self.global_median_ppsm = train_result.get('global_median_ppsm', None)
        self.global_median_ppland = train_result.get('global_median_ppland', None)
        self.global_median_price = train_result.get('global_median_price', None)

# Note: this section is described uniformly.
        test_process = DataProcessingPipeline(
            test_df,
            log_needed=True,
            norm_needed=True,
            one_hot_only=self.model_name in ['RandomForestRegressor', 'XGBRegressor'],
            train=False,
            outlier_bounds=train_result['outlier_bounds'],
            scaler=train_result['scaler'],
            lat_long_scaler=train_result['lat_long_scaler'],
            use_hex_features=use_hex_features,
            hex_resolution=self.hex_resolution,
            hex_stats=train_process.hex_stats if use_hex_features else None,
            global_median_ppsm=train_process.global_median_ppsm if use_hex_features else None,
            global_median_ppland=train_process.global_median_ppland if use_hex_features else None,
            global_median_price=train_process.global_median_price if use_hex_features else None,
            manual_text_params=self.manual_text_params,  # Note: this section is described uniformly.
            manual_text_params_path=self.manual_text_params_path,
        )
        test_processed = test_process.prepare_for_model()

        def split_X_y(df):
            X_num_cat = df.drop(columns=['price', 'description', 'description_raw'])
            X_text = df['description']
            X_text_raw = df['description_raw']
            y = df['price']
            return X_num_cat, X_text, X_text_raw, y

        X_train, X_text_train, X_text_raw_train, y_train = split_X_y(train_processed)
        X_test, X_text_test, X_text_raw_test, y_test = split_X_y(test_processed)

        return (X_train, X_test, X_text_train, X_text_test,
                X_text_raw_train, X_text_raw_test, y_train, y_test)


    def _calculate_original_scale_metrics(self, y_true_log, y_pred_log):
      """Вычисление метрик в исходной шкале цен после обратного преобразования"""
      temp_df = pd.DataFrame({
          'price': y_true_log,
          'predicted': y_pred_log
      })

      if self.verbose:
          self.logger.info(f"Before reverse - price mean: {temp_df['price'].mean()}")

      temp_df = self._reverse_transformations(temp_df)

      if self.verbose:
          self.logger.info(f"After reverse - price mean: {temp_df['price'].mean()}")

      if (temp_df['price'] < 1e5).any() or (temp_df['price'] > 1e9).any():
          self.logger.warning(
              f"Unrealistic price values detected: "
              f"min={temp_df['price'].min():.2f}, max={temp_df['price'].max():.2f}"
          )

      y_true = temp_df['price']
      y_pred = temp_df['predicted']

      return {
          'r2': r2_score(y_true, y_pred),
          'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
          'smape': 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-8)),
          'medape': self._calculate_medape(y_true, y_pred)
      }


    def _log_fold_results(self, fold, log_metrics, orig_metrics):
      mlflow.log_metrics({
          f"cv_log_r2_fold{fold}": log_metrics['r2'],
          f"cv_log_rmse_fold{fold}": log_metrics['rmse'],
          f"cv_log_smape_fold{fold}": log_metrics['smape'],
          f"cv_orig_r2_fold{fold}": orig_metrics['r2'],
          f"cv_orig_rmse_fold{fold}": orig_metrics['rmse'],
          f"cv_orig_smape_fold{fold}": orig_metrics['smape']
      })
      mlflow.log_metric("fold", fold)

      self.logger.info(
          f"Fold {fold} - Log scale: R²={log_metrics['r2']:.4f}, "
          f"RMSE={log_metrics['rmse']:.4f}, SMAPE={log_metrics['smape']:.4f}%\n"
          f"Original scale: R²={orig_metrics['r2']:.4f}, "
          f"RMSE={orig_metrics['rmse']:.4f}, SMAPE={orig_metrics['smape']:.4f}%"
      )



    def _objective(self, trial, X_train, X_text_train, X_text_raw_train, y_train, feature_columns):
      params = {
          'feature_set': self.fixed_feature_set,
          'embedding': self.fixed_embedding,
          **{k: self._suggest_parameter(trial, k, v)
            for k, v in self.model_params.items()},
          **self._suggest_embedding_params(trial)
      }


      try:
          cv_score, orig_r2 = self._cross_validate(
              params,
              X_train, X_text_train, X_text_raw_train,
              y_train, feature_columns
          )
          trial.set_user_attr("orig_r2", orig_r2)
          return cv_score
      except Exception as e:
          self.logger.error(f"Trial {trial.number} failed with error: {str(e)}", exc_info=True)
          return float('-inf')


    def _evaluate_final_model(self, model, X_test, X_text_test, X_text_raw_test, y_test):
      X_test_final = self._prepare_test_data(X_test, X_text_test, X_text_raw_test)
      y_pred = model.predict(X_test_final)

      log_metrics = self._calculate_metrics(y_test, y_pred)
      orig_metrics = self._calculate_original_scale_metrics(y_test, y_pred)

      return {
          'log_metrics': log_metrics,
          'orig_metrics': orig_metrics
      }, y_pred


    def _log_final_metrics(self, cv_metrics, test_metrics):
      """Логирует метрики кросс-валидации и тестового набора"""
      report = [
          "\nFinal Model Evaluation:",
          "=" * 50,
          "Cross-Validation (10-fold) Metrics:",
          "-" * 50
      ]

# Note: this section is described uniformly.
      for scale in ['log', 'orig']:
          report.append(f"{scale.upper()} SCALE:")
          for metric in ['r2', 'rmse', 'smape', 'medape']:  # Note: this section is described uniformly.
              stats = cv_metrics[f'{scale}_metrics'][metric]
              report.append(
                  f"{metric.upper()}: {stats['mean']:.4f} ± {stats['conf_interval']:.4f} "
                  f"(std: {stats['std']:.4f})"
              )
          report.append("-" * 50)

# Note: this section is described uniformly.
      report.extend([
          "\nTest Set Metrics:",
          "-" * 50
      ])
      for scale in ['log', 'orig']:
          report.append(f"{scale.upper()} SCALE:")
          for metric in ['r2', 'rmse', 'smape', 'medape']:  # Note: this section is described uniformly.
              value = test_metrics[f'{scale}_metrics'][metric]
              report.append(f"{metric.upper()}: {value:.4f}")
          report.append("-" * 50)

      full_report = "\n".join(report)
      self.logger.info(full_report)
      print(full_report)

    def _calculate_medape(self, y_true, y_pred):
      """Calculate Median Absolute Percentage Error"""
      with np.errstate(divide='ignore', invalid='ignore'):
          ape = 100 * np.abs((y_true - y_pred) / (y_true + 1e-8))
          ape = np.where(np.isinf(ape), np.nan, ape)  # Note: this section is described uniformly.
      return np.nanmedian(ape)


    def register_model_in_registry(self, run_id, artifact_path="full_model", registry_name=None):
        """
        Регистрирует модель в MLflow Model Registry после логирования.
        :param run_id: str, ID MLflow run (обычно из mlflow.active_run().info.run_id)
        :param artifact_path: str, под каким именем модель логировалась (по умолчанию 'full_model')
        :param registry_name: str, как будет называться модель в реестре (по умолчанию experiment+model)
        :return: ModelVersion object
        """
        import mlflow

        model_uri = f"runs:/{run_id}/{artifact_path}"
        if not registry_name:
# Note: this section is described uniformly.
            registry_name = f"{self.experiment_name}_{self.model_name}"
# Note: this section is described uniformly.
        result = mlflow.register_model(
            model_uri=model_uri,
            name=registry_name
        )
        self.logger.info(f"Model registered as '{registry_name}', version {result.version}")
        return result

    def fit_and_save_full_model(
        self,
        df: pd.DataFrame,
        combo_name: Optional[str] = None,
        feature_set=None,
        embedding=None,
        model_params=None,
        transformer_params=None,
        gate_params=None,
        additional_params=None,
        use_gate=None,
        use_pca=None,
        log_to_mlflow: bool = True,
        register_in_registry: bool = True,
        registry_name: str = None
    ) -> str:
        import mlflow

        params = {
            'use_hex_features': self.use_hex_features,
            'hex_resolution': self.hex_resolution if self.use_hex_features else None,
        }
# Note: this section is described uniformly.
        if hasattr(self, 'best_params') and self.best_params is not None:
            params.update(self.best_params)
# Note: this section is described uniformly.
        if feature_set is not None:
            params['feature_set'] = feature_set
        if embedding is not None:
            params['embedding'] = embedding
        if use_gate is not None:
            params['use_gate'] = use_gate
        if use_pca is not None:
            params['use_pca'] = use_pca
        for d in (model_params, transformer_params, gate_params, additional_params):
            if d:
                params.update(d)

# Note: this section is described uniformly.
        if params.get('feature_set') == 'categorical-only':
            params['manual_text_params'] = bool(self.manual_text_params)
        else:
            params['manual_text_params'] = False  # Note: this section is described uniformly.

        if combo_name is None:
            combo_name = self._generate_combo_name(params)
        save_dir = os.path.join(self.results_dir, combo_name)
        os.makedirs(save_dir, exist_ok=True)

        prepared = self._preprocess_data(df)

        df_prepared = prepared['processed_df']

        preprocessing_data = {
            'scaler': self.scaler,
            'lat_long_scaler': self.lat_long_scaler,
            'outlier_bounds': self.outlier_bounds,
            'hex_stats': getattr(self, 'hex_stats', None),
            'global_median_ppsm': getattr(self, 'global_median_ppsm', None),
            'global_median_ppland': getattr(self, 'global_median_ppland', None),
            'global_median_price': getattr(self, 'global_median_price', None),
            'manual_text_params': params['manual_text_params']  # Note: this section is described uniformly.
        }


        X_train = df_prepared.drop(columns=['price', 'description', 'description_raw'])
        X_text_train = df_prepared['description']
        X_text_raw_train = df_prepared['description_raw']
        y_train = df_prepared['price']

        feature_columns = X_train.columns
        pipeline = self._create_pipeline(params, feature_columns)
        X_final = self._prepare_data(params, X_train, X_text_train, X_text_raw_train)
        y_fit = y_train.values if isinstance(y_train, pd.Series) else y_train

        self.resource_monitor.log_resources("before_final_fit")
        pipeline.fit(X_final, y_fit)
        self.resource_monitor.log_resources("after_final_fit")

        transformer = self._extract_transformer(pipeline)
        tokenizer = None
        if transformer is not None and hasattr(transformer, "tokenizer"):
            tokenizer = transformer.tokenizer

        run_id = None
        if log_to_mlflow:
            mlflow.set_experiment(self.experiment_name)
            if mlflow.active_run() is not None:
                self.save_full_model_components(
                        pipeline=pipeline,
                        params=params,
                        save_dir=save_dir,
                        log_to_mlflow=True,
                        preprocessing_data=preprocessing_data,
                        transformer=transformer,
                        tokenizer=tokenizer
                )
# Note: this section is described uniformly.
                for k, v in params.items():
                    if v is not None:
                        mlflow.log_param(k, v)
                run_id = mlflow.active_run().info.run_id
            else:
                with mlflow.start_run(run_name=f"full_model_{combo_name}") as run:
                    self.save_full_model_components(
                        pipeline=pipeline,
                        params=params,
                        save_dir=save_dir,
                        log_to_mlflow=True,
                        preprocessing_data=preprocessing_data,
                        transformer=transformer,
                        tokenizer=tokenizer
                    )
                    for k, v in params.items():
                        if v is not None:
                            mlflow.log_param(k, v)
                    run_id = run.info.run_id

            if register_in_registry and run_id is not None:
                if not registry_name:
                    dataset = os.path.splitext(os.path.basename(
                        self.data_path[0] if isinstance(self.data_path, (list, tuple)) else self.data_path
                    ))[0]
                    registry_name = (
                        f"{self.model_name}"
                        f"+{dataset}"
                        f"+{params.get('embedding','none')}"
                        f"+pca_{params.get('use_pca', False)}"
                        f"+gate_{params.get('use_gate', False)}"
                        f"+textfeat_{params.get('manual_text_params', False)}"
                    )
                self.register_model_in_registry(
                    run_id=run_id,
                    artifact_path="full_model",
                    registry_name=registry_name
                )

        mlflow.set_tag("saved_model_dir", save_dir)
        mlflow.set_tag("saved_combo_name", combo_name)
        mlflow.set_tag("full_model_logged", "1")
        if register_in_registry and run_id is not None:
            mlflow.set_tag("registered_in_registry", "1")
            mlflow.set_tag("registry_name", registry_name)
        else:
            mlflow.set_tag("registered_in_registry", "0")

        self.logger.info(f"Full model saved to {save_dir} and logged to MLflow")
        return save_dir


    def _generate_combo_name(self, params: Dict) -> str:
        """Генерирует имя для комбинации параметров"""
        parts = [
            self.experiment_name,
            self.model_name,
            params.get('feature_set', 'none'),
            params.get('embedding', 'none'),
            f"pca_{params.get('use_pca', False)}",
            f"gate_{params.get('use_gate', False)}",
            f"manualtext_{params.get('manual_text_params', False)}"  # Note: this section is described uniformly.
        ]
        return "_".join(str(p) for p in parts)



    def _preprocess_data(self, df: pd.DataFrame) -> Dict:
        process = DataProcessingPipeline(
            df,
            log_needed=True,
            norm_needed=True,
            one_hot_only=self.model_name in ['RandomForestRegressor', 'XGBRegressor'],
            train=True,
            use_hex_features=self.use_hex_features,
            hex_resolution=self.hex_resolution,
            manual_text_params=self.manual_text_params,
            manual_text_params_path=self.manual_text_params_path,
        )

        df_base_prep = process.preprocess_base()


        prepared = process.prepare_for_model()




# Note: this section is described uniformly.
        self.scaler = prepared['scaler']
        self.lat_long_scaler = prepared['lat_long_scaler']
        self.outlier_bounds = prepared['outlier_bounds']
        self.hex_stats = prepared['hex_stats']
        self.global_median_ppsm = prepared['global_median_ppsm']
        self.global_median_ppland = prepared['global_median_ppland']
        self.global_median_price = prepared['global_median_price']
        return prepared

    def save_full_model_components(
                self,
        pipeline,
        params,
        save_dir,
        log_to_mlflow=True,
        transformer=None,
        tokenizer=None,
        preprocessing_data=None
        ):
        import joblib
        import json
        import os

        os.makedirs(save_dir, exist_ok=True)

# Note: this section is described uniformly.
        pipeline_path = os.path.join(save_dir, 'pipeline.pkl')
        joblib.dump(pipeline, pipeline_path)
        if log_to_mlflow:
            mlflow.log_artifact(pipeline_path, artifact_path='full_model')

# Note: this section is described uniformly.
        params_path = os.path.join(save_dir, 'params.json')
        params_to_save = dict(params)
# Note: this section is described uniformly.
        if params_to_save.get('feature_set') == 'categorical-only':
            params_to_save['manual_text_params'] = bool(params_to_save.get('manual_text_params', False))
        else:
            params_to_save['manual_text_params'] = False
        with open(params_path, 'w') as f:
            json.dump(params_to_save, f, indent=2)
        if log_to_mlflow:
            mlflow.log_artifact(params_path, artifact_path='params')

# Note: this section is described uniformly.
        if self.manual_text_params and self.manual_text_params_path:
            mlflow.log_artifact(self.manual_text_params_path, artifact_path='manual_text_features')

# Note: this section is described uniformly.
        if transformer is not None:
            transformer_dir = os.path.join(save_dir, 'transformer')
            try:
                if hasattr(transformer, "save"):
                    transformer.save(transformer_dir)
                    if log_to_mlflow:
                        mlflow.log_artifact(transformer_dir, artifact_path='transformer')
                else:
                    transformer_pkl = transformer_dir + ".pkl"
                    joblib.dump(transformer, transformer_pkl)
                    if log_to_mlflow:
                        mlflow.log_artifact(transformer_pkl, artifact_path='transformer')
            except Exception as e:
                print(f"[SAVE] Transformer save error: {e}")
        else:
            print("[SAVE] No transformer to save.")

# Note: this section is described uniformly.
        if tokenizer is not None:
            tok_dir = os.path.join(save_dir, 'tokenizer')
            os.makedirs(tok_dir, exist_ok=True)
            try:
                tokenizer.save_pretrained(tok_dir)
                if log_to_mlflow:
                    mlflow.log_artifact(tok_dir, artifact_path='tokenizer')
            except Exception as e:
                print(f"[SAVE] Tokenizer save error: {e}")
        elif transformer is not None and hasattr(transformer, 'tokenizer') and transformer.tokenizer is not None:
            tok_dir = os.path.join(save_dir, 'tokenizer')
            os.makedirs(tok_dir, exist_ok=True)
            try:
                transformer.tokenizer.save_pretrained(tok_dir)
                if log_to_mlflow:
                    mlflow.log_artifact(tok_dir, artifact_path='tokenizer')
            except Exception as e:
                print(f"[SAVE] Transformer.tokenizer save error: {e}")
        else:
            print("[SAVE] No tokenizer to save.")

# Note: this section is described uniformly.
        if preprocessing_data is not None:
            preproc_path = os.path.join(save_dir, 'preprocessing.pkl')
# Note: this section is described uniformly.
            if 'manual_text_params' not in preprocessing_data:
# Note: this section is described uniformly.
                if params_to_save.get('feature_set') == 'categorical-only':
                    preprocessing_data['manual_text_params'] = bool(params_to_save.get('manual_text_params', False))
                else:
                    preprocessing_data['manual_text_params'] = False
            try:
                joblib.dump(preprocessing_data, preproc_path)
                if log_to_mlflow:
                    mlflow.log_artifact(preproc_path, artifact_path='preprocessing')
            except Exception as e:
                print(f"[SAVE] Preprocessing save error: {e}")
        else:
            print("[SAVE] No preprocessing_data to save.")

# Note: this section is described uniformly.
        if self.manual_text_params and self.manual_text_features_dict:
            manual_dir = os.path.join(save_dir, 'manual_text_features')
            os.makedirs(manual_dir, exist_ok=True)
            manual_json = os.path.join(manual_dir, 'manual_text_features.json')
            with open(manual_json, 'w', encoding='utf-8') as f:
                json.dump(self.manual_text_features_dict, f, ensure_ascii=False, indent=2)
            if log_to_mlflow:
                mlflow.log_artifact(manual_json, artifact_path='manual_text_features')

        print(f"[SAVE] All available components saved to {save_dir} (logged to MLflow: {log_to_mlflow})")



    @classmethod
    def load_full_model(cls, model_dir):
        import joblib
        import json
        import os


        print(f"\n=== [LOAD] ===")
        print(f"Загрузка модели из {model_dir}")

# Note: this section is described uniformly.
        pipeline = None
        pipeline_path = os.path.join(model_dir, 'pipeline.pkl')
        if os.path.exists(pipeline_path):
            print(f"[LOAD] pipeline.pkl найден, грузим...")
            pipeline = joblib.load(pipeline_path)
            print(f"[LOAD] pipeline: {type(pipeline)}")
        else:
            print(f"[LOAD][WARN] pipeline.pkl НЕ найден!")

# Note: this section is described uniformly.
        params = None
        param_fnames = [
            os.path.join(model_dir, 'params.json'),
            os.path.join(os.path.dirname(model_dir), 'params', 'params.json'),  # Note: this section is described uniformly.
            os.path.join(model_dir, '..', 'params', 'params.json'),            # Note: this section is described uniformly.
        ]
        for fpath in param_fnames:
            fpath = os.path.abspath(fpath)
            if os.path.exists(fpath):
                print(f"[LOAD] {os.path.basename(fpath)} найден, грузим...")
                with open(fpath) as f:
                    params = json.load(f)
                break
        if params is None:
            print("[LOAD][WARN] Параметры не найдены!")
        else:
            print(f"[LOAD] params загружены: use_pca={params.get('use_pca')}, ...")

# Note: this section is described uniformly.
        preproc_path = os.path.join(model_dir, 'preprocessing.pkl')
        preproc = {}
        if os.path.exists(preproc_path):
            print(f"[LOAD] preprocessing.pkl найден, грузим...")
            preproc = joblib.load(preproc_path)
            print(f"[LOAD] preprocessing keys: {list(preproc.keys())}")
        else:
            print(f"[LOAD][WARN] preprocessing.pkl НЕ найден!")


        def _is_gated_transformer(dir):
            params_path = os.path.join(dir, 'params.json')
            if os.path.exists(params_path):
                with open(params_path) as f:
                    p = json.load(f)
                    return p.get('use_gate', False) or p.get('type', '').lower() == 'bert'
            return False

        def _is_rubert_embedder(dir):
            files = ['embedder_config.json', 'head_weights.pt']
            return all(os.path.exists(os.path.join(dir, f)) for f in files)


        transformer_dir = os.path.join(model_dir, 'transformer')
        transformer_pkl = os.path.join(model_dir, 'transformer.pkl')
        transformer = None
        transformer_loaded = False
        if os.path.isdir(transformer_dir):
            if _is_gated_transformer(transformer_dir):
# Note: this section is described uniformly.
                print(f"[LOAD] transformer/ — папка существует и params.json есть, пробуем как GatedTransformer...")
                try:
                    transformer = GatedTransformerWithTokenImportance.load(transformer_dir)
                    print(f"[LOAD] Загружен GatedTransformerWithTokenImportance!")
                    transformer_loaded = True
                except Exception as e:
                    print(f"[GATE LOAD FAIL] {e}")
            elif _is_rubert_embedder(transformer_dir):
                print(f"[LOAD] transformer/ — похоже на RuBertTiny2Embedder (без gate), пробуем...")
                try:
                    transformer = RuBertTiny2Embedder.load(transformer_dir)
                    print(f"[LOAD] Загружен RuBertTiny2Embedder!")
                    transformer_loaded = True
                except Exception as e:
                    print(f"[RUBERT LOAD FAIL] {e}")
            elif os.path.exists(os.path.join(transformer_dir, 'word2vec.model')):
                print(f"[LOAD] transformer/ содержит word2vec.model, пробуем Word2VecTransformer.load...")
                try:
                    from w2v_transformer_saving import Word2VecTransformer
                    transformer = Word2VecTransformer.load(transformer_dir)
                    print(f"[LOAD] Загружен Word2VecTransformer!")
                    transformer_loaded = True
                except Exception as e:
                    print(f"[W2V LOAD FAIL] {e}")
        if not transformer_loaded and os.path.exists(transformer_pkl):
            print(f"[LOAD] transformer.pkl найден, пробуем joblib.load...")
            try:
                transformer = joblib.load(transformer_pkl)
                print(f"[LOAD] Загружен transformer.pkl: {type(transformer)}")
                transformer_loaded = True
            except Exception as e:
                print(f"[joblib LOAD FAIL] {e}")

        if not transformer_loaded:
            print("[LOAD][WARN] Transformer НЕ загружен!")

# Note: this section is described uniformly.
        tokenizer = None
        tokenizer_dir = os.path.join(model_dir, 'tokenizer')
        if os.path.isdir(tokenizer_dir):
            try:
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
                print(f"[LOAD] Tokenizer загружен из tokenizer/")
            except Exception as e:
                print(f"[TOKENIZER LOAD FAIL] {e}")
        elif transformer is not None and hasattr(transformer, 'tokenizer'):
            tokenizer = transformer.tokenizer
            print(f"[LOAD] Tokenizer взят из transformer.tokenizer ({type(tokenizer)})")
        else:
            print("[LOAD][WARN] Tokenizer НЕ найден!")

# Note: this section is described uniformly.
        gate = None
        if transformer is not None:
            if hasattr(transformer, "gate_token_net"):
                gate = transformer.gate_token_net
                print("[LOAD] gate_token_net найден в transformer.")
            elif hasattr(transformer, "gate_net"):
                gate = transformer.gate_net
                print("[LOAD] gate_net найден в transformer.")
            else:
                print("[LOAD] gate внутри transformer НЕ найден.")
        else:
            print("[LOAD] transformer пустой — gate не ищем.")

# Note: this section is described uniformly.
        print("\n[LOAD] --- ИТОГ ---")
        print("pipeline:", type(pipeline))
        print("transformer:", type(transformer))
        print("tokenizer:", type(tokenizer))
        print("gate:", type(gate))
        print("params:", params is not None)
        print("scaler:", 'scaler' in preproc)
        print("[LOAD] bundle готов!\n")

        bundle = {
            'pipeline': pipeline,
            'transformer': transformer,
            'tokenizer': tokenizer,
            'transformer_raw': transformer_raw,
            'gate': gate,
            'params': params,
            'scaler': preproc.get('scaler'),
            'lat_long_scaler': preproc.get('lat_long_scaler'),
            'outlier_bounds': preproc.get('outlier_bounds'),
            'hex_stats': preproc.get('hex_stats'),
            'global_median_ppsm': preproc.get('global_median_ppsm'),
            'global_median_ppland': preproc.get('global_median_ppland'),
            'global_median_price': preproc.get('global_median_price'),
            'preprocessing_data': preproc,
        }
        return bundle


    @classmethod
    def load_full_model_from_mlflow(cls, run_id, local_dir=None):
      """
      Скачивает full_model артефакты из MLflow (по run_id), кладет в local_dir (tmp если не задано),
      и вызывает стандартный loader.
      """
      from mlflow.tracking import MlflowClient
      import tempfile

      client = MlflowClient()
      if local_dir is None:
          temp_dir = tempfile.mkdtemp(prefix="mlflow_model_")
      else:
          temp_dir = local_dir
          os.makedirs(temp_dir, exist_ok=True)

# Note: this section is described uniformly.
      for artifact in ['full_model', 'params', 'preprocessing', 'transformer', 'tokenizer']:
          try:
              client.download_artifacts(run_id, artifact, temp_dir)
          except Exception as e:
              print(f"[MLFLOW][WARN] Артефакт '{artifact}' не найден: {e}")

      print(f"[MLFLOW] Все артефакты скачаны в {temp_dir}")

# Note: this section is described uniformly.
      model_dir = os.path.join(temp_dir, 'full_model')
      if not os.path.exists(model_dir):
          model_dir = temp_dir
      return cls.load_full_model(model_dir)


    @classmethod
    def find_and_load_full_model(
        cls,
        experiment_name: str,
        param_filters: dict,
        local_dir: str = None,
        return_runinfo: bool = False,
        by: str = 'last'  # Note: this section is described uniformly.
    ):
        """
        Ищет run по experiment_name и заданным параметрам, скачивает и загружает full_model.
        :param cls: твой класс (например, PricePredictionExperiment)
        :param experiment_name: имя эксперимента в MLflow
        :param param_filters: dict вида {'feature_set':'mixed', 'embedding':'tfidf', ...}
        :param local_dir: куда скачивать артефакты (по умолчанию temp)
        :param return_runinfo: если True — вернёт ещё и инфу о run (run_id, params)
        :param by: 'last' — последний по времени, 'best_r2' — максимальный test_orig_r2
        :return: bundle (dict), опционально runinfo
        """
        from mlflow.tracking import MlflowClient

        client = MlflowClient()
        exp = client.get_experiment_by_name(experiment_name)
        if not exp:
            raise ValueError(f"Experiment '{experiment_name}' not found in MLflow!")

# Note: this section is described uniformly.
        filter_strings = []
        for k, v in param_filters.items():
            if isinstance(v, bool):
                v = str(v)
            filter_strings.append(f"params.{k} = '{v}'")
        filter_query = " and ".join(filter_strings)
        print(f"[MLFLOW][FIND] Using filter: {filter_query}")

        runs = client.search_runs(
            [exp.experiment_id],
            filter_query,
            order_by=["start_time DESC"]
        )
        if not runs:
            raise ValueError(f"No runs found for filter: {filter_query}")

# Note: this section is described uniformly.
        if by == 'last':
# Note: this section is described uniformly.
            selected_run = runs[0]
            print(f"[MLFLOW][FOUND] (last) run_id: {selected_run.info.run_id}")
        elif by == 'best_r2':
# Note: this section is described uniformly.
# Note: this section is described uniformly.
            def get_r2(r):
                m = r.data.metrics
                return m.get('test_orig_r2', float('-inf'))
            best_run = max(runs, key=get_r2)
            selected_run = best_run
            print(f"[MLFLOW][FOUND] (best_r2={get_r2(best_run)}) run_id: {best_run.info.run_id}")
        else:
            raise ValueError(f"Unknown 'by' value: {by}")

        run_id = selected_run.info.run_id
        params_found = selected_run.data.params

# Note: this section is described uniformly.
        bundle = cls.load_everything_from_mlflow(run_id)
        if return_runinfo:
            return bundle, {'run_id': run_id, 'params': params_found}
        return bundle

    def _extract_transformer(self, pipeline):
        """
        Извлекает текстовый трансформер из пайплайна (Pipeline с ColumnTransformer или просто шаг 'text').
        """
# Note: this section is described uniformly.
        if 'features' in pipeline.named_steps:
            coltr = pipeline.named_steps['features']
# Note: this section is described uniformly.
            for name, trf, _ in coltr.transformers_:
                if name == 'text':
                    print(f"[EXTRACT] Найден transformer под именем 'text': {type(trf)}")
                    return trf
            print("[EXTRACT] В ColumnTransformer нет шага 'text'")
            return None
# Note: this section is described uniformly.
        elif 'text' in pipeline.named_steps:
            print(f"[EXTRACT] Найден шаг 'text': {type(pipeline.named_steps['text'])}")
            return pipeline.named_steps['text']
        else:
            print("[EXTRACT] Не найден текстовый трансформер в пайплайне!")
            return None

    def _save_transformer(self, transformer, save_dir: str) -> None:
        """Сохраняет трансформер в поддиректорию"""
        tf_path = os.path.join(save_dir, 'text_transformer')
        if hasattr(transformer, 'save'):
            transformer.save(tf_path)
        else:
            import joblib
            joblib.dump(transformer, tf_path + '.pkl')

    def _save_tokenizer(self, transformer, save_dir: str) -> None:
        """Сохраняет токенизатор в поддиректорию"""
        if hasattr(transformer, 'tokenizer') and transformer.tokenizer is not None:
            tok_path = os.path.join(save_dir, 'tokenizer')
            os.makedirs(tok_path, exist_ok=True)
            transformer.tokenizer.save_pretrained(tok_path)

    def _save_gate_components(self, transformer, save_path):
      """Сохраняет компоненты gate отдельно"""
      if hasattr(transformer, 'gate_token_net'):
          gate_dir = os.path.join(save_path, 'gate')
          os.makedirs(gate_dir, exist_ok=True)

# Note: this section is described uniformly.
          torch.save(transformer.gate_token_net.state_dict(),
                  os.path.join(gate_dir, 'weights.pt'))

# Note: this section is described uniformly.
          gate_cfg = {
              'hidden_size': transformer.model.bert.config.hidden_size,
              'device': transformer.device
          }
          with open(os.path.join(gate_dir, 'config.json'), 'w') as f:
              json.dump(gate_cfg, f)

    def _load_gate_components(self, transformer, load_path):
      """Загружает компоненты gate в существующий трансформер"""
      gate_dir = os.path.join(load_path, 'gate')
      if os.path.exists(gate_dir):
          with open(os.path.join(gate_dir, 'config.json')) as f:
              gate_cfg = json.load(f)

          transformer.gate_token_net = TokenContextGate(
              gate_cfg['hidden_size']).to(gate_cfg['device'])

          transformer.gate_token_net.load_state_dict(
              torch.load(os.path.join(gate_dir, 'weights.pt'),
              map_location=gate_cfg['device']))


    @classmethod
    def load_everything_from_mlflow(cls, run_id, verbose=True):
        """
        Скачивает все артефакты (pipeline, params, preprocessing, transformer, tokenizer) из MLflow (по run_id),
        корректно обходит разную структуру папок, загружает нужные файлы.
        """
        import tempfile
        import os
        import joblib
        import json
        from mlflow.tracking import MlflowClient

        client = MlflowClient()
        temp_dir = tempfile.mkdtemp(prefix="mlflow_bundle_")

        to_download = [
            "full_model",       # Note: this section is described uniformly.
            "params",           # Note: this section is described uniformly.
            "preprocessing",    # Note: this section is described uniformly.
            "transformer",      # Note: this section is described uniformly.
            "tokenizer",        # Note: this section is described uniformly.
            ''
        ]
        for art in to_download:
            try:
                client.download_artifacts(run_id, art, temp_dir)
                if verbose: print(f"[MLFLOW] {art}: OK")
            except Exception as e:
                if verbose: print(f"[MLFLOW] {art}: нет ({e})")

        bundle = {}

# Note: this section is described uniformly.
        pipeline_path = os.path.join(temp_dir, "full_model", "pipeline.pkl")
        if os.path.exists(pipeline_path):
            bundle["pipeline"] = joblib.load(pipeline_path)
            if verbose: print("[LOAD] pipeline: OK")
        else:
            bundle["pipeline"] = None
            if verbose: print("[LOAD] pipeline.pkl не найден!")

# Note: this section is described uniformly.
        params_path = os.path.join(temp_dir, "params", "params.json")
        if os.path.exists(params_path):
            with open(params_path) as f:
                bundle["params"] = json.load(f)
            if verbose: print("[LOAD] params: OK")
        else:
            bundle["params"] = None
            if verbose: print("[LOAD] params.json не найден!")

# Note: this section is described uniformly.
        preproc_path = os.path.join(temp_dir, "preprocessing", "preprocessing.pkl")
        if os.path.exists(preproc_path):
            bundle["preprocessing"] = joblib.load(preproc_path)
            if verbose: print("[LOAD] preprocessing: OK")
        else:
            bundle["preprocessing"] = None
            if verbose: print("[LOAD] preprocessing.pkl не найден!")

# Note: this section is described uniformly.
        transformer = None
        base = os.path.join(temp_dir, "transformer")
# Note: this section is described uniformly.
# Note: this section is described uniformly.
# Note: this section is described uniformly.
# Note: this section is described uniformly.
        possible_pkl = [
            os.path.join(base, "transformer.pkl"),
            os.path.join(base, "transformer", "transformer.pkl"),
        ]
        possible_dirs = [
            os.path.join(base, "transformer"),
            base
        ]

        loaded = False
# Note: this section is described uniformly.
        for pkl_path in possible_pkl:
            if os.path.exists(pkl_path):
                try:
                    transformer = joblib.load(pkl_path)
                    loaded = True
                    if verbose: print(f"[LOAD] transformer.pkl: OK ({pkl_path})")
                    break
                except Exception as e:
                    if verbose: print(f"[LOAD] Ошибка при загрузке {pkl_path}: {e}")

# Note: this section is described uniformly.
        if not loaded:
            for tdir in possible_dirs:
                if os.path.isdir(tdir):
                    files = os.listdir(tdir)
# Note: this section is described uniformly.
                    if "vectorizer.pkl" in files:
                        try:
# Note: this section is described uniformly.
                            vectorizer = joblib.load(os.path.join(tdir, "vectorizer.pkl"))
                            params = joblib.load(os.path.join(tdir, "params.pkl")) if "params.pkl" in files else {}
                            transformer = {"vectorizer": vectorizer, "params": params}
                            loaded = True
                            if verbose: print(f"[LOAD] TFIDF vectorizer+params: OK ({tdir})")
                            break
                        except Exception as e:
                            if verbose: print(f"[LOAD] TFIDF в {tdir} не загрузился: {e}")
# Note: this section is described uniformly.
                    elif "word2vec.model" in files:
                        try:
                            from w2v_transformer_saving import Word2VecTransformer
                            transformer = Word2VecTransformer.load(tdir)
                            loaded = True
                            if verbose: print(f"[LOAD] Word2VecTransformer: OK ({tdir})")
                            break
                        except Exception as e:
                            if verbose: print(f"[LOAD] W2V в {tdir} не загрузился: {e}")
# Note: this section is described uniformly.
                    elif "embedder_config.json" in files and "head_weights.pt" in files:
                        try:
                            transformer = RuBertTiny2Embedder.load(tdir)
                            loaded = True
                            if verbose: print(f"[LOAD] RuBertTiny2Embedder: OK ({tdir})")
                            break
                        except Exception as e:
                            if verbose: print(f"[LOAD] RuBert в {tdir} не загрузился: {e}")
# Note: this section is described uniformly.
                    elif "params.json" in files:
                        try:
                            transformer = GatedTransformerWithTokenImportance.load(tdir)
                            loaded = True
                            if verbose: print(f"[LOAD] GatedTransformer: OK ({tdir})")
                            break
                        except Exception as e:
                            if verbose: print(f"[LOAD] GatedTransformer в {tdir} не загрузился: {e}")

        if not loaded:
            transformer = None
            if verbose: print("[LOAD] transformer не удалось загрузить!")

        bundle["transformer"] = transformer

# Note: this section is described uniformly.
        tokenizer_base = os.path.join(temp_dir, "tokenizer", "tokenizer")
        tokenizer = None
        if os.path.isdir(tokenizer_base):
            try:
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained(tokenizer_base)
                if verbose: print("[LOAD] tokenizer: OK")
            except Exception as e:
                if verbose: print(f"[LOAD] tokenizer не загрузился: {e}")
        bundle["tokenizer"] = tokenizer

# Note: this section is described uniformly.
        preproc = bundle.get("preprocessing") or {}
        for k in [
            "scaler", "lat_long_scaler", "outlier_bounds",
            "hex_stats", "global_median_ppsm", "global_median_ppland", "global_median_price"
        ]:
            bundle[k] = preproc.get(k) if isinstance(preproc, dict) else None

        if verbose:
            print("[BUNDLE] pipeline:", type(bundle.get('pipeline')))
            print("[BUNDLE] transformer:", type(bundle.get('transformer')))
            print("[BUNDLE] tokenizer:", type(bundle.get('tokenizer')))
            print("[BUNDLE] params:", bool(bundle.get('params')))
            print("[BUNDLE] preprocessing:", bool(bundle.get('preprocessing')))


# Note: this section is described uniformly.
        bundle['manual_text_features_dict'] = {}
        manual_art_dir = os.path.join(temp_dir, 'manual_text_features')
        if os.path.isdir(manual_art_dir):
            for fn in os.listdir(manual_art_dir):
                if fn.endswith('.json'):
                    path = os.path.join(manual_art_dir, fn)
                    with open(path, 'r', encoding='utf-8') as f:
                        bundle['manual_text_features_dict'] = json.load(f)
                    break


        return bundle
