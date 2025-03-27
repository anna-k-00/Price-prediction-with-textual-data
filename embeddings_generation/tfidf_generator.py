import pandas as pd
import re
import numpy as np
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
import optuna
import mlflow
import mlflow.sklearn
from datetime import datetime
from tqdm import tqdm
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils.validation import check_is_fitted
import numpy as np
from gensim.utils import simple_preprocess

class TfidfTransformer(BaseEstimator, TransformerMixin):
    """
    Custom transformer for TF-IDF vectorization with similar interface to Word2VecTransformer.
    
    Parameters:
        max_features (int): Maximum number of vocabulary items
        ngram_range (tuple): Range of n-grams to use (e.g., (1,2) for unigrams+bigrams)
        min_df (int): Minimum document frequency for vocabulary
        max_df (float): Maximum document frequency (proportion)
        preprocess (bool): Whether to perform basic text preprocessing
        norm (str): Norm for normalization ('l1', 'l2', None)
        use_idf (bool): Enable inverse-document-frequency reweighting
        smooth_idf (bool): Smooth IDF weights
        sublinear_tf (bool): Apply sublinear TF scaling
    """
    
    def __init__(self, max_features=None, ngram_range=(1,1), min_df=1, 
                 max_df=1.0, preprocess=True, norm='l2', use_idf=True,
                 smooth_idf=True, sublinear_tf=False):
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_df = max_df
        self.preprocess = preprocess
        self.norm = norm
        self.use_idf = use_idf
        self.smooth_idf = smooth_idf
        self.sublinear_tf = sublinear_tf
        
    def _tokenize(self, texts):
        """Tokenize input texts using simple preprocessing"""
        return [" ".join(simple_preprocess(text, 
                                    min_len=2,
                                    max_len=15,
                                    deacc=True
                                    )) 
                                      for text in texts]
    
    def fit(self, X, y=None):
        """Fit TF-IDF vectorizer"""
        if self.preprocess:
            processed_texts = self._tokenize(X)
        else:
            processed_texts = X  # assume X is already preprocessed
            
        self.vectorizer_ = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            min_df=self.min_df,
            max_df=self.max_df,
            norm=self.norm,
            use_idf=self.use_idf,
            smooth_idf=self.smooth_idf,
            sublinear_tf=self.sublinear_tf
        )
        
        self.vectorizer_.fit(processed_texts)
        return self
    
    def transform(self, X):
        """Transform input texts into TF-IDF vectors"""
        check_is_fitted(self, 'vectorizer_')
        
        if self.preprocess:
            processed_texts = self._tokenize(X)
        else:
            processed_texts = X
            
        return self.vectorizer_.transform(processed_texts)
    
    def fit_transform(self, X, y=None):
        """Fit and transform in one step"""
        return self.fit(X).transform(X)
