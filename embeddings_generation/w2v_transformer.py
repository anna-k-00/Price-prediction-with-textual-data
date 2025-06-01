import numpy as np
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

class Word2VecTransformer(BaseEstimator, TransformerMixin):
    """
    Custom transformer for converting text descriptions into Word2Vec embeddings.
    
    Parameters:
        vector_size (int): Dimensionality of the word vectors
        window (int): Maximum distance between current and predicted word
        min_count (int): Ignores words with frequency lower than this
        sg (int): Training algorithm - 1 for skip-gram, 0 for CBOW
        workers (int): Number of worker threads
        epochs (int): Number of iterations over the corpus
        pooling (str): How to aggregate word vectors ('mean', 'sum', 'max')
        preprocess (bool): Whether to perform basic text preprocessing
    """
    
    def __init__(self, vector_size=100, window=5, min_count=1, sg=0, 
                 workers=4, epochs=10, pooling='mean', preprocess=True):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.sg = sg
        self.workers = workers
        self.epochs = epochs
        self.pooling = pooling
        self.preprocess = preprocess
        
    def _tokenize(self, texts):
        """Tokenize input texts using simple preprocessing"""
        return [simple_preprocess(text, 
                            min_len=2,       # Minimum token length
                            max_len=15,      # Maximum token length
                            deacc=True)       # Convert to lowercase) 
                                  for text in texts]
    
    def _get_embeddings(self, tokens):
        """Convert tokens to document embeddings using the trained Word2Vec model"""
        embeddings = []
        for doc_tokens in tokens:
            doc_vectors = []
            for token in doc_tokens:
                if token in self.w2v_model_.wv:
                    doc_vectors.append(self.w2v_model_.wv[token])
            
            if len(doc_vectors) == 0:
                # If no words found in vocabulary, return zero vector
                embeddings.append(np.zeros(self.vector_size))
            else:
                doc_vectors = np.array(doc_vectors)
                if self.pooling == 'mean':
                    embeddings.append(doc_vectors.mean(axis=0))
                elif self.pooling == 'sum':
                    embeddings.append(doc_vectors.sum(axis=0))
                elif self.pooling == 'max':
                    embeddings.append(doc_vectors.max(axis=0))
                else:
                    raise ValueError(f"Unknown pooling method: {self.pooling}")
        
        return np.array(embeddings)
    
    def fit(self, X, y=None):
        """Fit Word2Vec model on the input texts"""
        if self.preprocess:
            tokens = self._tokenize(X)
        else:
            tokens = X  # assume X is already tokenized
            
        self.w2v_model_ = Word2Vec(
            sentences=tokens,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            sg=self.sg,
            workers=self.workers,
            epochs=self.epochs
        )
        
        return self
    
    def transform(self, X):
        """Transform input texts into document embeddings"""
        check_is_fitted(self, 'w2v_model_')
        
        if self.preprocess:
            tokens = self._tokenize(X)
        else:
            tokens = X  # assume X is already tokenized
            
        return self._get_embeddings(tokens)
    
    def fit_transform(self, X, y=None):
        """Fit and transform in one step"""
        return self.fit(X).transform(X)
