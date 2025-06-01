import os
import json
import torch
import numpy as np
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer, AdamW
from sklearn.base import BaseEstimator, TransformerMixin
from typing import Optional, List, Dict, Any, Union
import warnings

warnings.filterwarnings('ignore')

class RuBertTiny2Embedder(BaseEstimator, TransformerMixin):
    """
    A transformer class for generating embeddings using rubert-tiny2 model with optional fine-tuning.
    Supports multiple pooling strategies and can be used as part of scikit-learn pipelines.
    
    Attributes:
        max_length: Maximum sequence length for tokenization
        batch_size: Batch size for training and inference
        learning_rate: Learning rate for fine-tuning
        num_epochs: Number of training epochs
        device: Device to run the model on (cuda/cpu)
        use_cv: Whether to use cross-validation during training
        pooling_type: Type of pooling strategy (cls/mean/max/weighted)
        model_name: Name of the pretrained model
        model: The actual PyTorch model instance
        tokenizer: Tokenizer for text processing
    """
    
    DEFAULT_MODEL_NAME = "cointegrated/rubert-tiny2"

    def __init__(
        self,
        max_length: int = 512,
        batch_size: int = 128,
        learning_rate: float = 2e-5,
        num_epochs: int = 3,
        device: Optional[str] = None,
        use_cv: bool = False,
        pooling_type: str = "cls",
        model_name: Optional[str] = None
    ):
        """
        Initialize the embedder with configuration parameters.
        
        Args:
            max_length: Maximum sequence length for tokenization (default: 512)
            batch_size: Batch size for processing (default: 128)
            learning_rate: Learning rate for fine-tuning (default: 2e-5)
            num_epochs: Number of training epochs (default: 3)
            device: Device to use (cuda/cpu), auto-detected if None
            use_cv: Whether to use cross-validation (default: False)
            pooling_type: Pooling strategy (cls/mean/max/weighted) (default: cls)
            model_name: Optional custom model name (default: rubert-tiny2)
        """
        assert pooling_type in ["cls", "mean", "max", "weighted"], \
            f"pooling_type must be one of: cls, mean, max, weighted. Got {pooling_type}"

        self.max_length = max_length
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_cv = use_cv
        self.pooling_type = pooling_type
        self.model_name = model_name or self.DEFAULT_MODEL_NAME
        self.model = None
        self.tokenizer = None

    class _PricePredictionModel(nn.Module):
        """
        Internal PyTorch model combining BERT with a regression head.
        Implements different pooling strategies for embedding generation.
        """
        
        def __init__(self, model_name: str, pooling_type: str):
            """
            Initialize the internal model components.
            
            Args:
                model_name: Name of the pretrained BERT model
                pooling_type: Type of pooling strategy to use
            """
            super().__init__()
            self.bert = AutoModel.from_pretrained(model_name)
            self.pooling_type = pooling_type
            self.hidden_size = self.bert.config.hidden_size
            self.regressor = nn.Linear(self.hidden_size, 1)

            if self.pooling_type == "weighted":
                self.attention = nn.Sequential(
                    nn.Linear(self.hidden_size, self.hidden_size),
                    nn.Tanh(),
                    nn.Linear(self.hidden_size, 1)
                )

        def forward(self, input_ids, attention_mask):
            """
            Forward pass through the model.
            
            Args:
                input_ids: Tokenized input IDs
                attention_mask: Attention mask for the input
                
            Returns:
                Tuple of (regression output, pooled embeddings)
            """
            outputs = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=False,
                output_hidden_states=False
            )
            last_hidden = outputs.last_hidden_state

            if self.pooling_type == "cls":
                pooled = last_hidden[:, 0, :]
            elif self.pooling_type == "mean":
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
                sum_embeddings = torch.sum(last_hidden * input_mask_expanded, dim=1)
                sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                pooled = sum_embeddings / sum_mask
            elif self.pooling_type == "max":
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
                last_hidden[input_mask_expanded == 0] = -torch.inf
                pooled = torch.max(last_hidden, dim=1)[0]
            elif self.pooling_type == "weighted":
                weights = self.attention(last_hidden).squeeze(-1)
                weights = weights.masked_fill(attention_mask == 0, -torch.inf)
                weights = torch.softmax(weights, dim=1)
                pooled = torch.sum(last_hidden * weights.unsqueeze(-1), dim=1)

            return self.regressor(pooled), pooled

    class _TextPriceDataset(Dataset):
        """
        Internal PyTorch Dataset for handling text-price pairs.
        """
        
        def __init__(self, texts, prices, tokenizer, max_length):
            """
            Initialize the dataset.
            
            Args:
                texts: List of input texts
                prices: Corresponding prices/target values
                tokenizer: Tokenizer for text processing
                max_length: Maximum sequence length
            """
            self.texts = texts.tolist() if hasattr(texts, 'tolist') else list(texts)
            self.prices = prices.tolist() if hasattr(prices, 'tolist') else list(prices)
            self.tokenizer = tokenizer
            self.max_length = max_length

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            encoding = self.tokenizer(
                str(self.texts[idx]),
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            return {
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'price': torch.tensor(float(self.prices[idx]), dtype=torch.float32)
            }

    def fit(self, X, y=None):
        """
        Fit the model on training data.
        
        Args:
            X: Input texts
            y: Target values (optional for pure embedding generation)
            
        Returns:
            self: Fitted transformer instance
        """
        X = self._convert_to_list(X)
        y = self._convert_to_list(y) if y is not None else None

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = self._PricePredictionModel(
            model_name=self.model_name,
            pooling_type=self.pooling_type
        ).to(self.device)

        if y is not None:
            self._train_model(X, y)

        return self

    def get_token_importance(self, texts, top_n=20):
        """
        Get token/word importance scores for the input texts.
        
        Args:
            texts: Input texts to analyze
            top_n: Number of top tokens to return
            
        Returns:
            List of (token, weight) tuples for each input text
            
        Raises:
            ValueError: If model doesn't support token importance
        """
        if not hasattr(self, 'best_model'):
            raise ValueError("Model not trained yet")

        transformer = self._get_text_transformer()

        if isinstance(transformer, GatedTransformerWithTokenImportance):
            if isinstance(transformer.text_transformer, RuBertTiny2Embedder):
                return transformer.get_bert_token_importance(texts)
            else:
                return transformer.get_feature_importance(top_n)
        elif isinstance(transformer, RuBertTiny2Embedder):
            if getattr(transformer, "pooling_type", None) == "weighted":
                return transformer.get_token_importance(texts)
            else:
                raise ValueError("Token importance available only for weighted pooling (attention)")
        else:
            raise ValueError("Current model doesn't support token importance")

    def _convert_to_list(self, data):
        """
        Convert input data to list format.
        
        Args:
            data: Input data (pandas/numpy/list)
            
        Returns:
            List containing the data
        """
        if data is None:
            return None
        if hasattr(data, 'iloc'):
            return data.iloc[:, 0].tolist() if data.ndim > 1 else data.tolist()
        if hasattr(data, 'tolist'):
            return data.tolist()
        return list(data)

    def _train_model(self, X, y):
        """
        Internal training procedure for fine-tuning the model.
        
        Args:
            X: Training texts
            y: Training targets
        """
        dataset = self._TextPriceDataset(X, y, self.tokenizer, self.max_length)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        optimizer = AdamW(self.model.parameters(), lr=self.learning_rate)
        loss_fn = nn.MSELoss()

        self.model.train()
        for epoch in range(self.num_epochs):
            total_loss = 0
            for batch in dataloader:
                optimizer.zero_grad()
                inputs = {k: v.to(self.device) for k, v in batch.items()}
                predictions, _ = self.model(inputs['input_ids'], inputs['attention_mask'])
                loss = loss_fn(predictions.squeeze(), inputs['price'])
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            print(f"Epoch {epoch+1}/{self.num_epochs} - Loss: {total_loss/len(dataloader):.4f}")

    def transform(self, X):
        """
        Generate embeddings for input texts.
        
        Args:
            X: Input texts to transform
            
        Returns:
            numpy.ndarray: Generated embeddings
            
        Raises:
            RuntimeError: If model hasn't been trained
        """
        X = self._convert_to_list(X)

        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not trained. Call fit() first.")

        dataset = self._TextPriceDataset(X, [0]*len(X), self.tokenizer, self.max_length)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        self.model.eval()
        embeddings = []
        with torch.no_grad():
            for batch in dataloader:
                inputs = {k: v.to(self.device) for k, v in batch.items()}
                _, emb = self.model(inputs['input_ids'], inputs['attention_mask'])
                embeddings.append(emb.cpu().numpy())

        return np.concatenate(embeddings, axis=0)

    def save(self, path: str):
        """
        Save the model to a directory.
        
        Args:
            path: Directory path to save the model
        """
        os.makedirs(path, exist_ok=True)

        self.model.bert.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

        config = {
            'max_length': self.max_length,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'num_epochs': self.num_epochs,
            'device': str(self.device),
            'use_cv': self.use_cv,
            'pooling_type': self.pooling_type,
            'model_name': self.model_name
        }
        with open(os.path.join(path, 'config.json'), 'w') as f:
            json.dump(config, f, indent=2)

        state_dict = {
            'regressor_state_dict': self.model.regressor.state_dict()
        }
        if self.pooling_type == "weighted":
            state_dict['attention_state_dict'] = self.model.attention.state_dict()

        torch.save(state_dict, os.path.join(path, 'head_weights.pt'))

    @classmethod
    def load(cls, path: str):
        """
        Load a saved model from directory.
        
        Args:
            path: Directory path containing the saved model
            
        Returns:
            RuBertTiny2Embedder: Loaded model instance
        """
        with open(os.path.join(path, 'config.json')) as f:
            config = json.load(f)

        instance = cls(
            max_length=config['max_length'],
            batch_size=config['batch_size'],
            learning_rate=config['learning_rate'],
            num_epochs=config['num_epochs'],
            device=config['device'],
            use_cv=config.get('use_cv', False),
            pooling_type=config['pooling_type'],
            model_name=config['model_name']
        )

        instance.tokenizer = AutoTokenizer.from_pretrained(path)
        instance.model = instance._PricePredictionModel(
            model_name=path,
            pooling_type=config['pooling_type']
        ).to(instance.device)

        head_weights = torch.load(
            os.path.join(path, 'head_weights.pt'),
            map_location=instance.device
        )
        instance.model.regressor.load_state_dict(head_weights['regressor_state_dict'])
        if config['pooling_type'] == "weighted":
            instance.model.attention.load_state_dict(head_weights['attention_state_dict'])

        return instance
