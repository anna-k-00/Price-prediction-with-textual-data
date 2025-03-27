import warnings
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer, AdamW
from sklearn.base import BaseEstimator, TransformerMixin
from typing import List, Optional

warnings.filterwarnings('ignore')

class RuBertTiny2Embedder(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        max_length: int = 512,
        batch_size: int = 128,
        num_epochs: int = 5,
        n_splits: Optional[int] = None,
        learning_rate: float = 2e-5,
        sample_size: Optional[int] = None,
        device: Optional[str] = None,
        use_cv: bool = False
    ):
        """
        Initialize the rubert-tiny2 embedder with configurable parameters.
        
        Args:
            max_length: Maximum sequence length
            batch_size: Training and inference batch size
            num_epochs: Number of training epochs
            n_splits: Number of cross-validation splits (None when use_cv=False)
            learning_rate: Learning rate for optimizer
            sample_size: Optional subsample size for quick testing
            device: Device to use ('cuda' or 'cpu'), auto-detects if None
            use_cv: Whether to perform internal cross-validation
        """
        self.max_length = max_length
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.n_splits = n_splits if use_cv else None
        self.learning_rate = learning_rate
        self.sample_size = sample_size
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_cv = use_cv
        
        self.model = None
        self.tokenizer = None
        self.cv_scores = None
    
    class _TextPriceDataset(Dataset):
        """Internal dataset class for text-price pairs"""
        def __init__(self, texts, prices, tokenizer, max_length):
            self.texts = texts
            self.prices = prices
            self.tokenizer = tokenizer
            self.max_length = max_length
            
        def __len__(self):
            return len(self.texts)
        
        def __getitem__(self, idx):
            text = str(self.texts[idx])
            price = self.prices[idx]
            
            encoding = self.tokenizer(
                text,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            return {
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'price': torch.tensor(price, dtype=torch.float32)
            }
    
    class _PricePredictionModel(nn.Module):
        """Internal model with regression head"""
        def __init__(self, base_model_name: str = "cointegrated/rubert-tiny2", dropout_rate: float = 0.1):
            super().__init__()
            self.bert = AutoModel.from_pretrained(base_model_name)
            self.dropout = nn.Dropout(dropout_rate)
            self.regressor = nn.Linear(self.bert.config.hidden_size, 1)
            
        def forward(self, input_ids, attention_mask):
            outputs = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            pooled_output = outputs.last_hidden_state[:, 0, :]  # Using [CLS] token
            pooled_output = self.dropout(pooled_output)
            return self.regressor(pooled_output), pooled_output
    
    def _train_epoch(self, model, dataloader, optimizer, loss_fn):
        """Internal training loop for one epoch"""
        model.train()
        total_loss = 0
        
        for batch in dataloader:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            prices = batch['price'].to(self.device)
            
            optimizer.zero_grad()
            predictions, _ = model(input_ids, attention_mask)
            
            loss = loss_fn(predictions.squeeze(), prices)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(dataloader)
    
    def fit(self, X: List[str], y: List[float]):
        """
        Fine-tune the model on text-price pairs.
        
        Args:
            X: List of training texts
            y: Corresponding list of prices/target values
        """
        if self.sample_size and len(X) > self.sample_size:
            indices = np.random.choice(len(X), self.sample_size, replace=False)
            X = [X[i] for i in indices]
            y = [y[i] for i in indices]
        
        model_name = "cointegrated/rubert-tiny2"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = self._PricePredictionModel(model_name).to(self.device)
        
        if self.use_cv and self.n_splits and self.n_splits > 1:
            return self._fit_with_cv(X, y)
        else:
            return self._fit_direct(X, y)
    
    def _fit_direct(self, X, y):
        """Train without cross-validation"""
        dataset = self._TextPriceDataset(X, y, self.tokenizer, self.max_length)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        optimizer = AdamW(self.model.parameters(), lr=self.learning_rate)
        loss_fn = nn.MSELoss()
        
        for epoch in range(self.num_epochs):
            train_loss = self._train_epoch(self.model, dataloader, optimizer, loss_fn)
            print(f"Epoch {epoch+1}/{self.num_epochs} - Train Loss: {train_loss:.4f}")
        
        return self
    
    def _fit_with_cv(self, X, y):
        """Train with cross-validation (standalone mode)"""
        from sklearn.model_selection import KFold
        
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        self.cv_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            print(f"\n=== Fold {fold + 1}/{self.n_splits} ===")
            
            train_texts = [X[i] for i in train_idx]
            train_prices = [y[i] for i in train_idx]
            val_texts = [X[i] for i in val_idx]
            val_prices = [y[i] for i in val_idx]
            
            # Reinitialize model for each fold
            model = self._PricePredictionModel().to(self.device)
            optimizer = AdamW(model.parameters(), lr=self.learning_rate)
            loss_fn = nn.MSELoss()
            
            train_dataset = self._TextPriceDataset(train_texts, train_prices, self.tokenizer, self.max_length)
            val_dataset = self._TextPriceDataset(val_texts, val_prices, self.tokenizer, self.max_length)
            
            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
            
            best_val_loss = float('inf')
            
            for epoch in range(self.num_epochs):
                train_loss = self._train_epoch(model, train_loader, optimizer, loss_fn)
                val_loss = self._evaluate(model, val_loader, loss_fn)
                
                print(f"Epoch {epoch+1}/{self.num_epochs} - Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(model.state_dict(), f"best_model_fold{fold}.pt")
            
            self.cv_scores.append(best_val_loss)
            print(f"Fold {fold+1} Best Val Loss: {best_val_loss:.4f}")
        
        # Train final model on all data
        print("\nTraining final model on all data...")
        return self._fit_direct(X, y)
    
    def _evaluate(self, model, dataloader, loss_fn):
        """Evaluate model on validation set"""
        model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                prices = batch['price'].to(self.device)
                
                predictions, _ = model(input_ids, attention_mask)
                total_loss += loss_fn(predictions.squeeze(), prices).item()
        
        return total_loss / len(dataloader)
    
    def transform(self, X: List[str]) -> np.ndarray:
        """
        Generate embeddings for new texts using the fine-tuned model.
        
        Args:
            X: List of texts to embed
            
        Returns:
            Numpy array of embeddings (shape [n_samples, hidden_size])
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model has not been trained. Call fit() first.")
        
        dataset = self._TextPriceDataset(X, [0]*len(X), self.tokenizer, self.max_length)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        
        self.model.eval()
        all_embeddings = []
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                _, embeddings = self.model(input_ids, attention_mask)
                all_embeddings.append(embeddings.cpu().numpy())
        
        return np.concatenate(all_embeddings, axis=0)
