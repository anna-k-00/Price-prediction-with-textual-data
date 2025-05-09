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
    def __init__(self, max_length=512, batch_size=128, learning_rate=2e-5, device=None):
        self.max_length = max_length
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.tokenizer = None

    class _TextPriceDataset(Dataset):
        __slots__ = ['texts', 'prices', 'tokenizer', 'max_length']  # Уменьшаем накладные расходы
        
        def __init__(self, texts, prices, tokenizer, max_length):
            # Быстрое преобразование без лишних проверок (предполагаем правильный вход)
            self.texts = texts.values if hasattr(texts, 'values') else texts
            self.prices = prices.values if hasattr(prices, 'values') else prices
            self.tokenizer = tokenizer
            self.max_length = max_length
        
        def __len__(self):
            return len(self.texts)
        
        def __getitem__(self, idx):
            # Минимально необходимые преобразования
            return {
                'input_ids': self.tokenizer(
                    str(self.texts[idx]),
                    max_length=self.max_length,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )['input_ids'].flatten(),
                'attention_mask': self.tokenizer(
                    str(self.texts[idx]),
                    max_length=self.max_length,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )['attention_mask'].flatten(),
                'price': torch.tensor(float(self.prices[idx]), dtype=torch.float32)
            }

    def fit(self, X, y=None):
        # Быстрая проверка и преобразование
        if hasattr(X, 'iloc'):
            X = X.iloc[:, 0] if X.ndim > 1 else X
        
        self.tokenizer = AutoTokenizer.from_pretrained("cointegrated/rubert-tiny2")
        self.model = self._PricePredictionModel().to(self.device)
        
        if y is not None:
            dataset = self._TextPriceDataset(X, y, self.tokenizer, self.max_length)
            dataloader = DataLoader(dataset, 
                                  batch_size=self.batch_size, 
                                  shuffle=True,
                                  pin_memory=True)  # Ускоряет передачу на GPU
            
            optimizer = AdamW(self.model.parameters(), lr=self.learning_rate)
            loss_fn = nn.MSELoss()
            
            # Одного эпоха обычно достаточно для фича-экстрактора
            self._train_epoch(self.model, dataloader, optimizer, loss_fn)
        
        return self

    def transform(self, X):
        if hasattr(X, 'iloc'):
            X = X.iloc[:, 0] if X.ndim > 1 else X
        
        dataset = self._TextPriceDataset(X, [0]*len(X), self.tokenizer, self.max_length)
        dataloader = DataLoader(dataset, 
                              batch_size=self.batch_size, 
                              shuffle=False,
                              pin_memory=True)
        
        self.model.eval()
        all_embeddings = []
        
        with torch.no_grad():
            for batch in dataloader:
                inputs = {k: v.to(self.device) for k, v in batch.items() 
                         if k in ['input_ids', 'attention_mask']}
                _, embeddings = self.model(**inputs)
                all_embeddings.append(embeddings.cpu().numpy())
        
        return np.concatenate(all_embeddings, axis=0)
    
    
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
