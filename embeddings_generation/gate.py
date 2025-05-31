import numpy as np
import torch
import torch.nn as nn
from scipy.sparse import issparse
from sklearn.base import BaseEstimator, TransformerMixin
from typing import Dict, List, Union, Optional
from functools import lru_cache
import logging
import torch
import torch.nn as nn
import numpy as np
from typing import Optional
from copy import deepcopy
import torch.nn.functional as F

from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import json
import logging
from copy import deepcopy
from typing import Dict, List, Union, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse import issparse
from sklearn.base import BaseEstimator, TransformerMixin
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer


class HiddenBertDataset(Dataset):
    """Dataset class for storing BERT hidden states, attention masks and targets."""
    def __init__(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor, targets):
        """
        Initialize dataset with BERT hidden states, attention masks and targets.
        
        Args:
            hidden_states: Tensor of shape [N, L, H] containing hidden states
            attention_mask: Tensor of shape [N, L] containing attention masks (0/1)
            targets: Array-like of length N containing target values
        """
        self.hidden_states = hidden_states
        self.attention_mask = attention_mask
        self.targets = torch.FloatTensor(targets)

    def __len__(self):
        """Return number of samples in dataset."""
        return self.hidden_states.size(0)

    def __getitem__(self, idx):
        """Get single sample by index."""
        return (
            self.hidden_states[idx],
            self.attention_mask[idx],
            self.targets[idx],
        )


class EarlyStopping:
    """Early stopping mechanism based on validation loss."""
    def __init__(self, patience: int = 5, min_delta: float = 1e-4):
        """
        Initialize early stopping tracker.
        
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum improvement required to reset patience counter
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False

    def __call__(self, val_loss: float) -> bool:
        """
        Update early stopping state based on current validation loss.
        
        Args:
            val_loss: Current validation loss value
            
        Returns:
            True if training should stop, False otherwise
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop


class TokenContextGate(nn.Module):
    """Neural gate that combines per-token scores with global context scores."""
    def __init__(self, hidden_size: int):
        """
        Initialize token context gate.
        
        Args:
            hidden_size: Dimension of input hidden states
        """
        super().__init__()
        self.token_fc = nn.Linear(hidden_size, 1)
        self.context_fc = nn.Linear(hidden_size, 1)

    def forward(self, hs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the gate network.
        
        Args:
            hs: Input hidden states of shape [B, L, H]
            
        Returns:
            Combined token and context scores of shape [B, L, 1]
        """
        tok_scores = self.token_fc(hs)          # [B, L, 1]
        cls_vec    = hs[:, 0, :]               # [B, H]
        ctx_scores = self.context_fc(cls_vec)  # [B, 1]
        ctx_scores = ctx_scores.unsqueeze(1)   # [B, 1, 1]
        return tok_scores + ctx_scores        # [B, L, 1]


class GatedTransformerWithTokenImportance(BaseEstimator, TransformerMixin):
    """Transformer with learnable gating mechanism for token/feature importance."""
    def __init__(
        self,
        text_transformer,
        gate_threshold: float = 0.5,
        use_gate: bool = True,
        hidden_dim: int = 64,
        device: Optional[str] = None,
        ngram_range: tuple = (1, 1),
        tokenizer_params: Optional[dict] = None,
        bert_aggregation: str = 'mean',
        gate_epochs: int = 50,
        gate_lr: float = 1e-3,
        gate_mode: str = 'soft',
        l1_reg: float = 0.01,
        bert_batch_size: Optional[int] = None,
        gate_batch_size: Optional[int] = None
    ):
        """
        Initialize gated transformer.
        
        Args:
            text_transformer: Base text transformer model
            gate_threshold: Threshold for gating mechanism
            use_gate: Whether to use gating mechanism
            hidden_dim: Dimension of hidden layers in gate network
            device: Device to run computations on
            ngram_range: Range of n-grams to consider
            tokenizer_params: Parameters for tokenizer
            bert_aggregation: Method for aggregating BERT outputs
            gate_epochs: Number of epochs to train gate
            gate_lr: Learning rate for gate training
            gate_mode: Type of gating ('soft' or 'hard')
            l1_reg: L1 regularization strength
            bert_batch_size: Batch size for BERT processing
            gate_batch_size: Batch size for gate training
        """
        super().__init__()
        self.text_transformer = text_transformer
        self.gate_threshold = gate_threshold
        self.use_gate = use_gate
        self.hidden_dim = hidden_dim
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.ngram_range = ngram_range
        self.tokenizer_params = tokenizer_params or {}
        self.bert_aggregation = bert_aggregation
        self.gate_epochs = gate_epochs
        self.gate_lr = gate_lr
        self.gate_mode = gate_mode
        self.l1_reg = l1_reg

        self.bert_batch_size = bert_batch_size or getattr(text_transformer, 'batch_size', 32)
        self.gate_batch_size = gate_batch_size or 32

        self._init_logger()
        if hasattr(text_transformer, 'model'):
            self._init_bert_tokenizer()

    def _init_logger(self):
        """Initialize logger for the class."""
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)

    def _init_bert_tokenizer(self):
        """Initialize BERT tokenizer if available."""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                "cointegrated/rubert-tiny2", **self.tokenizer_params
            )
        except ImportError:
            self.logger.warning("transformers not found; BERT disabled")

    def _get_embedding_dim(self, X):
        """Get output dimension of the base transformer."""
        if hasattr(self.text_transformer, 'output_dim'):
            return self.text_transformer.output_dim
        if hasattr(self.text_transformer, 'n_features_in_'):
            return self.text_transformer.n_features_in_
        sample = self.text_transformer.transform(X[:1])
        if issparse(sample):
            sample = sample.toarray()
        return sample.shape[1]

    def _get_base_embeddings(self, X):
        """Get base embeddings from text transformer."""
        if hasattr(self.text_transformer, 'model') and hasattr(self.text_transformer.model, 'bert'):
            raise RuntimeError("For BERT, call fit() once and use _bert_hidden_states")
        emb = self.text_transformer.transform(X)
        if issparse(emb):
            emb = emb.toarray()
        return torch.FloatTensor(emb).to(self.device)

    def fit(self, X, y=None):
        """
        Fit the gated transformer model.
        
        Args:
            X: Input data
            y: Target values
            
        Returns:
            self
        """
        self.text_transformer.fit(X, y)
        if not self.use_gate or y is None:
            return self

        if hasattr(self.text_transformer, "model") and hasattr(self, 'tokenizer'):
            all_hs, all_masks = [], []
            for i in range(0, len(X), self.bert_batch_size):
                batch = list(X[i : i + self.bert_batch_size])
                enc = self.tokenizer(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=128,
                )
                enc = {k: v.to(self.device) for k, v in enc.items()}

                with torch.no_grad():
                    hs = self.text_transformer.model.bert(
                        input_ids=enc["input_ids"],
                        attention_mask=enc["attention_mask"],
                    ).last_hidden_state

                all_hs.append(hs.cpu())
                all_masks.append(enc["attention_mask"].cpu())
                del hs, enc
                torch.cuda.empty_cache()

            self._bert_hidden_states = torch.cat(all_hs, dim=0)
            self._bert_attention_mask = torch.cat(all_masks, dim=0)

            self._train_gate_bert(
                self._bert_hidden_states,
                self._bert_attention_mask,
                y,
            )
        else:
            emb = self._get_base_embeddings(X)
            self._train_gate_vector(emb, y)

        return self

    def _train_gate_bert(self, hidden_states, attention_mask, y):
        """Train BERT-based gating mechanism."""
        dataset = HiddenBertDataset(hidden_states, attention_mask, y)
        loader  = DataLoader(dataset, batch_size=self.gate_batch_size, shuffle=True)
        N, L, H = hidden_states.shape

        self.gate_token_net = TokenContextGate(H).to(self.device)
        self.regressor      = nn.Linear(H, 1).to(self.device)

        optimizer = torch.optim.AdamW(
            list(self.gate_token_net.parameters()) +
            list(self.regressor.parameters()),
            lr=self.gate_lr
        )

        for epoch in range(self.gate_epochs):
            epoch_loss = total_active = total_tokens = 0
            for hs_b, mask_b, y_b in loader:
                hs_b = hs_b.to(self.device)
                mask_b = mask_b.to(self.device)
                y_b    = y_b.to(self.device)

                scores = self.gate_token_net(hs_b)
                masks  = torch.sigmoid(scores) * mask_b.unsqueeze(-1)
                weights = masks.squeeze(-1)
                norm = weights.sum(1, keepdim=True) + 1e-8
                weights = (weights / norm).unsqueeze(-1)

                rep = (hs_b * weights).sum(dim=1)
                pred = self.regressor(rep).squeeze(-1)

                loss = F.mse_loss(pred, y_b) + self.l1_reg * masks.mean()

                total_active += (masks > 0.5).float().sum().item()
                total_tokens += mask_b.sum().item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * hs_b.size(0)

            avg_loss = epoch_loss / len(dataset)
            pct_active = total_active / total_tokens * 100
            print(f"[Gate BERT] Epoch {epoch:03d} | "
                  f"Loss={avg_loss:.4f} | Active tokens={pct_active:.2f}%")

    def _train_gate_vector(self, embeddings: torch.Tensor, y):
        """Train vector-based gating mechanism."""
        N, D = embeddings.shape
        self.gate_net = nn.Sequential(
            nn.Linear(D, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, D),
            nn.Sigmoid()
        ).to(self.device)

        optimizer = torch.optim.AdamW(self.gate_net.parameters(), lr=self.gate_lr)
        y_t = torch.FloatTensor(y).to(self.device)
        loss_fn = nn.MSELoss()

        for epoch in range(self.gate_epochs):
            optimizer.zero_grad()
            masks = self.gate_net(embeddings)
            gated = embeddings * masks
            rep = gated.mean(dim=1)
            loss = loss_fn(rep, y_t) + self.l1_reg * masks.mean()
            loss.backward()
            optimizer.step()
            print(f"[Gate Vector] Epoch {epoch:03d} | Loss={loss.item():.4f}")

        self.gate_weights_ = self.gate_net[-1].weight.data.clone()

    def transform(self, X, mode: Optional[str] = None) -> np.ndarray:
        """
        Transform input data using gated mechanism.
        
        Args:
            X: Input data to transform
            mode: Gating mode ('soft' or 'hard')
            
        Returns:
            Transformed embeddings as numpy array
        """
        if not self.use_gate:
            emb = self._get_base_embeddings(X)
            return emb.cpu().numpy()

        if hasattr(self, 'tokenizer') and hasattr(self.text_transformer, 'model'):
            return self._transform_bert_batched(X)

        emb = self._get_base_embeddings(X)
        return self._transform_vector(emb, mode or self.gate_mode)

    def _transform_bert_batched(self, X_texts: Union[List[str], np.ndarray]) -> np.ndarray:
        """Transform text data using BERT-based gating in batches."""
        reps = []
        for i in range(0, len(X_texts), self.bert_batch_size):
            chunk = list(X_texts[i : i + self.bert_batch_size])
            enc = self.tokenizer(
                chunk,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128,
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}

            with torch.no_grad():
                hs = self.text_transformer.model.bert(
                    input_ids=enc["input_ids"],
                    attention_mask=enc["attention_mask"],
                ).last_hidden_state

                scores = self.gate_token_net(hs)
                masks  = torch.sigmoid(scores) * enc["attention_mask"].unsqueeze(-1)
                weights = masks.squeeze(-1)
                norm = weights.sum(1, keepdim=True) + 1e-8
                weights = (weights / norm).unsqueeze(-1)

                rep = (hs * weights).sum(dim=1)

            reps.append(rep.cpu().numpy())
            del hs, scores, masks, weights, rep, enc
            torch.cuda.empty_cache()

        return np.vstack(reps)

    def _transform_vector(self, embeddings: torch.Tensor, mode: str) -> np.ndarray:
        """Transform vector embeddings using gating mechanism."""
        masks = self.gate_net(embeddings)
        if mode == 'adaptive':
            thr = torch.quantile(masks, 0.7)
            mask = (masks > thr).float()
            mask += 0.2 * (masks > thr / 2).float()
        else:
            mask = torch.sigmoid((masks - self.gate_threshold) * 10)
        gated = embeddings * mask
        return gated.cpu().numpy()

    def get_bert_token_importance(
        self,
        texts: List[str],
        mode: str = 'soft'
    ) -> List[List[tuple]]:
        """
        Get token importance scores for input texts.
        
        Args:
            texts: List of input texts
            mode: Scoring mode
            
        Returns:
            List of (token, score) pairs for each text
        """
        if not hasattr(self, "gate_token_net"):
            raise RuntimeError("Gate not trained: call fit() first")
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not initialized")

        enc = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}

        with torch.no_grad():
            hs = self.text_transformer.model.bert(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
            ).last_hidden_state
            scores = self.gate_token_net(hs).squeeze(-1) * enc["attention_mask"].float()

        batch_tokens = [
            self.tokenizer.convert_ids_to_tokens(ids)
            for ids in enc["input_ids"]
        ]
        batch_scores = scores.cpu().numpy().tolist()

        return [list(zip(toks, sc)) for toks, sc in zip(batch_tokens, batch_scores)]

    def get_feature_importance(self, top_n: int = 20, mode: str = 'soft') -> Dict[str, float]:
        """
        Get feature importance scores.
        
        Args:
            top_n: Number of top features to return
            mode: Scoring mode ('soft' or 'hard')
            
        Returns:
            Dictionary of {feature: importance_score}
        """
        if not hasattr(self.text_transformer, 'get_feature_names_out'):
            raise ValueError("Only for vectorizer-based transformers")
        if not hasattr(self, 'gate_weights_'):
            raise RuntimeError("Call fit() first")

        vocab = self.text_transformer.get_feature_names_out()
        if mode == 'hard':
            weights = (self.gate_weights_ > self.gate_threshold).float()
        else:
            weights = torch.sigmoid(self.gate_weights_)

        items = [
            (ph, float(weights[i]))
            for i, ph in enumerate(vocab)
            if self.ngram_range[0] <= len(ph.split()) <= self.ngram_range[1]
        ]
        items = sorted(items, key=lambda x: abs(x[1]), reverse=True)
        return dict(items[:top_n])

    def save(self, path: str):
        """
        Save model to disk.
        
        Args:
            path: Directory path to save model to
        """
        import os
        import json
        import joblib

        print(f"\n[SAVE][GATE] >>> Saving to {path}")
        os.makedirs(path, exist_ok=True)

        params = {
            'gate_threshold':   self.gate_threshold,
            'use_gate':         self.use_gate,
            'hidden_dim':       self.hidden_dim,
            'device':           self.device,
            'ngram_range':      list(self.ngram_range),
            'tokenizer_params': self.tokenizer_params,
            'bert_aggregation': self.bert_aggregation,
            'gate_epochs':      self.gate_epochs,
            'gate_lr':          self.gate_lr,
            'gate_mode':        self.gate_mode,
            'l1_reg':           self.l1_reg,
        }
        with open(os.path.join(path, 'params.json'), 'w') as f:
            json.dump(params, f)
        print(f"[SAVE][GATE] params.json saved: {os.path.join(path, 'params.json')}")

        text_tr_path = os.path.join(path, 'text_transformer')
        if hasattr(self.text_transformer, 'save'):
            print(f"[SAVE][GATE] Saving text_transformer using .save() → {text_tr_path}")
            self.text_transformer.save(text_tr_path)
        else:
            print(f"[SAVE][GATE] Saving text_transformer via joblib → {text_tr_path}.pkl")
            joblib.dump(self.text_transformer, text_tr_path + '.pkl')

        if not self.use_gate:
            print(f"[SAVE][GATE] use_gate=False — not saving gate weights")
            return

        if hasattr(self, 'gate_net'):
            print(f"[SAVE][GATE] gate_net found — saving vector gate")
            torch.save({
                'type': 'vector',
                'gate_net_state': self.gate_net.state_dict(),
                'gate_weights':   self.gate_weights_
            }, os.path.join(path, 'gate_net.pt'))
            print(f"[SAVE][GATE] gate_net.pt saved: {os.path.join(path, 'gate_net.pt')}")
        elif hasattr(self, 'gate_token_net'):
            print(f"[SAVE][GATE] gate_token_net found — saving bert gate")
            torch.save({
                'type': 'bert',
                'gate_token_net_state': self.gate_token_net.state_dict(),
                'regressor_state': self.regressor.state_dict()
            }, os.path.join(path, 'gate_token_net.pt'))
            print(f"[SAVE][GATE] gate_token_net.pt saved: {os.path.join(path, 'gate_token_net.pt')}")
        else:
            print("[SAVE][GATE] WARNING: gate_net/gate_token_net not found!")

        if hasattr(self, 'tokenizer') and self.tokenizer is not None:
            tok_path = os.path.join(path, 'tokenizer')
            os.makedirs(tok_path, exist_ok=True)
            self.tokenizer.save_pretrained(tok_path)
            print(f"[SAVE][GATE] tokenizer saved: {tok_path}")
        else:
            print(f"[SAVE][GATE] Tokenizer not found — not saving.")

        print(f"\n[SAVE][GATE] Contents of {path}:")
        for root, dirs, files in os.walk(path):
            print(root, "dirs:", dirs, "files:", files)

    @classmethod
    def load(cls, path: str):
        """
        Load model from disk.
        
        Args:
            path: Directory path containing saved model
            
        Returns:
            Loaded GatedTransformerWithTokenImportance instance
        """
        import os
        import joblib
        import json
        from transformers import AutoTokenizer

        with open(os.path.join(path, 'params.json'), 'r') as f:
            params = json.load(f)

        tf_dir = os.path.join(path, 'text_transformer')
        if os.path.isdir(tf_dir):
            text_transformer = RuBertTiny2Embedder.load(tf_dir)
        elif os.path.exists(tf_dir + '.pkl'):
            import joblib
            text_transformer = joblib.load(tf_dir + '.pkl')
        else:
            raise FileNotFoundError(f"text_transformer not found in {tf_dir} or {tf_dir + '.pkl'}")

        inst = cls(
            text_transformer=text_transformer,
            gate_threshold=params['gate_threshold'],
            use_gate=params['use_gate'],
            hidden_dim=params['hidden_dim'],
            device=params['device'],
            ngram_range=tuple(params['ngram_range']),
            tokenizer_params=params['tokenizer_params'],
            bert_aggregation=params['bert_aggregation'],
            gate_epochs=params['gate_epochs'],
            gate_lr=params['gate_lr'],
            gate_mode=params.get('gate_mode', 'soft'),
            l1_reg=params.get('l1_reg', 0.01),
        )

        if not inst.use_gate:
            return inst

        bert_gate_path = os.path.join(path, 'gate_token_net.pt')
        vector_gate_path = os.path.join(path, 'gate_net.pt')

        if os.path.exists(bert_gate_path):
            checkpoint = torch.load(bert_gate_path, map_location=inst.device)
            H = inst._get_embedding_dim(["example"])
            inst.gate_token_net = TokenContextGate(H).to(inst.device)
            inst.regressor = nn.Linear(H, 1).to(inst.device)
            inst.gate_token_net.load_state_dict(checkpoint['gate_token_net_state'])
            inst.regressor.load_state_dict(checkpoint['regressor_state'])
        elif os.path.exists(vector_gate_path):
            checkpoint = torch.load(vector_gate_path, map_location=inst.device)
            inst.embedding_dim_ = inst._get_embedding_dim(["example"])
            inst.gate_net = nn.Sequential(
                nn.Linear(inst.embedding_dim_, inst.hidden_dim),
                nn.ReLU(),
                nn.Linear(inst.hidden_dim, inst.embedding_dim_),
                nn.Sigmoid()
            ).to(inst.device)
            inst.gate_net.load_state_dict(checkpoint['gate_net_state'])
            inst.gate_weights_ = checkpoint['gate_weights']

        tok_dir = os.path.join(path, 'tokenizer')
        if os.path.isdir(tok_dir):
            inst.tokenizer = AutoTokenizer.from_pretrained(tok_dir)

        return inst
