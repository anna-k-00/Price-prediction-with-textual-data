import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.sparse import issparse


class ANNRegressor(BaseEstimator, TransformerMixin):
    """Artificial Neural Network Regressor compatible with scikit-learn API."""
    def __init__(self, neurons_layer1=128, neurons_layer2=64, neurons_layer3=32,
                 learning_rate=0.001, batch_size=64, epochs=100,
                 l1_reg=0.0, l2_reg=0.0, activation='relu',
                 dropout_rate=0.2, early_stopping_patience=5,
                 device=None, verbose=0):
        """
        Initialize the ANN regressor.
        
        Args:
            neurons_layer1: Number of neurons in first hidden layer
            neurons_layer2: Number of neurons in second hidden layer
            neurons_layer3: Number of neurons in third hidden layer
            learning_rate: Learning rate for optimizer
            batch_size: Training batch size
            epochs: Maximum number of training epochs
            l1_reg: L1 regularization strength
            l2_reg: L2 regularization strength
            activation: Activation function ('relu' or 'sigmoid')
            dropout_rate: Dropout rate between layers
            early_stopping_patience: Patience for early stopping
            device: Device to run computations on
            verbose: Verbosity level
        """
        self.neurons_layer1 = neurons_layer1
        self.neurons_layer2 = neurons_layer2
        self.neurons_layer3 = neurons_layer3
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.early_stopping_patience = early_stopping_patience
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.verbose = verbose
        self.model = None

    class _ANN(nn.Module):
        """Internal neural network architecture implementation."""
        def __init__(self, input_size, cfg):
            """
            Initialize the neural network layers.
            
            Args:
                input_size: Dimension of input features
                cfg: Configuration dictionary containing layer parameters
            """
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
            """
            Get activation function module.
            
            Args:
                activation: Name of activation function ('relu' or 'sigmoid')
                
            Returns:
                Corresponding PyTorch activation module
            """
            return nn.ReLU() if activation == 'relu' else nn.Sigmoid()

        def forward(self, x):
            """Forward pass through the network."""
            return self.layers(x).squeeze()

    def _convert_to_tensor(self, X):
        """
        Convert input data to PyTorch tensor with sparse matrix handling.
        
        Args:
            X: Input data (array, sparse matrix, DataFrame or Series)
            
        Returns:
            PyTorch FloatTensor on configured device
        """
        if issparse(X):
            X = X.toarray()
        elif isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
            X = X.values
        return torch.FloatTensor(X).to(self.device)

    def fit(self, X, y):
        """
        Train the neural network on input data.
        
        Args:
            X: Training features
            y: Training targets
            
        Returns:
            self
        """
        X_tensor = self._convert_to_tensor(X)
        y_tensor = self._convert_to_tensor(y)

        model_cfg = {
            'neurons_layer1': self.neurons_layer1,
            'neurons_layer2': self.neurons_layer2,
            'neurons_layer3': self.neurons_layer3,
            'dropout_rate': self.dropout_rate,
            'activation': self.activation
        }

        self.model = self._ANN(X_tensor.shape[1], model_cfg).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        loss_fn = nn.MSELoss()

        best_loss = float('inf')
        patience_counter = 0

        for epoch in range(self.epochs):
            self.model.train()
            epoch_loss = 0

            for i in range(0, len(X_tensor), self.batch_size):
                batch_X = X_tensor[i:i+self.batch_size]
                batch_y = y_tensor[i:i+self.batch_size]

                optimizer.zero_grad()
                predictions = self.model(batch_X)
                loss = loss_fn(predictions, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / (len(X_tensor) / self.batch_size)

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
        """
        Make predictions using the trained model.
        
        Args:
            X: Input features for prediction
            
        Returns:
            Numpy array of predictions
        """
        X_tensor = self._convert_to_tensor(X)
        self.model.eval()
        with torch.no_grad():
            return self.model(X_tensor).cpu().numpy()
