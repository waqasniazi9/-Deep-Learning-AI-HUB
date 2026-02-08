"""
Data loading and preprocessing utilities for efficient ML pipeline.
Handles datasets, preprocessing, and data augmentation.
"""

import logging
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.datasets import load_iris, load_digits, make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Dict, Tuple, List, Optional
from utils import memoize_with_ttl

logger = logging.getLogger(__name__)


class TensorDataset(Dataset):
    """Custom PyTorch Dataset for tensor data."""

    def __init__(self, X: np.ndarray, y: np.ndarray, transform=None):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.X[idx]
        y = self.y[idx]

        if self.transform:
            x = self.transform(x)

        return x, y


class DataManager:
    """Manages dataset loading, preprocessing, and batching."""

    def __init__(self, batch_size: int = 32, test_size: float = 0.2, random_state: int = 42):
        self.batch_size = batch_size
        self.test_size = test_size
        self.random_state = random_state
        self.scaler = StandardScaler()
        self._cached_datasets = {}

    @memoize_with_ttl(ttl_seconds=3600)
    def get_iris(self, split: bool = True) -> Dict:
        """Load and preprocess Iris dataset."""
        logger.info("Loading Iris dataset")
        iris = load_iris()
        X, y = iris.data, iris.target

        X = self.scaler.fit_transform(X)

        if split:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.test_size, random_state=self.random_state
            )

            train_dataset = TensorDataset(X_train, y_train)
            test_dataset = TensorDataset(X_test, y_test)

            return {
                'train_loader': DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True),
                'test_loader': DataLoader(test_dataset, batch_size=self.batch_size),
                'train_X': X_train, 'train_y': y_train,
                'test_X': X_test, 'test_y': y_test,
                'input_size': X.shape[1],
                'num_classes': len(np.unique(y))
            }

        return {'X': X, 'y': y, 'input_size': X.shape[1], 'num_classes': len(np.unique(y))}

    @memoize_with_ttl(ttl_seconds=3600)
    def get_digits(self, split: bool = True) -> Dict:
        """Load and preprocess Digits dataset."""
        logger.info("Loading Digits dataset")
        digits = load_digits()
        X, y = digits.data, digits.target

        X = X / 16.0  # Normalize to [0, 1]

        if split:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.test_size, random_state=self.random_state
            )

            # Reshape for CNN (add channel dimension)
            X_train = X_train.reshape(-1, 1, 8, 8)
            X_test = X_test.reshape(-1, 1, 8, 8)

            train_dataset = TensorDataset(X_train, y_train)
            test_dataset = TensorDataset(X_test, y_test)

            return {
                'train_loader': DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True),
                'test_loader': DataLoader(test_dataset, batch_size=self.batch_size),
                'train_X': X_train, 'train_y': y_train,
                'test_X': X_test, 'test_y': y_test,
                'input_size': 64,
                'num_classes': 10
            }

        return {'X': X, 'y': y, 'input_size': 64, 'num_classes': 10}

    def get_synthetic_classification(self, n_samples: int = 1000, n_features: int = 20,
                                     n_classes: int = 2) -> Dict:
        """Generate synthetic classification dataset."""
        logger.info(
            f"Generating synthetic classification dataset ({n_samples} samples)")
        X, y = make_classification(n_samples=n_samples, n_features=n_features,
                                   n_informative=n_features-2, n_classes=n_classes,
                                   random_state=self.random_state)

        X = self.scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )

        train_dataset = TensorDataset(X_train, y_train)
        test_dataset = TensorDataset(X_test, y_test)

        return {
            'train_loader': DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True),
            'test_loader': DataLoader(test_dataset, batch_size=self.batch_size),
            'train_X': X_train, 'train_y': y_train,
            'test_X': X_test, 'test_y': y_test,
            'input_size': n_features,
            'num_classes': n_classes
        }

    def get_synthetic_regression(self, n_samples: int = 500, n_features: int = 10) -> Dict:
        """Generate synthetic regression dataset."""
        logger.info(
            f"Generating synthetic regression dataset ({n_samples} samples)")
        X, y = make_regression(n_samples=n_samples, n_features=n_features,
                               random_state=self.random_state)

        X = self.scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )

        train_dataset = TensorDataset(X_train, y_train.astype(np.int32))
        test_dataset = TensorDataset(X_test, y_test.astype(np.int32))

        return {
            'train_loader': DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True),
            'test_loader': DataLoader(test_dataset, batch_size=self.batch_size),
            'train_X': X_train, 'train_y': y_train,
            'test_X': X_test, 'test_y': y_test,
            'input_size': n_features,
            'output_size': 1
        }

    def get_dataset(self, dataset_name: str, **kwargs) -> Dict:
        """Factory method to get dataset by name."""
        datasets = {
            'iris': self.get_iris,
            'digits': self.get_digits,
            'synthetic_classification': self.get_synthetic_classification,
            'synthetic_regression': self.get_synthetic_regression
        }

        if dataset_name not in datasets:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        return datasets[dataset_name](**kwargs)

    def get_data_info(self, dataset_name: str) -> Dict:
        """Get information about a dataset without loading it."""
        info = {
            'iris': {'samples': 150, 'features': 4, 'classes': 3, 'description': 'Iris flower classification'},
            'digits': {'samples': 1797, 'features': 64, 'classes': 10, 'description': 'Hand-written digits'},
            'synthetic_classification': {'samples': 'variable', 'features': 'variable', 'classes': 'variable'},
            'synthetic_regression': {'samples': 'variable', 'features': 'variable', 'output': 'continuous'}
        }
        return info.get(dataset_name, {})


__all__ = ['TensorDataset', 'DataManager']
