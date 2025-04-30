import torch
import os
from typing import Tuple, Dict, List, Optional, Union, Callable
from pathlib import Path

from src.datasets import prepare_transformer_data, get_transformer_batch, load_text_data, load_and_encode_text
from src.utils import get_device, set_seed, decode_text, analyze_text_data


class TransformerDataManager:
    """
    Manager class for handling transformer text data loading and processing.
    This class provides a simple interface for getting data batches and configurations
    for transformer models.
    """
    
    def __init__(self, 
                 data_dir: str = 'src/transformer-datasets',
                 dataset_file: str = 'input_childSpeech_trainingSet.txt',
                 block_size: int = 256,
                 batch_size: int = 64,
                 max_iters: int = 2000,
                 eval_interval: int = 500,
                 learning_rate: float = 3e-4,
                 eval_iters: int = 200,
                 n_embd: int = 192,
                 n_head: int = 4,
                 n_layer: int = 2,
                 dropout: float = 0.2,
                 train_ratio: float = 0.9,
                 seed: int = 1337):
        """
        Initialize the transformer data manager with all hyperparameters.
        
        Args:
            data_dir: Directory containing text dataset files
            dataset_file: Name of the dataset file to use
            block_size: Context length for sequences
            batch_size: Batch size for training
            max_iters: Maximum number of training iterations
            eval_interval: Interval between loss estimations
            learning_rate: Learning rate for optimizer
            eval_iters: Number of iterations for loss estimation
            n_embd: Embedding dimension
            n_head: Number of attention heads
            n_layer: Number of transformer layers
            dropout: Dropout probability
            train_ratio: Ratio of data to use for training
            seed: Random seed for reproducibility
        """
        # Set configuration
        self.data_dir = Path(data_dir)
        self.dataset_file = dataset_file
        self.block_size = block_size
        self.batch_size = batch_size
        self.max_iters = max_iters
        self.eval_interval = eval_interval
        self.learning_rate = learning_rate
        self.eval_iters = eval_iters
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_layer = n_layer
        self.dropout = dropout
        self.train_ratio = train_ratio
        self.seed = seed
        self.device = get_device()
        
        # Set seed for reproducibility
        set_seed(seed)
        
        # Initialize empty attributes
        self.train_data = None
        self.val_data = None
        self.stoi = None
        self.itos = None
        self.vocab_size = None
        
        # Construct the file path
        self.filepath = str(self.data_dir / self.dataset_file)
        
        # Verify file exists
        if not os.path.exists(self.filepath):
            raise FileNotFoundError(f"Dataset file not found: {self.filepath}")
        
        # Prepare data
        self._prepare_data()
    
    def _prepare_data(self):
        """Prepare the transformer data."""
        # Load and prepare data
        (
            self.train_data, 
            self.val_data, 
            self.stoi, 
            self.itos
        ) = prepare_transformer_data(
            filepath=self.filepath,
            train_ratio=self.train_ratio,
            seed=self.seed
        )
        
        # Set vocabulary size
        self.vocab_size = len(self.stoi)
    
    def get_batch(self, split: str = 'train') -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a batch of data for training or validation.
        
        Args:
            split: Either 'train' or 'val'
            
        Returns:
            Tuple of (x, y) tensors
        """
        data = self.train_data if split == 'train' else self.val_data
        
        return get_transformer_batch(
            data=data,
            block_size=self.block_size,
            batch_size=self.batch_size,
            device=self.device
        )
    
    def encode_text(self, text: str) -> List[int]:
        """
        Encode text string to token indices.
        
        Args:
            text: Text string to encode
            
        Returns:
            List of token indices
        """
        return [self.stoi.get(c, 0) for c in text]
    
    def decode_indices(self, indices: List[int]) -> str:
        """
        Decode token indices to text string.
        
        Args:
            indices: List of token indices
            
        Returns:
            Decoded text string
        """
        return decode_text(indices, self.itos)
    
    def get_config(self) -> Dict:
        """
        Get the configuration for the transformer model.
        
        Returns:
            Dictionary with configuration parameters
        """
        return {
            # Data configuration
            'dataset_file': self.dataset_file,
            'train_data_size': len(self.train_data),
            'val_data_size': len(self.val_data),
            'vocab_size': self.vocab_size,
            
            # Model hyperparameters
            'block_size': self.block_size,
            'batch_size': self.batch_size,
            'n_embd': self.n_embd,
            'n_head': self.n_head,
            'n_layer': self.n_layer,
            'dropout': self.dropout,
            
            # Training hyperparameters
            'max_iters': self.max_iters,
            'eval_interval': self.eval_interval,
            'learning_rate': self.learning_rate,
            'eval_iters': self.eval_iters,
            
            # Environment
            'device': str(self.device),
            'seed': self.seed
        }
    
    def analyze_dataset(self) -> Dict:
        """
        Analyze the loaded dataset.
        
        Returns:
            Dictionary with dataset statistics
        """
        # Get sample text from train data
        train_sample = self.decode_indices(self.train_data[:1000].tolist())
        
        stats = {
            'train_size': len(self.train_data),
            'val_size': len(self.val_data),
            'vocab_size': self.vocab_size,
            'sample': train_sample[:200] + '...' if len(train_sample) > 200 else train_sample
        }
        
        return stats
    
    def load_test_dataset(self, test_filepath: str) -> torch.Tensor:
        """
        Load and encode a test dataset using the current character mappings.
        
        Args:
            test_filepath: Path to the test dataset file
            
        Returns:
            Encoded test data tensor
        """
        return load_and_encode_text(test_filepath, self.stoi) 