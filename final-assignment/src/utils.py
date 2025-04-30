import random
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from typing import Optional, Dict, List, Tuple, Callable, Any


def set_seed(seed: int) -> None:
    """
    Set random seeds for reproducibility across all libraries.
    
    Args:
        seed: Integer seed value to use
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
        
    # Additional configurations for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """
    Get the appropriate device (CUDA or CPU) for PyTorch operations.
    
    Returns:
        torch.device: The device to use for computations
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_random_tensor(shape: tuple, 
                         seed: Optional[int] = None,
                         device: Optional[torch.device] = None) -> torch.Tensor:
    """
    Create a random tensor with a specific shape and seed.
    
    Args:
        shape: Tuple specifying the shape of the tensor
        seed: Optional seed for reproducibility
        device: Optional device to place the tensor on
        
    Returns:
        Random tensor of the specified shape
    """
    if seed is not None:
        set_seed(seed)
        
    if device is None:
        device = get_device()
        
    return torch.randn(shape, device=device)


def get_data_loader(X: torch.Tensor, y: Optional[torch.Tensor] = None, batch_size: int = 8) -> DataLoader:
    """
    Create a DataLoader for the given data.
    
    Args:
        X: Input features tensor of shape (N, d)
        y: Optional target values tensor of shape (N,). If None, creates a dataset with only X
        batch_size: Size of mini-batches
        
    Returns:
        DataLoader that yields batches of (X,) or (X, y) pairs depending on if y is provided
    """
    # Create dataset based on whether y is provided
    if y is not None:
        # Ensure y has the right shape for the model
        if y.dim() == 1:
            y = y.view(-1, 1)
        dataset = TensorDataset(X, y)
    else:
        # Create dataset with just X
        dataset = TensorDataset(X)
    
    # Create data loader
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True
    )


def create_text_tensor_dataset(data: torch.Tensor, 
                               block_size: int,
                               device: Optional[torch.device] = None) -> TensorDataset:
    """
    Create a TensorDataset for text data using sliding window approach.
    
    Args:
        data: Encoded text data tensor
        block_size: Context length for each sample
        device: Device to place tensors on
        
    Returns:
        TensorDataset with input-target pairs for language modeling
    """
    if device is None:
        device = get_device()
    
    # Create input-target pairs
    inputs = []
    targets = []
    
    for i in range(len(data) - block_size):
        # Input is a sequence of block_size tokens
        inputs.append(data[i:i+block_size])
        # Target is the sequence shifted by one position
        targets.append(data[i+1:i+block_size+1])
    
    # Convert to tensors
    x = torch.stack(inputs).to(device)
    y = torch.stack(targets).to(device)
    
    return TensorDataset(x, y)


def get_text_data_loader(data: torch.Tensor, 
                         block_size: int, 
                         batch_size: int,
                         shuffle: bool = True,
                         device: Optional[torch.device] = None) -> DataLoader:
    """
    Create a DataLoader for text data.
    
    Args:
        data: Encoded text data tensor
        block_size: Context length for each sample
        batch_size: Size of mini-batches
        shuffle: Whether to shuffle the data
        device: Device to place tensors on
        
    Returns:
        DataLoader for language modeling
    """
    dataset = create_text_tensor_dataset(data, block_size, device)
    
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle
    )


def decode_text(indices: List[int], itos: Dict[int, str]) -> str:
    """
    Decode a list of token indices back to a string.
    
    Args:
        indices: List of integer token indices
        itos: Dictionary mapping indices to characters
        
    Returns:
        Decoded string
    """
    return ''.join([itos[idx] for idx in indices])


def analyze_text_data(text: str) -> Dict[str, Any]:
    """
    Analyze a text dataset and return statistics.
    
    Args:
        text: Text data to analyze
        
    Returns:
        Dictionary containing various statistics about the text
    """
    # Character-level statistics
    chars = sorted(list(set(text)))
    char_vocab_size = len(chars)
    total_chars = len(text)
    
    # Word-level statistics
    words = text.split()
    unique_words = set(words)
    word_count = len(words)
    word_vocab_size = len(unique_words)
    
    # Return statistics dictionary
    return {
        "char_vocab_size": char_vocab_size,
        "total_chars": total_chars,
        "word_vocab_size": word_vocab_size,
        "word_count": word_count,
        "sample": text[:500] if len(text) > 500 else text,
        "unique_chars": chars
    }
