import numpy as np
import torch
from typing import Tuple, Dict, List, Callable, Optional
from src.utils import set_seed, create_random_tensor, get_device
import os


def make_synthetic_linreg(N: int, d: int, noise_std: float, seed: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """
    Generate synthetic data for linear regression with the model: y = w⊤x + b + ε,
    where ε is Gaussian noise with standard deviation noise_std.
    
    Args:
        N: Number of data points to generate
        d: Dimensionality of the feature space
        noise_std: Standard deviation of the Gaussian noise
        seed: Random seed for reproducibility
        
    Returns:
        Tuple containing:
            - X: Input features of shape (N, d)
            - y: Target values of shape (N,)
            - w: True weight vector of shape (d,)
            - b: True bias term (scalar)
    """
    # Set seed for reproducibility
    set_seed(seed)
    
    # Generate input features X ~ N(0, I)
    X = create_random_tensor((N, d), seed=seed)
    
    # Generate true parameters w ~ N(0, I) and b ~ N(0, 1)
    # We use a fixed seed + 1 to ensure w and b are different from X but consistent across runs
    w = create_random_tensor((d,), seed=seed+1)
    b = torch.randn(1).item()  # Scalar bias term
    
    # Generate target values: y = Xw + b + noise
    y_clean = torch.matmul(X, w) + b
    
    # Add Gaussian noise with standard deviation noise_std
    # Use seed+2 to ensure noise is different but consistent
    noise = create_random_tensor((N,), seed=seed+2) * noise_std
    y = y_clean + noise
    
    return X, y, w, b


def make_gaussian_cluster(m: int = 25, std: float = 0.25, seed: int = 42) -> torch.Tensor:
    """
    Generate data points from a 2D Gaussian distribution centered at origin. (Week 6 Training Data)
    
    Original function
    def generate_trainingdata(m=25):
        return np.array([0,0])+0.25*np.random.randn(m,2)
    
    Args:
        m: Number of data points to generate
        std: Standard deviation of the Gaussian noise
        seed: Random seed for reproducibility
        
    Returns:
        X: Data points of shape (m, 2)
    """
    # Set seed for reproducibility
    set_seed(seed)
    
    # Generate points from a 2D Gaussian centered at origin
    X = create_random_tensor((m, 2), seed=seed) * std
    
    return X


def load_text_data(filepath: str) -> str:
    """
    Load text data from a file.
    
    Args:
        filepath: Path to the text file
        
    Returns:
        The content of the text file as a string
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()


def create_char_mappings(text: str) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Create character-to-index and index-to-character mappings.
    
    Args:
        text: Text data to create mappings from
        
    Returns:
        Tuple containing:
            - stoi: Dictionary mapping characters to indices
            - itos: Dictionary mapping indices to characters
    """
    # Get unique characters in sorted order
    chars = sorted(list(set(text)))
    
    # Create mappings
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    
    return stoi, itos


def prepare_transformer_data(filepath: str, 
                             train_ratio: float = 0.9, 
                             seed: int = 1337) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, int], Dict[int, str]]:
    """
    Prepare data for transformer model training and evaluation from a single text file.
    
    Args:
        filepath: Path to the text file
        train_ratio: Ratio of data to use for training (rest for validation)
        seed: Random seed for reproducibility
        
    Returns:
        Tuple containing:
            - train_data: Training data tensor
            - val_data: Validation data tensor
            - stoi: Character to index mapping
            - itos: Index to character mapping
    """
    # Set seed for reproducibility
    set_seed(seed)
    
    # Load text data
    text_data = load_text_data(filepath)
    
    # Create character mappings
    stoi, itos = create_char_mappings(text_data)
    
    # Create encoder function
    encode = lambda s: [stoi[c] for c in s]
    
    # Encode data
    data = torch.tensor(encode(text_data), dtype=torch.long)
    
    # Split into train and validation sets
    n = int(train_ratio * len(data))
    train_data = data[:n]
    val_data = data[n:]
    
    return train_data, val_data, stoi, itos


def encode_text(text: str, stoi: Dict[str, int], unk_token: int = 0) -> List[int]:
    """
    Encode text using given character to index mapping.
    
    Args:
        text: Text to encode
        stoi: Character to index mapping dictionary
        unk_token: Index to use for unknown characters
        
    Returns:
        List of encoded indices
    """
    return [stoi.get(c, unk_token) for c in text]


def load_and_encode_text(filepath: str, stoi: Dict[str, int], unk_token: int = 0) -> torch.Tensor:
    """
    Load and encode text from a file using existing character mapping.
    Can be used for loading test data with existing vocabulary.
    
    Args:
        filepath: Path to the text file
        stoi: Character to index mapping 
        unk_token: Index to use for unknown characters
        
    Returns:
        Tensor of encoded indices
    """
    # Verify file exists
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Text file not found: {filepath}")
    
    # Load text data
    text_data = load_text_data(filepath)
    
    # Encode using provided character mappings
    encoded_data = encode_text(text_data, stoi, unk_token)
    
    # Convert to tensor
    return torch.tensor(encoded_data, dtype=torch.long)


def get_transformer_batch(data: torch.Tensor, 
                          block_size: int, 
                          batch_size: int, 
                          device: Optional[torch.device] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate a batch of data for transformer training or evaluation.
    Works for any dataset (train, validation, or test).
    
    Args:
        data: Encoded text data tensor
        block_size: Maximum context length
        batch_size: Number of sequences in a batch
        device: Device to place tensors on (defaults to utils.get_device())
        
    Returns:
        Tuple containing:
            - x: Input tensor of shape (batch_size, block_size)
            - y: Target tensor of shape (batch_size, block_size)
    """
    if device is None:
        device = get_device()
    
    # Generate random starting indices
    ix = torch.randint(len(data) - block_size, (batch_size,))
    
    # Create input sequences
    x = torch.stack([data[i:i+block_size] for i in ix])
    
    # Create target sequences (shifted by one position)
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    
    # Move to device
    x, y = x.to(device), y.to(device)
    
    return x, y
