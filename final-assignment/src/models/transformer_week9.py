import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from typing import Optional, Tuple, List, Dict

from src.models.transformer_data import TransformerDataManager


class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size: int, n_embd: int, block_size: int, dropout: float):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=True)
        self.query = nn.Linear(n_embd, head_size, bias=True)
        self.value = nn.Linear(n_embd, head_size, bias=True)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B, T, C = x.shape
        k = self.key(x)   # (B,T,hs)
        q = self.query(x) # (B,T,hs)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,hs)
        out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out


class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, n_head: int, head_size: int, n_embd: int, dropout: float):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, n_embd, block_size=2048, dropout=dropout) for _ in range(n_head)])
        self.proj = nn.Linear(head_size * n_head, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd: int, n_head: int, dropout: float):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size, n_embd, dropout)
        self.ffwd = FeedForward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPTLanguageModel(nn.Module):
    """GPT Language Model that uses hyperparameters from TransformerDataManager (same as Week 9 - ML Module)"""
    
    def __init__(self, data_manager: TransformerDataManager):
        super().__init__()
        # Get hyperparameters from data manager
        self.data_manager = data_manager
        self.block_size = data_manager.block_size
        self.vocab_size = data_manager.vocab_size
        self.n_embd = data_manager.n_embd
        self.n_head = data_manager.n_head
        self.n_layer = data_manager.n_layer
        self.dropout = data_manager.dropout
        self.device = data_manager.device
        
        # Transformer components
        self.token_embedding_table = nn.Embedding(self.vocab_size, self.n_embd)
        self.position_embedding_table = nn.Embedding(self.block_size, self.n_embd)
        self.blocks = nn.Sequential(*[Block(self.n_embd, n_head=self.n_head, dropout=self.dropout) 
                                     for _ in range(self.n_layer)])
        self.ln_f = nn.LayerNorm(self.n_embd)  # final layer norm
        self.lm_head = nn.Linear(self.n_embd, self.vocab_size)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            
    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T = idx.shape
        
        # Get token and position embeddings
        tok_emb = self.token_embedding_table(idx)  # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=self.device))  # (T,C)
        x = tok_emb + pos_emb  # (B,T,C)
        
        # Apply transformer blocks and final layer norm
        x = self.blocks(x)  # (B,T,C)
        x = self.ln_f(x)  # (B,T,C)
        
        # Get logits
        logits = self.lm_head(x)  # (B,T,vocab_size)
        
        # Calculate loss if targets are provided
        loss = None
        if targets is not None:
            loss = get_cross_entropy_loss_value(logits, targets)
            
        return logits, loss
    
    def generate(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        """
        Generate text by sampling from the model.
        
        Args:
            idx: Context tensor of shape (B, T)
            max_new_tokens: Number of tokens to generate
            
        Returns:
            Generated tensor of shape (B, T+max_new_tokens)
        """
        self.eval()
        
        # Generate tokens
        for _ in range(max_new_tokens):
            # Crop context to block_size
            idx_cond = idx[:, -self.block_size:]
            
            # Get predictions
            logits, _ = self(idx_cond)
            
            # Focus on the last time step
            logits = logits[:, -1, :]  # (B, C)
            
            # Apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            
            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            
            # Append sampled index to the sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
            
        return idx
    
    @torch.no_grad()
    def estimate_loss(self, data_manager: TransformerDataManager) -> Dict[str, float]:
        """
        Estimate the loss for the model on train and validation datasets.
        
        Args:
            data_manager: TransformerDataManager instance to get batches from
            
        Returns:
            Dictionary with train and val losses
        """
        out = {}
        self.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(data_manager.eval_iters)
            for k in range(data_manager.eval_iters):
                X, Y = data_manager.get_batch(split)
                logits, _ = self(X)
                losses[k] = get_cross_entropy_loss_value(logits, Y).item()
            out[split] = losses.mean()
        self.train()
        return out
    
    @torch.no_grad()
    def evaluate_test_loss(self, test_data: torch.Tensor, block_size: int, batch_size: int, eval_iters: int) -> float:
        """
        Evaluate loss on test data.
        
        Args:
            test_data: Encoded test data tensor
            block_size: Context length for sequences
            batch_size: Batch size for evaluation
            eval_iters: Number of iterations for loss estimation
            
        Returns:
            Test loss (scalar)
        """
        from src.datasets import get_transformer_batch
        
        self.eval()
        losses = torch.zeros(eval_iters)
        
        for k in range(eval_iters):
            xb, yb = get_transformer_batch(test_data, block_size, batch_size, self.device)
            logits, _ = self(xb)
            losses[k] = get_cross_entropy_loss_value(logits, yb).item()
            
        return losses.mean().item()


def get_cross_entropy_loss() -> nn.CrossEntropyLoss:
    """
    Get the Cross Entropy loss function commonly used for language modeling.
    
    Returns:
        Cross Entropy loss function
    """
    return nn.CrossEntropyLoss()


def get_cross_entropy_loss_value(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Calculate cross entropy loss between logits and targets.
    
    Args:
        logits: Predicted logits of shape (B, T, vocab_size) or (B*T, vocab_size)
        targets: Target indices of shape (B, T) or (B*T)
        
    Returns:
        Loss value
    """
    if logits.dim() == 3:
        # Reshape if we have 3D input
        B, T, C = logits.shape
        logits = logits.view(B*T, C)
        targets = targets.view(B*T)
    
    return F.cross_entropy(logits, targets)


def calculate_baseline_loss(vocab_size: int) -> float:
    """
    Calculate the baseline loss assuming uniform distribution over vocabulary.
    
    Args:
        vocab_size: Size of the vocabulary
        
    Returns:
        Baseline loss value
    """
    uniform_prob = 1.0 / vocab_size
    return -math.log(uniform_prob) 