import torch
from torch.optim.optimizer import Optimizer
from typing import Callable, Iterable, Union


class SGD(Optimizer):
    """
    Implementation of standard Stochastic Gradient Descent with constant learning rate.
    
    Args:
        params: Iterable of parameters to optimize or dicts defining parameter groups
        lr: Learning rate (default: 0.01)
    """
    def __init__(
        self, 
        params: Iterable[Union[torch.Tensor, dict]],
        *,
        lr: float = 0.01
    ):
        if lr <= 0.0:
            raise ValueError(f"Invalid learning rate: {lr} - should be positive")
        
        defaults = {"lr": lr}
        super().__init__(params, defaults)
    
    def step(self, closure: Callable[[], torch.Tensor] = None) -> Union[torch.Tensor, None]:
        """
        Performs a single optimization step using constant learning rate.
        
        Args:
            closure: A callable that reevaluates the model and returns the loss (optional)

        Returns:
            The loss value if closure is provided, otherwise None
        """
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            lr = group['lr']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                # Simple SGD update
                with torch.no_grad():
                    p.add_(p.grad, alpha=-lr)
        
        return loss
