import torch
from torch.optim.optimizer import Optimizer
from typing import Callable, Optional, Iterable, Union


class PolyakSGD(Optimizer):
    """
    Implementation of SGD with Polyak step size.
    
    The step size is computed using the formula:
        α = (f_N(θ) - f*) / (||∇f_N(θ)||^2 + eps)
    
    where f_N(θ) is the current loss, f* is the known minimum value of the loss function,
    ∇f_N(θ) is the gradient, and eps is a small constant for numerical stability.
    
    Args:
        params: Iterable of parameters to optimize or dicts defining parameter groups
        eps: Small positive constant to prevent division by zero (default: 1e-8)
        f_star: Known minimum value of the loss function (default: 0.0)
    """
    def __init__(
        self, 
        params: Iterable[Union[torch.Tensor, dict]],
        *,
        eps: float = 1e-8,
        f_star: float = 0.0
    ):
        if eps <= 0.0:
            raise ValueError(f"Invalid epsilon value: {eps} - should be positive")
        
        defaults = {"eps": eps, "f_star": f_star}
        super().__init__(params, defaults)
        self._last_step_size: Optional[float] = None  # Store the last step size
    
    @property
    def last_step_size(self) -> Optional[float]:
        """Returns the step size computed in the last call to step()."""
        return self._last_step_size
    
    def step(self, closure: Callable[[], torch.Tensor]) -> torch.Tensor:
        """
        Performs a single optimization step using Polyak's adaptive step size.
        
        Args:
            closure: A callable that reevaluates the model and returns the loss

        Returns:
            The loss value returned by the closure
        """
        # Call closure to compute current loss
        loss = closure()
        
        # Initialize squared gradient norm
        grad_norm_sq = 0.0
        
        # Accumulate squared gradient norm across all parameters
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                grad_norm_sq += torch.sum(grad * grad).item()
        
        # Compute Polyak step size
        f_star = self.defaults['f_star']
        eps = self.defaults['eps']
        
        # Ensure loss is detached to get a scalar
        loss_value = loss.item()
        
        # Compute step size
        step_size = max(0.0, (loss_value - f_star) / (grad_norm_sq + eps))
        self._last_step_size = step_size  # Store the computed step size
        
        # Update parameters
        with torch.no_grad():
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is None:
                        continue
                    
                    p.add_(p.grad, alpha=-step_size)
        
        return loss
