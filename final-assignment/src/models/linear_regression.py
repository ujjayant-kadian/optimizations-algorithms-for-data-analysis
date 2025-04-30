import torch
import torch.nn as nn

class LinReg(nn.Module):
    """
    Linear regression model implemented as a PyTorch module.
    
    Args:
        d: Input feature dimension
    """
    def __init__(self, d: int):
        super(LinReg, self).__init__()
        self.linear = nn.Linear(d, 1)  # Linear layer with d inputs and 1 output
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the linear regression model.
        
        Args:
            x: Input tensor of shape (batch_size, d)
            
        Returns:
            Predictions of shape (batch_size, 1)
        """
        return self.linear(x)


def get_mse_loss() -> nn.MSELoss:
    """
    Get the Mean Squared Error loss function commonly used for linear regression.
    
    Returns:
        MSE loss function
    """
    return nn.MSELoss()
