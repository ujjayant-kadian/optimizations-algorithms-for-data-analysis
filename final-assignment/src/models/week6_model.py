import torch
import torch.nn as nn

Tensor = torch.Tensor


class Week6Model(nn.Module):
    def __init__(self):
        super().__init__()
        # x is 2-vector of parameters
        self.x = nn.Parameter(torch.randn(2))
        
    def forward(self, minibatch: Tensor) -> Tensor:
        """
        Compute the loss function given a minibatch of data points.
        
        Args:
            minibatch: Tensor of shape (batch_size, 2) containing data points
            
        Returns:
            Loss value as a scalar tensor
        """
        return self._loss_function(self.x, minibatch)
    
    def _loss_function(self, x: Tensor, minibatch: Tensor) -> Tensor:
        """
        Computes loss function: sum_{w in training data} f(x,w) (Same as Week 6)
        
        Args:
            x: Parameter vector of shape (2,)
            minibatch: Tensor of shape (batch_size, 2) containing data points
            
        Returns:
            Average loss over the minibatch
        """
        total_loss = torch.tensor(0.0, device=x.device)
        count = 0
        
        for w in minibatch:
            z = x - w - 1
            # min(20*(z[0]**2+z[1]**2), (z[0]+9)**2+(z[1]+10)**2)
            term1 = 20 * (z[0]**2 + z[1]**2)
            term2 = (z[0] + 9)**2 + (z[1] + 10)**2
            point_loss = torch.minimum(term1, term2)
            total_loss = total_loss + point_loss
            count += 1
            
        return total_loss / count
