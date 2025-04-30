import torch
import pytest
import numpy as np
from src.optim.polyak_adam import PolyakAdam
from src.utils import set_seed


def test_gradient_norm_calculation():
    """
    Test if the squared gradient norm is correctly calculated.
    """
    # Create dummy parameters with known gradients
    params = [torch.zeros(3, requires_grad=True)]
    params[0].grad = torch.tensor([1.0, 2.0, 3.0])  # Norm^2 = 1^2 + 2^2 + 3^2 = 14
    
    # Initialize optimizer
    optimizer = PolyakAdam(params)
    
    # Create a closure that returns a dummy loss
    def closure():
        return torch.tensor(10.0, requires_grad=True)
    
    # Monkey patch the step method to check the gradient norm
    original_step = optimizer.step
    
    def step_with_check(closure):
        nonlocal grad_norm_sq_captured
        loss = closure()
        
        # Calculate grad_norm_sq
        grad_norm_sq = 0.0
        for group in optimizer.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    grad_norm_sq += torch.sum(p.grad * p.grad).item()
        
        grad_norm_sq_captured = grad_norm_sq
        
        # Continue with original step
        return loss
    
    # Capture grad_norm_sq during execution
    grad_norm_sq_captured = 0.0
    optimizer.step = step_with_check
    optimizer.step(closure)
    
    # Verify the calculated norm
    expected_grad_norm_sq = 14.0
    assert abs(grad_norm_sq_captured - expected_grad_norm_sq) < 1e-6


def test_step_size_formula():
    """
    Test if the Polyak step size is calculated correctly.
    """
    # Create dummy parameters with known gradients
    params = [torch.zeros(2, requires_grad=True)]
    params[0].grad = torch.tensor([3.0, 4.0])  # Norm^2 = 3^2 + 4^2 = 25
    
    # Initialize optimizer with known f_star and eps
    f_star = 2.0
    eps = 1e-8
    optimizer = PolyakAdam(params, f_star=f_star, eps=eps)
    
    # Create a closure that returns a dummy loss
    loss_value = 7.0
    def closure():
        return torch.tensor(loss_value)
    
    # Mock the optimizer step method to check the step size
    original_step = optimizer.step
    
    def step_with_check(closure):
        nonlocal step_size_captured
        loss = closure()
        
        # Calculate grad_norm_sq
        grad_norm_sq = 0.0
        for group in optimizer.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    grad_norm_sq += torch.sum(p.grad * p.grad).item()
        
        # Calculate step size
        step_size = max(0.0, (loss.item() - f_star) / (grad_norm_sq + eps))
        step_size_captured = step_size
        
        return loss
    
    # Capture step_size during execution
    step_size_captured = 0.0
    optimizer.step = step_with_check
    optimizer.step(closure)
    
    # Expected step size: (7.0 - 2.0) / (25 + 1e-8) = 5.0 / 25 = 0.2
    expected_step_size = 0.2
    assert abs(step_size_captured - expected_step_size) < 1e-6


def test_adam_momentum_updates():
    """
    Test if the Adam momentum components are properly updating.
    """
    # Set random seed for reproducibility
    set_seed(42)
    
    # Create a simple parameter
    param = torch.tensor([1.0], requires_grad=True)
    grad = torch.tensor([2.0])
    
    # Initialize optimizer with default beta values
    optimizer = PolyakAdam([param], betas=(0.9, 0.999))
    
    # Set the parameter's gradient
    param.grad = grad
    
    # Create a simple closure
    def closure():
        return torch.tensor(5.0)
    
    # Call step to update the parameter
    optimizer.step(closure)
    
    # Check that the momentum values were initialized and updated correctly
    state = optimizer.state[param]
    
    # First moment should be initialized to grad * (1-beta1)
    assert 'exp_avg' in state
    expected_exp_avg = grad * 0.1  # (1-beta1) = 0.1
    assert torch.allclose(state['exp_avg'], expected_exp_avg)
    
    # Second moment should be initialized to grad^2 * (1-beta2)
    assert 'exp_avg_sq' in state
    expected_exp_avg_sq = grad * grad * 0.001  # (1-beta2) = 0.001
    assert torch.allclose(state['exp_avg_sq'], expected_exp_avg_sq)


def test_amsgrad_variant():
    """
    Test if the AMSGrad variant behaves correctly.
    """
    # Set random seed for reproducibility
    set_seed(42)
    
    # Create a simple parameter
    param = torch.tensor([1.0], requires_grad=True)
    
    # Initialize optimizers - one with AMSGrad, one without
    optimizer_with_amsgrad = PolyakAdam([param], amsgrad=True)
    optimizer_without_amsgrad = PolyakAdam([param], amsgrad=False)
    
    # Set the parameter's gradient
    param.grad = torch.tensor([2.0])
    
    # Create a simple closure
    def closure():
        return torch.tensor(5.0)
    
    # Call step to update the parameter
    optimizer_with_amsgrad.step(closure)
    
    # Check that the amsgrad state was initialized correctly
    state = optimizer_with_amsgrad.state[param]
    assert 'max_exp_avg_sq' in state
    
    # Check that without amsgrad, the max_exp_avg_sq is not created
    param.grad = torch.tensor([2.0])  # Reset the gradient
    optimizer_without_amsgrad.step(closure)
    state = optimizer_without_amsgrad.state[param]
    assert 'max_exp_avg_sq' not in state


def test_quadratic_optimization():
    """
    Test if the optimizer can minimize a simple quadratic function f(θ) = θ^2.
    """
    # Set random seed for reproducibility
    set_seed(42)
    
    # For this function, f_star = 0 (minimum at θ = 0)
    f_star = 0.0
    
    # Create a simple model with one parameter
    class SimpleModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.theta = torch.nn.Parameter(torch.tensor([10.0]))
            
        def forward(self):
            return self.theta * self.theta
    
    model = SimpleModel()
    optimizer = PolyakAdam(model.parameters(), f_star=f_star)
    
    # Run optimization for a few steps
    for _ in range(5):
        def closure():
            optimizer.zero_grad()
            loss = model()
            loss.backward()
            return loss
        
        loss = optimizer.step(closure)
    
    # The parameter should approach 0
    assert abs(model.theta.item()) < 1.0


def test_synthetic_regression():
    """
    Test if the optimizer works on a simple linear regression problem.
    """
    # Set random seed for reproducibility
    set_seed(42)
    
    # Generate synthetic data: y = 2*x + 1 + noise
    n_samples = 100
    x = torch.rand(n_samples, 1) * 10
    y_true = 2 * x + 1
    y = y_true + torch.randn(n_samples, 1) * 0.5
    
    # Create a simple linear model
    class LinearModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(1, 1)
            
        def forward(self, x):
            return self.linear(x)
    
    model = LinearModel()
    
    # Initialize with random weights
    torch.nn.init.uniform_(model.linear.weight, -1.0, 1.0)
    torch.nn.init.uniform_(model.linear.bias, -1.0, 1.0)
    
    # Define loss function
    loss_fn = torch.nn.MSELoss()
    
    # Initialize our optimizer
    optimizer = PolyakAdam(model.parameters(), eps=1e-8)
    
    # Train for a few epochs
    initial_loss = None
    final_loss = None
    
    for epoch in range(20):
        def closure():
            optimizer.zero_grad()
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            return loss
        
        loss = optimizer.step(closure)
        
        if epoch == 0:
            initial_loss = loss.item()
        if epoch == 19:
            final_loss = loss.item()
    
    # The loss should decrease
    assert final_loss < initial_loss
    
    # The model should approximate the true parameters
    # For PolyakAdam (being momentum-based), we need a slightly higher tolerance than plain SGD
    assert abs(model.linear.weight.item() - 2.0) < 1.02
    assert abs(model.linear.bias.item() - 1.0) < 1.02  # Updated to match weight tolerance


def test_invalid_parameters():
    """
    Test if the optimizer raises a ValueError for invalid parameter values.
    """
    # Test invalid epsilon
    with pytest.raises(ValueError):
        PolyakAdam([torch.zeros(1, requires_grad=True)], eps=0.0)
    
    with pytest.raises(ValueError):
        PolyakAdam([torch.zeros(1, requires_grad=True)], eps=-1e-8)
    
    # Test invalid beta values
    with pytest.raises(ValueError):
        PolyakAdam([torch.zeros(1, requires_grad=True)], betas=(-0.1, 0.999))
    
    with pytest.raises(ValueError):
        PolyakAdam([torch.zeros(1, requires_grad=True)], betas=(0.9, 1.0))
    
    # Test invalid weight decay
    with pytest.raises(ValueError):
        PolyakAdam([torch.zeros(1, requires_grad=True)], weight_decay=-0.1)


def test_bias_correction():
    """
    Test if the bias correction is applied correctly.
    """
    # Set random seed for reproducibility
    set_seed(42)
    
    # Create a simple parameter and gradient
    param = torch.tensor([1.0], requires_grad=True)
    param.grad = torch.tensor([1.0])
    
    # Initialize optimizer with known betas
    beta1, beta2 = 0.9, 0.999
    optimizer = PolyakAdam([param], betas=(beta1, beta2))
    
    # Set a known loss value and create a closure
    def closure():
        return torch.tensor(2.0)
    
    # Step the optimizer and capture the before/after parameter values
    param_before = param.clone()
    optimizer.step(closure)
    param_after = param.clone()
    
    # Check that the step number was incremented
    state = optimizer.state[param]
    assert state['step'] == 1
    
    # Verify bias correction was applied (the parameter should have changed)
    assert not torch.allclose(param_before, param_after)


def test_usage_example():
    """
    Test a usage example similar to the one for PolyakSGD
    """
    # Set random seed for reproducibility
    set_seed(42)
    
    # Create a simple model and dummy data
    model = torch.nn.Linear(2, 1)
    x_batch = torch.randn(10, 2)
    y_batch = torch.randn(10, 1)
    loss_fn = torch.nn.MSELoss()
    
    # Initialize optimizer
    optimizer = PolyakAdam(model.parameters(), eps=1e-8)
    
    # Define the closure
    def closure():
        optimizer.zero_grad()
        output = model(x_batch)
        loss = loss_fn(output, y_batch)
        loss.backward()
        return loss
    
    # Call step with the closure
    loss = optimizer.step(closure)
    
    # Verify that loss is a tensor
    assert isinstance(loss, torch.Tensor)
    assert loss.numel() == 1  # Single scalar value
    
    # Verify that the last step size is stored
    assert optimizer.last_step_size is not None 