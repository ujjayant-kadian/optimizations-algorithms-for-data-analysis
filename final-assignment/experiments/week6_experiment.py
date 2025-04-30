import os
import yaml
import argparse
import numpy as np
import torch
import json
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, List, Union, Tuple, Optional, Any
from pathlib import Path

from src.utils import set_seed, get_device, get_data_loader
from src.datasets import make_gaussian_cluster
from src.models.week6_model import Week6Model
from src.optim.sgd import SGD
from src.optim.polyak_sgd import PolyakSGD


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run Week 6 optimization experiments")
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/week6_experiment_config.yaml",
        help="Path to configuration YAML file"
    )
    return parser.parse_args()


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def train_model(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    epochs: int,
    is_polyak: bool = False
) -> Tuple[List[float], List[float], List[Tuple[float, float]]]:
    """
    Train a model with the given optimizer and data loader.
    
    Args:
        model: PyTorch model to train
        optimizer: Optimizer to use for training
        data_loader: DataLoader providing training data
        device: Device to run training on
        epochs: Number of epochs to train for
        is_polyak: Whether the optimizer is PolyakSGD
        
    Returns:
        Tuple of (training losses, step sizes if Polyak, parameter history)
    """
    model.train()
    losses = []
    step_sizes = [] if is_polyak else None
    param_history = []  # Store x parameter values for contour plotting
    
    # Store initial parameter values
    with torch.no_grad():
        param_history.append((model.x[0].item(), model.x[1].item()))
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        batches = 0
        
        for (minibatch,) in data_loader:
            minibatch = minibatch.to(device)
            
            def closure():
                optimizer.zero_grad()
                loss = model(minibatch)
                loss.backward()
                return loss
            
            if is_polyak:
                loss = optimizer.step(closure)
                step_sizes.append(optimizer.last_step_size)
            else:
                loss = closure()
                optimizer.step()
            
            epoch_loss += loss.item()
            batches += 1
            
            # Store current parameter values
            with torch.no_grad():
                param_history.append((model.x[0].item(), model.x[1].item()))
        
        avg_epoch_loss = epoch_loss / batches
        losses.append(avg_epoch_loss)
        
    return losses, step_sizes, param_history


def compare_optimizers(
    config: Dict[str, Any],
    batch_size: Union[int, str],
    device: torch.device,
    log_dir: str
) -> Dict[str, Any]:
    """
    Compare Constant Step Size SGD with Polyak Step Size SGD.
    
    Args:
        config: Configuration dictionary
        batch_size: Batch size for training (int or "full")
        device: Device to run training on
        log_dir: Directory to save logs
        
    Returns:
        Dictionary with results
    """
    print(f"Comparing optimizers: batch_size={batch_size}")
    
    # Generate dataset
    m = config["data"]["m"]  # Number of data points
    std = config["data"]["std"]  # Standard deviation of Gaussian noise
    seed = config["data"]["seed"]  # Random seed
    epochs = config["training"]["epochs"]
    
    X = make_gaussian_cluster(m=m, std=std, seed=seed)
    
    # Use full batch if specified
    actual_batch_size = m if batch_size == "full" else batch_size
    data_loader = get_data_loader(X, batch_size=actual_batch_size)
    
    # Directory for comparison results
    comp_log_dir = os.path.join(
        log_dir, 
        f"comparison_batch{batch_size}"
    )
    os.makedirs(comp_log_dir, exist_ok=True)
    
    # Train with Constant Step Size SGD
    lr = config["optimizer"]["sgd"]["lr"]
    model_sgd = Week6Model().to(device)
    optimizer_sgd = SGD(model_sgd.parameters(), lr=lr)
    
    sgd_losses, _, sgd_param_history = train_model(
        model=model_sgd,
        optimizer=optimizer_sgd,
        data_loader=data_loader,
        device=device,
        epochs=epochs
    )
    
    # Train with Polyak Step Size SGD
    model_polyak = Week6Model().to(device)
    optimizer_polyak = PolyakSGD(
        model_polyak.parameters(),
        eps=config["optimizer"]["polyak"]["eps"],
        f_star=config["optimizer"]["polyak"]["f_star"]
    )
    
    polyak_losses, polyak_step_sizes, polyak_param_history = train_model(
        model=model_polyak,
        optimizer=optimizer_polyak,
        data_loader=data_loader,
        device=device,
        epochs=epochs,
        is_polyak=True
    )
    
    # Save results
    results = {
        "sgd": {
            "losses": sgd_losses,
            "lr": lr,
            "param_history": sgd_param_history
        },
        "polyak": {
            "losses": polyak_losses,
            "step_sizes": polyak_step_sizes,
            "param_history": polyak_param_history
        },
        "config": {
            "batch_size": batch_size,
            "epochs": epochs
        }
    }
    
    # Save results to file
    with open(os.path.join(comp_log_dir, "comparison_results.json"), "w") as f:
        # Convert param_history to serializable format
        results_copy = results.copy()
        results_copy["sgd"]["param_history"] = [[float(x), float(y)] for x, y in results["sgd"]["param_history"]]
        results_copy["polyak"]["param_history"] = [[float(x), float(y)] for x, y in results["polyak"]["param_history"]]
        json.dump(results_copy, f, indent=2)
    
    # Plot loss comparison
    plt.figure(figsize=(10, 6))
    plt.plot(sgd_losses, label=f"Constant Step Size (LR={lr})")
    plt.plot(polyak_losses, label="Polyak Step Size")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Loss Comparison: Constant vs Polyak Step Size\nBatch Size={batch_size}")
    plt.legend()
    plt.savefig(os.path.join(comp_log_dir, "loss_comparison.png"))
    plt.close()
    
    # Plot Polyak step sizes
    plt.figure(figsize=(10, 6))
    plt.plot(polyak_step_sizes)
    plt.xlabel("Batch Update")
    plt.ylabel("Step Size")
    plt.title(f"Polyak Step Sizes Over Training\nBatch Size={batch_size}")
    plt.savefig(os.path.join(comp_log_dir, "polyak_step_sizes.png"))
    plt.close()
    
    return results


def create_batch_size_comparison_plot(
    results: Dict[Union[int, str], Dict],
    log_dir: str
) -> None:
    """
    Create a plot comparing final losses for different batch sizes.
    
    Args:
        results: Dictionary mapping batch sizes to results
        log_dir: Directory to save plot
    """
    plt.figure(figsize=(10, 6))
    
    # Extract batch sizes and prepare lists for plotting
    batch_sizes = sorted(results.keys(), key=lambda bs: int(bs) if bs != "full" else float('inf'))
    batch_labels = [str(bs) for bs in batch_sizes]
    sgd_final_losses = [results[bs]["sgd"]["losses"][-1] for bs in batch_sizes]
    polyak_final_losses = [results[bs]["polyak"]["losses"][-1] for bs in batch_sizes]
    
    x = list(range(len(batch_sizes)))
    
    # Plot final losses
    plt.plot(x, sgd_final_losses, 'o-', label="SGD (Constant Step Size)")
    plt.plot(x, polyak_final_losses, 'x--', label="Polyak Step Size")
    plt.xticks(x, batch_labels)
    plt.xlabel("Batch Size")
    plt.ylabel("Final Loss")
    plt.title("Effect of Batch Size on Final Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, "batch_size_comparison.png"))
    plt.close()


def create_convergence_comparison_plot(
    results: Dict[Union[int, str], Dict],
    log_dir: str
) -> None:
    """
    Create a plot comparing convergence speed for different batch sizes.
    
    Args:
        results: Dictionary mapping batch sizes to results
        log_dir: Directory to save plot
    """
    plt.figure(figsize=(12, 10))
    
    # Sort batch sizes for consistent plotting
    # Convert batch sizes to strings for consistent sorting
    sorted_items = sorted(results.items(), key=lambda x: (
            str(x[0]) == "full",  # Sort "full" last
            int(x[0]) if str(x[0]) != "full" else float('inf')  # Sort numerically
        ))
    
    # Plot SGD convergence
    plt.subplot(2, 1, 1)
    for batch_size, result in sorted_items:
        sgd_losses = result["sgd"]["losses"]
        plt.plot(sgd_losses, label=f"Batch={batch_size}, LR={result['sgd']['lr']:.4f}")
    
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("SGD Convergence for Different Batch Sizes")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Plot Polyak convergence
    plt.subplot(2, 1, 2)
    for batch_size, result in sorted_items:
        polyak_losses = result["polyak"]["losses"]
        plt.plot(polyak_losses, label=f"Batch={batch_size}")
    
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Polyak SGD Convergence for Different Batch Sizes")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, "convergence_comparison.png"))
    plt.close()


def create_contour_plots(
    results: Dict[Union[int, str], Dict],
    log_dir: str
) -> None:
    """
    Create contour plots showing the optimization paths for both optimizers.
    
    Args:
        results: Dictionary mapping batch sizes to results
        log_dir: Directory to save plots
    """
    # Define loss function to create contour
    def loss_func(x, y, training_data):
        loss = 0.0
        count = 0
        for w in training_data:
            # Move tensor to CPU before converting to numpy
            w = w.cpu().numpy()
            z = np.array([x, y]) - w - 1
            term1 = 20 * (z[0]**2 + z[1]**2)
            term2 = (z[0] + 9)**2 + (z[1] + 10)**2
            point_loss = min(term1, term2)
            loss += point_loss
            count += 1
        return loss / count
    
    # Create mesh grid for contour
    x_min, x_max = -15, 15
    y_min, y_max = -15, 15
    n_points = 100
    
    X = np.linspace(x_min, x_max, n_points)
    Y = np.linspace(y_min, y_max, n_points)
    X_grid, Y_grid = np.meshgrid(X, Y)
    
    # Get training data
    m = results[next(iter(results.keys()))]["config"]["epochs"]
    std = 0.25  # Using default from make_gaussian_cluster
    seed = 42   # Using default seed
    training_data = make_gaussian_cluster(m=m, std=std, seed=seed)
    
    # Compute loss landscape
    Z = np.zeros_like(X_grid)
    for i in range(n_points):
        for j in range(n_points):
            Z[i, j] = loss_func(X_grid[i, j], Y_grid[i, j], training_data)
    
    # Create separate contour plots for SGD and Polyak for each batch size
    for batch_size, result in results.items():
        # SGD contour plot
        plt.figure(figsize=(10, 6))
        plt.contour(X_grid, Y_grid, Z, levels=20)
        
        # Get SGD parameter history
        sgd_params = result["sgd"]["param_history"]
        sgd_x = [p[0] for p in sgd_params]
        sgd_y = [p[1] for p in sgd_params]
        
        # Plot SGD trajectory
        plt.plot(sgd_x, sgd_y, marker='.', label=f'SGD (LR={result["sgd"]["lr"]})')
        plt.plot(sgd_x[0], sgd_y[0], 'rx', markersize=10, label='Start')
        plt.plot(sgd_x[-1], sgd_y[-1], 'ro', markersize=10, label='End')
        
        plt.xlabel('x[0]')
        plt.ylabel('x[1]')
        plt.title(f'SGD Trajectory on Loss Contour (Batch Size={batch_size})')
        plt.legend()
        plt.savefig(os.path.join(log_dir, f"sgd_contour_batch{batch_size}.png"))
        plt.close()
        
        # Polyak contour plot
        plt.figure(figsize=(10, 6))
        plt.contour(X_grid, Y_grid, Z, levels=20)
        
        # Get Polyak parameter history
        polyak_params = result["polyak"]["param_history"]
        polyak_x = [p[0] for p in polyak_params]
        polyak_y = [p[1] for p in polyak_params]
        
        # Plot Polyak trajectory
        plt.plot(polyak_x, polyak_y, marker='.', label='Polyak Step Size')
        plt.plot(polyak_x[0], polyak_y[0], 'bx', markersize=10, label='Start')
        plt.plot(polyak_x[-1], polyak_y[-1], 'bo', markersize=10, label='End')
        
        plt.xlabel('x[0]')
        plt.ylabel('x[1]')
        plt.title(f'Polyak Trajectory on Loss Contour (Batch Size={batch_size})')
        plt.legend()
        plt.savefig(os.path.join(log_dir, f"polyak_contour_batch{batch_size}.png"))
        plt.close()
    
    # Create comparison plots for different batch sizes
    # SGD comparison
    plt.figure(figsize=(10, 6))
    plt.contour(X_grid, Y_grid, Z, levels=20)
    
    for batch_size, result in results.items():
        sgd_params = result["sgd"]["param_history"]
        sgd_x = [p[0] for p in sgd_params]
        sgd_y = [p[1] for p in sgd_params]
        plt.plot(sgd_x, sgd_y, marker='.', label=f'Batch={batch_size}')
    
    plt.xlabel('x[0]')
    plt.ylabel('x[1]')
    plt.title('SGD Trajectories for Different Batch Sizes')
    plt.legend()
    plt.savefig(os.path.join(log_dir, "sgd_batch_comparison.png"))
    plt.close()
    
    # Polyak comparison
    plt.figure(figsize=(10, 6))
    plt.contour(X_grid, Y_grid, Z, levels=20)
    
    for batch_size, result in results.items():
        polyak_params = result["polyak"]["param_history"]
        polyak_x = [p[0] for p in polyak_params]
        polyak_y = [p[1] for p in polyak_params]
        plt.plot(polyak_x, polyak_y, marker='.', label=f'Batch={batch_size}')
    
    plt.xlabel('x[0]')
    plt.ylabel('x[1]')
    plt.title('Polyak Trajectories for Different Batch Sizes')
    plt.legend()
    plt.savefig(os.path.join(log_dir, "polyak_batch_comparison.png"))
    plt.close()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    
    # Set device
    device = get_device()
    print(f"Using device: {device}")
    
    # Create log directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_log_dir = os.path.join(config["logging"]["log_dir"], f"week6_{timestamp}")
    os.makedirs(base_log_dir, exist_ok=True)
    
    # Save configuration
    with open(os.path.join(base_log_dir, "config.yaml"), "w") as f:
        yaml.dump(config, f)
    
    # Run experiments for each batch size
    results = {}
    
    for batch_size in config["training"]["batch_sizes"]:
        result = compare_optimizers(
            config=config,
            batch_size=batch_size,
            device=device,
            log_dir=base_log_dir
        )
        
        results[batch_size] = result
    
    # Create summary plots
    create_batch_size_comparison_plot(results, base_log_dir)
    create_convergence_comparison_plot(results, base_log_dir)
    
    # Create contour plots
    create_contour_plots(results, base_log_dir)
    
    print(f"Experiments completed. Results saved to {base_log_dir}")


if __name__ == "__main__":
    main()
