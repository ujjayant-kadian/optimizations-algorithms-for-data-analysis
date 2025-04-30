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
import shutil

from src.utils import set_seed, get_device, get_data_loader
from src.datasets import make_synthetic_linreg
from src.models.linear_regression import LinReg, get_mse_loss
from src.optim.sgd import SGD
from src.optim.polyak_sgd import PolyakSGD
from src.optim.polyak_adam import PolyakAdam


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run synthetic regression experiments")
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/synthetic_regression_config.yaml",
        help="Path to configuration YAML file"
    )
    parser.add_argument(
        "--best_lr_path",
        type=str,
        default=None,
        help="Path to a previously saved best_learning_rates.json file to skip learning rate search"
    )
    parser.add_argument(
        "--demo_mode",
        action="store_true",
        help="Run only for specific noise and seed across different batch sizes for demonstration"
    )
    parser.add_argument(
        "--demo_noise",
        type=float,
        default=1.0,
        help="Noise level to use in demo mode"
    )
    parser.add_argument(
        "--demo_seed",
        type=int,
        default=42,
        help="Seed to use in demo mode"
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
    criterion: torch.nn.Module,
    device: torch.device,
    epochs: int,
    track_step_size: bool = False
) -> Tuple[List[float], List[float]]:
    """
    Train a model with the given optimizer and data loader.
    
    Args:
        model: PyTorch model to train
        optimizer: Optimizer to use for training
        data_loader: DataLoader providing training data
        criterion: Loss function
        device: Device to run training on
        epochs: Number of epochs to train for
        track_step_size: Whether to track step sizes (for Polyak-type optimizers)
        
    Returns:
        Tuple of (training losses, step sizes if tracking step sizes)
    """
    model.train()
    losses = []
    step_sizes = [] if track_step_size else None
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        batches = 0
        
        for X_batch, y_batch in data_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            def closure():
                optimizer.zero_grad()
                output = model(X_batch)
                loss = criterion(output, y_batch)
                loss.backward()
                return loss
            
            if track_step_size:
                loss = optimizer.step(closure)
                step_sizes.append(optimizer.last_step_size)
            else:
                loss = closure()
                optimizer.step()
            
            epoch_loss += loss.item()
            batches += 1
            
        avg_epoch_loss = epoch_loss / batches
        losses.append(avg_epoch_loss)
        
    return losses, step_sizes


def find_best_lr(
    config: Dict[str, Any],
    noise_std: float,
    batch_size: Union[int, str],
    seed: int,
    device: torch.device,
    log_dir: str
) -> Optional[float]:
    """
    Find the best learning rate for Constant Step Size SGD for the given configuration.
    
    Args:
        config: Configuration dictionary
        noise_std: Noise standard deviation for data generation
        batch_size: Batch size for training (int or "full")
        seed: Random seed
        device: Device to run training on
        log_dir: Directory to save logs
        
    Returns:
        Best learning rate or None if no suitable learning rate found
    """
    print(f"Finding best LR for noise_std={noise_std}, batch_size={batch_size}, seed={seed}")
    
    # Generate synthetic data
    n_samples = config["data"]["n_samples"]
    dimension = config["data"]["dimension"]
    epochs = config["training"]["epochs"]
    
    X, y, true_w, true_b = make_synthetic_linreg(
        N=n_samples,
        d=dimension,
        noise_std=noise_std,
        seed=seed
    )
    
    # Use full batch if specified
    actual_batch_size = n_samples if batch_size == "full" else batch_size
    data_loader = get_data_loader(X, y, actual_batch_size)
    
    # Initialize results dictionary
    results = {}
    best_lr = None
    best_final_loss = float('inf')
    
    # Create log subdirectory
    lr_log_dir = os.path.join(log_dir, f"lr_search_noise{noise_std}_batch{batch_size}_seed{seed}")
    os.makedirs(lr_log_dir, exist_ok=True)
    
    # Try different learning rates
    for lr in config["optimizer"]["learning_rates"]:
        # Create model and optimizer
        model = LinReg(dimension).to(device)
        criterion = get_mse_loss()
        optimizer = SGD(model.parameters(), lr=lr)
        
        # Train model
        try:
            losses, _ = train_model(
                model=model,
                optimizer=optimizer,
                data_loader=data_loader,
                criterion=criterion,
                device=device,
                epochs=epochs
            )
            
            # Check if training diverged (NaN or very high loss)
            if np.isnan(losses[-1]) or losses[-1] > 1e6:
                print(f"  LR {lr} diverged, final loss: {losses[-1]}")
                continue
                
            # Check if loss decreased
            loss_decreased = losses[0] > losses[-1]
            
            # Calculate loss oscillation
            oscillation = 0
            if len(losses) > 2:
                # Calculate average oscillation in the second half of training
                half_idx = len(losses) // 2
                for i in range(half_idx, len(losses) - 1):
                    oscillation += abs(losses[i+1] - losses[i])
                oscillation /= (len(losses) - half_idx - 1) if len(losses) - half_idx - 1 > 0 else 1
            
            # Save results
            results[lr] = {
                "final_loss": losses[-1],
                "loss_decreased": loss_decreased,
                "oscillation": oscillation,
                "losses": losses
            }
            
            # Update best learning rate if this one is better
            if loss_decreased and losses[-1] < best_final_loss and oscillation < 0.1 * losses[-1]:
                best_lr = lr
                best_final_loss = losses[-1]
                
            print(f"  LR {lr} - Final loss: {losses[-1]:.6f}, Oscillation: {oscillation:.6f}")
            
        except Exception as e:
            print(f"  Error with LR {lr}: {str(e)}")
    
    # Save results to file
    with open(os.path.join(lr_log_dir, "lr_search_results.json"), "w") as f:
        # Convert to serializable format
        serializable_results = {str(k): {
            "final_loss": v["final_loss"],
            "loss_decreased": v["loss_decreased"],
            "oscillation": v["oscillation"],
            "losses": v["losses"]
        } for k, v in results.items()}
        json.dump(serializable_results, f, indent=2)
    
    # Plot loss curves for different learning rates
    plt.figure(figsize=(10, 6))
    for lr, result in results.items():
        plt.plot(result["losses"], label=f"LR={lr}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Loss vs. Epoch for Different Learning Rates\nNoise={noise_std}, Batch Size={batch_size}")
    plt.legend()
    plt.savefig(os.path.join(lr_log_dir, "lr_comparison.png"))
    plt.close()
    
    # If no good learning rate found, use the one with lowest final loss
    if best_lr is None and results:
        best_lr = min(results.keys(), key=lambda lr: results[lr]["final_loss"])
        print(f"No ideal learning rate found. Using LR={best_lr} with lowest final loss.")
    
    return best_lr


def compare_optimizers(
    config: Dict[str, Any],
    noise_std: float,
    batch_size: Union[int, str],
    seed: int,
    best_lr: float,
    device: torch.device,
    log_dir: str
) -> Dict[str, Any]:
    """
    Compare Constant Step Size SGD with Polyak Step Size SGD and PolyakAdam.
    
    Args:
        config: Configuration dictionary
        noise_std: Noise standard deviation for data generation
        batch_size: Batch size for training (int or "full")
        seed: Random seed
        best_lr: Best learning rate for Constant Step Size SGD
        device: Device to run training on
        log_dir: Directory to save logs
        
    Returns:
        Dictionary with results
    """
    print(f"Comparing optimizers: noise_std={noise_std}, batch_size={batch_size}, seed={seed}")
    
    # Generate synthetic data
    n_samples = config["data"]["n_samples"]
    dimension = config["data"]["dimension"]
    epochs = config["training"]["epochs"]
    
    X, y, true_w, true_b = make_synthetic_linreg(
        N=n_samples,
        d=dimension,
        noise_std=noise_std,
        seed=seed
    )
    
    # Use full batch if specified
    actual_batch_size = n_samples if batch_size == "full" else batch_size
    data_loader = get_data_loader(X, y, actual_batch_size)
    
    # Directory for comparison results
    comp_log_dir = os.path.join(
        log_dir, 
        f"comparison_noise{noise_std}_batch{batch_size}_seed{seed}"
    )
    os.makedirs(comp_log_dir, exist_ok=True)
    
    # Train with Constant Step Size SGD
    model_sgd = LinReg(dimension).to(device)
    criterion = get_mse_loss()
    optimizer_sgd = SGD(model_sgd.parameters(), lr=best_lr)
    
    sgd_losses, _ = train_model(
        model=model_sgd,
        optimizer=optimizer_sgd,
        data_loader=data_loader,
        criterion=criterion,
        device=device,
        epochs=epochs
    )
    
    # Train with Polyak Step Size SGD
    model_polyak = LinReg(dimension).to(device)
    optimizer_polyak = PolyakSGD(
        model_polyak.parameters(),
        eps=config["optimizer"]["polyak"]["eps"],
        f_star=config["optimizer"]["polyak"]["f_star"]
    )
    
    polyak_losses, polyak_step_sizes = train_model(
        model=model_polyak,
        optimizer=optimizer_polyak,
        data_loader=data_loader,
        criterion=criterion,
        device=device,
        epochs=epochs,
        track_step_size=True
    )
    
    # Train with PolyakAdam (without AMSGrad)
    model_polyak_adam = LinReg(dimension).to(device)
    optimizer_polyak_adam = PolyakAdam(
        model_polyak_adam.parameters(),
        eps=config["optimizer"]["polyak"]["eps"],
        f_star=config["optimizer"]["polyak"]["f_star"],
        alpha=config["optimizer"]["polyak_adam"]["alpha"] if "polyak_adam" in config["optimizer"] else 0.003,
        polyak_factor=config["optimizer"]["polyak_adam"]["polyak_factor"] if "polyak_adam" in config["optimizer"] else 0.3,
        amsgrad=config["optimizer"]["polyak_adam"]["amsgrad"] if "polyak_adam" in config["optimizer"] and "amsgrad" in config["optimizer"]["polyak_adam"] else False
    )
    
    polyak_adam_losses, polyak_adam_step_sizes = train_model(
        model=model_polyak_adam,
        optimizer=optimizer_polyak_adam,
        data_loader=data_loader,
        criterion=criterion,
        device=device,
        epochs=epochs,
        track_step_size=True
    )
    
    # Train with PolyakAdam with AMSGrad enabled
    model_polyak_adam_ams = LinReg(dimension).to(device)
    optimizer_polyak_adam_ams = PolyakAdam(
        model_polyak_adam_ams.parameters(),
        eps=config["optimizer"]["polyak"]["eps"],
        f_star=config["optimizer"]["polyak"]["f_star"],
        alpha=config["optimizer"]["polyak_adam"]["alpha"] if "polyak_adam" in config["optimizer"] else 0.003,
        polyak_factor=config["optimizer"]["polyak_adam"]["polyak_factor"] if "polyak_adam" in config["optimizer"] else 0.3,
        amsgrad=True  # Always enable AMSGrad for this optimizer
    )
    
    polyak_adam_ams_losses, polyak_adam_ams_step_sizes = train_model(
        model=model_polyak_adam_ams,
        optimizer=optimizer_polyak_adam_ams,
        data_loader=data_loader,
        criterion=criterion,
        device=device,
        epochs=epochs,
        track_step_size=True
    )
    
    # Save results
    results = {
        "sgd": {
            "losses": sgd_losses,
            "lr": best_lr
        },
        "polyak": {
            "losses": polyak_losses,
            "step_sizes": polyak_step_sizes
        },
        "polyak_adam": {
            "losses": polyak_adam_losses,
            "step_sizes": polyak_adam_step_sizes
        },
        "polyak_adam_ams": {
            "losses": polyak_adam_ams_losses,
            "step_sizes": polyak_adam_ams_step_sizes
        },
        "config": {
            "noise_std": noise_std,
            "batch_size": batch_size,
            "seed": seed,
            "epochs": epochs
        }
    }
    
    # Save results to file
    with open(os.path.join(comp_log_dir, "comparison_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    # Plot loss comparison
    plt.figure(figsize=(10, 6))
    plt.plot(sgd_losses, label=f"Constant Step Size (LR={best_lr})")
    plt.plot(polyak_losses, label="Polyak Step Size")
    plt.plot(polyak_adam_losses, label="PolyakAdam")
    plt.plot(polyak_adam_ams_losses, label="PolyakAdam+AMSGrad")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Loss Comparison: SGD vs Polyak vs PolyakAdam\nNoise={noise_std}, Batch Size={batch_size}")
    plt.legend()
    plt.savefig(os.path.join(comp_log_dir, "loss_comparison.png"))
    plt.close()
    
    # Plot Polyak step sizes
    plt.figure(figsize=(10, 6))
    plt.plot(polyak_step_sizes, label="Polyak SGD")
    plt.plot(polyak_adam_step_sizes, label="PolyakAdam")
    plt.plot(polyak_adam_ams_step_sizes, label="PolyakAdam+AMSGrad")
    plt.xlabel("Batch Update")
    plt.ylabel("Step Size")
    plt.title(f"Step Sizes Over Training\nNoise={noise_std}, Batch Size={batch_size}")
    plt.legend()
    plt.savefig(os.path.join(comp_log_dir, "step_sizes.png"))
    plt.close()
    
    return results


def run_demo_comparison(
    config: Dict[str, Any],
    best_lr_grid: Dict[Tuple[float, Union[int, str], int], float],
    demo_noise: float,
    demo_seed: int,
    device: torch.device,
    log_dir: str
) -> None:
    """
    Run comparison between optimizers for a specific noise level and seed across different batch sizes.
    
    Args:
        config: Configuration dictionary
        best_lr_grid: Dictionary mapping (noise_std, batch_size, seed) to best learning rate
        demo_noise: Noise level to use for demonstration
        demo_seed: Seed to use for demonstration
        device: Device to run training on
        log_dir: Directory to save logs
    """
    print(f"\nRUNNING DEMO MODE: Comparing optimizers for noise={demo_noise}, seed={demo_seed}")
    
    # Create a special demo directory
    demo_dir = os.path.join(log_dir, f"demo_noise{demo_noise}_seed{demo_seed}")
    os.makedirs(demo_dir, exist_ok=True)
    
    # Check if the specified noise and seed exist in the best_lr_grid
    demo_keys = [k for k in best_lr_grid.keys() if k[0] == demo_noise and k[2] == demo_seed]
    if not demo_keys:
        raise ValueError(f"No learning rates found for noise={demo_noise}, seed={demo_seed}. "
                         f"Please provide a valid --best_lr_path with these configurations.")
    
    # Filter batch sizes available in the demo keys
    available_batch_sizes = [k[1] for k in demo_keys]
    print(f"Running comparison for batch sizes: {available_batch_sizes}")
    
    # Run comparison for each batch size
    results = {}
    batch_comparison_data = {
        "batch_sizes": [],
        "sgd_final_losses": [],
        "polyak_final_losses": [],
        "polyak_adam_final_losses": [],
        "polyak_adam_ams_final_losses": [],
        "sgd_lrs": []
    }
    
    for batch_size in available_batch_sizes:
        key = (demo_noise, batch_size, demo_seed)
        best_lr = best_lr_grid.get(key)
        
        if best_lr is None:
            print(f"Warning: No learning rate found for batch_size={batch_size}")
            continue
        
        print(f"\nComparing batch_size={batch_size} with learning rate={best_lr}")
        
        result = compare_optimizers(
            config=config,
            noise_std=demo_noise,
            batch_size=batch_size,
            seed=demo_seed,
            best_lr=best_lr,
            device=device,
            log_dir=demo_dir
        )
        
        results[batch_size] = result
        
        # Collect data for batch size comparison plot
        batch_comparison_data["batch_sizes"].append(str(batch_size))
        batch_comparison_data["sgd_final_losses"].append(result["sgd"]["losses"][-1])
        batch_comparison_data["polyak_final_losses"].append(result["polyak"]["losses"][-1])
        batch_comparison_data["polyak_adam_final_losses"].append(result["polyak_adam"]["losses"][-1])
        batch_comparison_data["polyak_adam_ams_final_losses"].append(result["polyak_adam_ams"]["losses"][-1])
        batch_comparison_data["sgd_lrs"].append(best_lr)
    
    # Create a unified plot to show the effect of batch size
    create_batch_size_comparison_plot(batch_comparison_data, demo_dir, demo_noise, demo_seed)
    
    # Create convergence speed plot
    create_convergence_comparison_plot(results, demo_dir, demo_noise, demo_seed)
    
    # Save detailed results
    with open(os.path.join(demo_dir, "demo_results.json"), "w") as f:
        serializable_results = {str(k): v for k, v in results.items()}
        json.dump(serializable_results, f, indent=2)
    
    print(f"\nDemo comparison completed. Results saved to {demo_dir}")


def create_batch_size_comparison_plot(
    data: Dict[str, List],
    log_dir: str,
    noise: float,
    seed: int
) -> None:
    """
    Create a plot comparing final losses for different batch sizes.
    
    Args:
        data: Dictionary with batch sizes and final losses
        log_dir: Directory to save plot
        noise: Noise level used in the demonstration
        seed: Seed used in the demonstration
    """
    plt.figure(figsize=(10, 8))
    
    # Ensure data is sorted by batch size (handle 'full' special case)
    indices = list(range(len(data["batch_sizes"])))
    
    # Sort numeric batch sizes, putting 'full' at the end
    indices.sort(key=lambda i: (data["batch_sizes"][i] == "full", 
                               int(data["batch_sizes"][i]) if data["batch_sizes"][i] != "full" else float('inf')))
    
    batch_sizes = [data["batch_sizes"][i] for i in indices]
    sgd_losses = [data["sgd_final_losses"][i] for i in indices]
    polyak_losses = [data["polyak_final_losses"][i] for i in indices]
    polyak_adam_losses = [data["polyak_adam_final_losses"][i] for i in indices]
    polyak_adam_ams_losses = [data["polyak_adam_ams_final_losses"][i] for i in indices]
    sgd_lrs = [data["sgd_lrs"][i] for i in indices]
    
    x = list(range(len(batch_sizes)))
    
    # Plot final losses
    plt.subplot(2, 1, 1)
    plt.plot(x, sgd_losses, 'o-', label="SGD (Constant Step Size)")
    plt.plot(x, polyak_losses, 'x--', label="Polyak Step Size")
    plt.plot(x, polyak_adam_losses, '*-.', label="PolyakAdam")
    plt.plot(x, polyak_adam_ams_losses, '*--', label="PolyakAdam+AMSGrad")
    plt.xticks(x, batch_sizes)
    plt.xlabel("Batch Size")
    plt.ylabel("Final Loss")
    plt.title(f"Effect of Batch Size on Final Loss (Noise={noise}, Seed={seed})")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Plot learning rates
    plt.subplot(2, 1, 2)
    plt.plot(x, sgd_lrs, 'o-', color='green')
    plt.xticks(x, batch_sizes)
    plt.xlabel("Batch Size")
    plt.ylabel("Learning Rate")
    plt.title("Best Learning Rate for Each Batch Size")
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, "batch_size_comparison.png"))
    plt.close()


def create_convergence_comparison_plot(
    results: Dict[Union[int, str], Dict],
    log_dir: str,
    noise: float,
    seed: int
) -> None:
    """
    Create a plot comparing convergence speed for different batch sizes.
    
    Args:
        results: Dictionary with batch sizes and training results
        log_dir: Directory to save plot
        noise: Noise level used in the demonstration
        seed: Seed used in the demonstration
    """
    plt.figure(figsize=(15, 16))
    
    # Plot SGD convergence
    plt.subplot(4, 1, 1)
    # Convert batch sizes to strings for consistent sorting
    sorted_items = sorted(results.items(), key=lambda x: (
            str(x[0]) == "full",  # Sort "full" last
            int(x[0]) if str(x[0]) != "full" else float('inf')  # Sort numerically
        ))
    
    for batch_size, result in sorted_items:
        sgd_losses = result["sgd"]["losses"]
        plt.plot(sgd_losses, label=f"Batch={batch_size}, LR={result['sgd']['lr']:.4f}")
    
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"SGD Convergence for Different Batch Sizes (Noise={noise}, Seed={seed})")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Plot Polyak convergence
    plt.subplot(4, 1, 2)
    for batch_size, result in sorted_items:
        polyak_losses = result["polyak"]["losses"]
        plt.plot(polyak_losses, label=f"Batch={batch_size}")
    
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Polyak SGD Convergence for Different Batch Sizes (Noise={noise}, Seed={seed})")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Plot PolyakAdam convergence
    plt.subplot(4, 1, 3)
    for batch_size, result in sorted_items:
        polyak_adam_losses = result["polyak_adam"]["losses"]
        plt.plot(polyak_adam_losses, label=f"Batch={batch_size}")
    
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"PolyakAdam Convergence for Different Batch Sizes (Noise={noise}, Seed={seed})")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Plot PolyakAdam+AMSGrad convergence
    plt.subplot(4, 1, 4)
    for batch_size, result in sorted_items:
        polyak_adam_ams_losses = result["polyak_adam_ams"]["losses"]
        plt.plot(polyak_adam_ams_losses, label=f"Batch={batch_size}")
    
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"PolyakAdam+AMSGrad Convergence for Different Batch Sizes (Noise={noise}, Seed={seed})")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, "convergence_comparison.png"))
    plt.close()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    
    # Set device
    device = get_device()
    print(f"Using device: {device}")
    
    # Create log directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_log_dir = os.path.join(config["logging"]["log_dir"], timestamp)
    os.makedirs(base_log_dir, exist_ok=True)
    
    # Save configuration
    with open(os.path.join(base_log_dir, "config.yaml"), "w") as f:
        yaml.dump(config, f)
    
    # Load previously saved best learning rates if provided, otherwise do grid search
    best_lr_grid = {}
    
    if args.best_lr_path:
        print(f"Loading best learning rates from {args.best_lr_path}")
        try:
            with open(args.best_lr_path, "r") as f:
                serialized_lr_grid = json.load(f)
                
            # Convert keys back to tuples (noise_std, batch_size, seed)
            for key, value in serialized_lr_grid.items():
                # Parse the key format "noise{noise_std}_batch{batch_size}_seed{seed}"
                parts = key.replace("noise", "").replace("batch", "").replace("seed", "").split("_")
                noise_std = float(parts[0])
                
                # Handle 'full' batch size
                batch_part = parts[1]
                batch_size = int(batch_part) if batch_part.isdigit() else batch_part
                
                seed = int(parts[2])
                best_lr_grid[(noise_std, batch_size, seed)] = value
                
            # Copy the loaded file to the new log directory
            shutil.copy(args.best_lr_path, os.path.join(base_log_dir, "best_learning_rates.json"))
            print(f"Loaded {len(best_lr_grid)} learning rate configurations")
            
        except Exception as e:
            print(f"Error loading best learning rates: {str(e)}")
            print("Falling back to grid search")
            best_lr_grid = {}
    
    # Perform grid search if no valid best learning rates were loaded
    if not best_lr_grid:
        if args.demo_mode:
            raise ValueError("Demo mode requires providing a valid --best_lr_path")
            
        print("Performing learning rate grid search")
        lr_search_seed = config["data"]["seeds"][0]  # Use only the first seed for LR search
        
        for noise_std in config["data"]["noise_std_values"]:
            for batch_size in config["training"]["batch_sizes"]:
                # Find best LR for this noise and batch size configuration
                best_lr = find_best_lr(
                    config=config,
                    noise_std=noise_std,
                    batch_size=batch_size,
                    seed=lr_search_seed,
                    device=device,
                    log_dir=base_log_dir
                )
                
                # Store best LR for all seeds with this noise_std and batch_size
                for seed in config["data"]["seeds"]:
                    key = (noise_std, batch_size, seed)
                    best_lr_grid[key] = best_lr
        
        # Save best learning rates
        with open(os.path.join(base_log_dir, "best_learning_rates.json"), "w") as f:
            # Convert to serializable format
            serializable_lr_grid = {
                f"noise{k[0]}_batch{k[1]}_seed{k[2]}": v 
                for k, v in best_lr_grid.items()
            }
            json.dump(serializable_lr_grid, f, indent=2)
    
    # Run in demo mode if specified
    if args.demo_mode:
        run_demo_comparison(
            config=config,
            best_lr_grid=best_lr_grid,
            demo_noise=args.demo_noise,
            demo_seed=args.demo_seed,
            device=device,
            log_dir=base_log_dir
        )
        return
    
    # Otherwise run full comparison for all configurations
    results = {}
    
    for key, best_lr in best_lr_grid.items():
        noise_std, batch_size, seed = key
        
        # If no good learning rate found, use a small default value
        if best_lr is None:
            print(f"Warning: No good learning rate found for noise_std={noise_std}, batch_size={batch_size}, seed={seed}")
            best_lr = 0.001  # Default to a small learning rate
        
        result = compare_optimizers(
            config=config,
            noise_std=noise_std,
            batch_size=batch_size,
            seed=seed,
            best_lr=best_lr,
            device=device,
            log_dir=base_log_dir
        )
        
        results[key] = result
    
    # Generate summary statistics and plots
    generate_summary(results, base_log_dir)
    
    print(f"Experiments completed. Results saved to {base_log_dir}")


def generate_summary(results: Dict[Tuple[float, Union[int, str], int], Dict[str, Any]], log_dir: str) -> None:
    """
    Generate summary statistics and plots for the experiments.
    
    Args:
        results: Dictionary of experiment results
        log_dir: Directory to save summary results
    """
    summary_dir = os.path.join(log_dir, "summary")
    os.makedirs(summary_dir, exist_ok=True)
    
    # Create summary tables and plots
    noise_values = sorted(set(k[0] for k in results.keys()))
    
    # Convert all batch sizes to strings before sorting to avoid type comparison issues
    batch_sizes = [str(k[1]) for k in results.keys()]
    # Sort batch sizes with "full" at the end
    batch_sizes = sorted(set(batch_sizes), key=lambda x: (x == "full", x if x != "full" else "z"))
    
    # Summary of final losses
    summary_data = {
        "noise_std": [],
        "batch_size": [],
        "sgd_final_loss_mean": [],
        "polyak_final_loss_mean": [],
        "polyak_adam_final_loss_mean": [],
        "polyak_adam_ams_final_loss_mean": [],
        "sgd_final_loss_std": [],
        "polyak_final_loss_std": [],
        "polyak_adam_final_loss_std": [],
        "polyak_adam_ams_final_loss_std": []
    }
    
    for noise_std in noise_values:
        for batch_size in batch_sizes:
            # Collect results for this configuration across seeds
            config_results = [
                results[key] 
                for key in results 
                if key[0] == noise_std and str(key[1]) == batch_size
            ]
            
            if config_results:
                sgd_final_losses = [r["sgd"]["losses"][-1] for r in config_results]
                polyak_final_losses = [r["polyak"]["losses"][-1] for r in config_results]
                polyak_adam_final_losses = [r["polyak_adam"]["losses"][-1] for r in config_results]
                polyak_adam_ams_final_losses = [r["polyak_adam_ams"]["losses"][-1] for r in config_results]
                
                summary_data["noise_std"].append(noise_std)
                summary_data["batch_size"].append(batch_size)
                summary_data["sgd_final_loss_mean"].append(np.mean(sgd_final_losses))
                summary_data["polyak_final_loss_mean"].append(np.mean(polyak_final_losses))
                summary_data["polyak_adam_final_loss_mean"].append(np.mean(polyak_adam_final_losses))
                summary_data["polyak_adam_ams_final_loss_mean"].append(np.mean(polyak_adam_ams_final_losses))
                summary_data["sgd_final_loss_std"].append(np.std(sgd_final_losses))
                summary_data["polyak_final_loss_std"].append(np.std(polyak_final_losses))
                summary_data["polyak_adam_final_loss_std"].append(np.std(polyak_adam_final_losses))
                summary_data["polyak_adam_ams_final_loss_std"].append(np.std(polyak_adam_ams_final_losses))
    
    # Save summary data
    with open(os.path.join(summary_dir, "summary_data.json"), "w") as f:
        # Convert to serializable format
        serializable_summary = {
            "noise_std": summary_data["noise_std"],
            "batch_size": [str(bs) for bs in summary_data["batch_size"]],
            "sgd_final_loss_mean": summary_data["sgd_final_loss_mean"],
            "polyak_final_loss_mean": summary_data["polyak_final_loss_mean"],
            "polyak_adam_final_loss_mean": summary_data["polyak_adam_final_loss_mean"],
            "polyak_adam_ams_final_loss_mean": summary_data["polyak_adam_ams_final_loss_mean"],
            "sgd_final_loss_std": summary_data["sgd_final_loss_std"],
            "polyak_final_loss_std": summary_data["polyak_final_loss_std"],
            "polyak_adam_final_loss_std": summary_data["polyak_adam_final_loss_std"],
            "polyak_adam_ams_final_loss_std": summary_data["polyak_adam_ams_final_loss_std"]
        }
        json.dump(serializable_summary, f, indent=2)
    
    # Create comparison plots
    
    # Plot by noise level
    plt.figure(figsize=(12, 8))
    for i, batch_size in enumerate(batch_sizes):
        # Find relevant indices
        indices = [j for j, bs in enumerate(summary_data["batch_size"]) if bs == batch_size]
        
        if indices:
            noise_vals = [summary_data["noise_std"][j] for j in indices]
            sgd_means = [summary_data["sgd_final_loss_mean"][j] for j in indices]
            polyak_means = [summary_data["polyak_final_loss_mean"][j] for j in indices]
            polyak_adam_means = [summary_data["polyak_adam_final_loss_mean"][j] for j in indices]
            polyak_adam_ams_means = [summary_data["polyak_adam_ams_final_loss_mean"][j] for j in indices]
            
            plt.plot(noise_vals, sgd_means, marker='o', label=f"SGD, batch={batch_size}")
            plt.plot(noise_vals, polyak_means, marker='x', linestyle='--', label=f"Polyak, batch={batch_size}")
            plt.plot(noise_vals, polyak_adam_means, marker='*', linestyle='-.', label=f"PolyakAdam, batch={batch_size}")
            plt.plot(noise_vals, polyak_adam_ams_means, marker='*--', label=f"PolyakAdam+AMSGrad, batch={batch_size}")
    
    plt.xlabel("Noise Standard Deviation")
    plt.ylabel("Final Loss (Mean)")
    plt.title("Effect of Noise on Final Loss for Different Optimizers and Batch Sizes")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(summary_dir, "noise_effect.png"))
    plt.close()
    
    # Plot by batch size
    plt.figure(figsize=(12, 8))
    for i, noise_std in enumerate(noise_values):
        # Find relevant indices
        indices = [j for j, ns in enumerate(summary_data["noise_std"]) if ns == noise_std]
        
        if indices:
            batch_vals = [summary_data["batch_size"][j] for j in indices]
            sgd_means = [summary_data["sgd_final_loss_mean"][j] for j in indices]
            polyak_means = [summary_data["polyak_final_loss_mean"][j] for j in indices]
            polyak_adam_means = [summary_data["polyak_adam_final_loss_mean"][j] for j in indices]
            polyak_adam_ams_means = [summary_data["polyak_adam_ams_final_loss_mean"][j] for j in indices]
            
            # Handle string batch sizes for plotting
            x_positions = list(range(len(batch_vals)))
            
            plt.plot(x_positions, sgd_means, marker='o', label=f"SGD, noise={noise_std}")
            plt.plot(x_positions, polyak_means, marker='x', linestyle='--', label=f"Polyak, noise={noise_std}")
            plt.plot(x_positions, polyak_adam_means, marker='*', linestyle='-.', label=f"PolyakAdam, noise={noise_std}")
            plt.plot(x_positions, polyak_adam_ams_means, marker='*--', label=f"PolyakAdam+AMSGrad, noise={noise_std}")
    
    plt.xlabel("Batch Size Index")
    plt.ylabel("Final Loss (Mean)")
    plt.title("Effect of Batch Size on Final Loss for Different Optimizers and Noise Levels")
    plt.xticks(list(range(len(batch_sizes))), [str(bs) for bs in batch_sizes])
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(summary_dir, "batch_size_effect.png"))
    plt.close()


if __name__ == "__main__":
    main()
