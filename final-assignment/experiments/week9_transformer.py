import torch
import os
import yaml
import json
import argparse
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
from datetime import datetime
from itertools import product

from src.models.transformer_data import TransformerDataManager
from src.models.transformer_week9 import GPTLanguageModel, calculate_baseline_loss
from src.utils import set_seed, get_device
from src.optim.polyak_sgd import PolyakSGD
from src.optim.sgd import SGD


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.
    
    Returns:
        Command line arguments
    """
    parser = argparse.ArgumentParser(description="Compare different optimizers for transformer training")
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/week9_transformer_config.yaml",
        help="Path to configuration YAML file"
    )
    parser.add_argument(
        "--tune", 
        action="store_true",
        help="Enable hyperparameter tuning (overrides config setting)"
    )
    return parser.parse_args()


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def train_with_optimizer(
    model: GPTLanguageModel,
    data_manager: TransformerDataManager,
    optimizer_name: str,
    optimizer_kwargs: Dict,
    max_iters: int,
    eval_interval: int,
    early_stopping_config: Optional[Dict] = None,
    model_save_dir: Optional[Path] = None,
    return_best: bool = True
) -> Tuple[Dict[str, List[float]], GPTLanguageModel]:
    """
    Train a transformer model with a specified optimizer.
    
    Args:
        model: The transformer model to train
        data_manager: Data manager for getting batches
        optimizer_name: Name of the optimizer (e.g., 'Adam', 'SGD', 'AdamW')
        optimizer_kwargs: Parameters for the optimizer
        max_iters: Maximum number of training iterations
        eval_interval: Interval for evaluation
        early_stopping_config: Configuration for early stopping
        model_save_dir: Directory to save model checkpoints
        return_best: Whether to return the best model based on validation loss
        
    Returns:
        Tuple containing:
            - metrics: Dictionary with training metrics (train_losses, val_losses, iterations)
            - best_model: Best model if return_best is True, otherwise final model
    """
    # Initialize optimizer
    if optimizer_name == "Polyak":
        optimizer = PolyakSGD(model.parameters(), **optimizer_kwargs)
    elif optimizer_name == "SGD":
        optimizer = SGD(model.parameters(), **optimizer_kwargs)
    else:
        optimizer_class = getattr(torch.optim, optimizer_name)
        optimizer = optimizer_class(model.parameters(), **optimizer_kwargs)
    
    # Initialize metrics tracking
    metrics = {
        'train_losses': [],
        'val_losses': [],
        'iterations': [],
    }
    
    # Add step sizes list for Polyak optimizer
    if optimizer_name == "Polyak":
        metrics['step_sizes'] = []
    
    # Setup model saving
    if model_save_dir:
        model_save_dir.mkdir(exist_ok=True, parents=True)
        best_val_loss = float('inf')
        best_model_path = model_save_dir / f"{optimizer_name}_best_model.pt"
    else:
        best_val_loss = float('inf')
    
    # Save initial model if desired
    best_model = model
    
    # Set up early stopping if configured
    early_stopping_triggered = False
    no_improvement_count = 0
    min_eval_iters = 0
    
    if early_stopping_config and early_stopping_config.get('enabled', False):
        patience = early_stopping_config.get('patience', 5)
        min_delta = early_stopping_config.get('min_delta', 0.001)
        min_epochs = early_stopping_config.get('min_epochs', 1)
        # Calculate minimum iterations before early stopping can trigger
        min_eval_iters = min_epochs * (data_manager.train_data.size(0) // data_manager.batch_size)
        min_eval_iters = min_eval_iters // eval_interval * eval_interval  # Round to eval_interval
    
    # Training loop
    for iter_num in range(max_iters):
        # Evaluation
        if iter_num % eval_interval == 0 or iter_num == max_iters - 1:
            losses = model.estimate_loss(data_manager)
            train_loss = losses['train']
            val_loss = losses['val']
            
            # Record metrics
            metrics['train_losses'].append(train_loss)
            metrics['val_losses'].append(val_loss)
            metrics['iterations'].append(iter_num)
            
            # Handle early stopping
            if early_stopping_config and early_stopping_config.get('enabled', False) and iter_num >= min_eval_iters:
                if val_loss < best_val_loss - min_delta:
                    # Improvement found
                    no_improvement_count = 0
                else:
                    # No significant improvement
                    no_improvement_count += 1
                    if no_improvement_count >= patience:
                        early_stopping_triggered = True
                        print(f"[{optimizer_name}] Early stopping triggered at iteration {iter_num}: "
                              f"No improvement for {patience} evaluations")
                        # Add early stopping info to metrics
                        metrics['early_stopped'] = True
                        metrics['early_stopping_iter'] = iter_num
                        break
            
            print(f"[{optimizer_name}] step {iter_num}: train loss {train_loss:.4f}, val loss {val_loss:.4f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                if model_save_dir:
                    torch.save(model.state_dict(), best_model_path)
                if return_best:
                    best_model = GPTLanguageModel(data_manager)
                    if model_save_dir:
                        best_model.load_state_dict(torch.load(best_model_path))
                    else:
                        # If no save dir, we need to manually copy parameters
                        best_model.load_state_dict(model.state_dict())
            
            # Record Polyak step size if applicable
            if optimizer_name == "Polyak":
                metrics['step_sizes'].append(optimizer.last_step_size)
        
        # Get batch and calculate loss
        xb, yb = data_manager.get_batch('train')
        
        if optimizer_name == "Polyak":
            def closure():
                optimizer.zero_grad()
                logits, loss = model(xb, yb)
                loss.backward()
                return loss
            
            loss = optimizer.step(closure)
        else:
            logits, loss = model(xb, yb)
            
            # Update weights
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    
    # Add final metrics
    metrics['best_val_loss'] = best_val_loss
    if not early_stopping_triggered and early_stopping_config and early_stopping_config.get('enabled', False):
        metrics['early_stopped'] = False
    
    # Save final model
    if model_save_dir:
        final_model_path = model_save_dir / f"{optimizer_name}_final_model.pt"
        torch.save(model.state_dict(), final_model_path)
    
    return metrics, best_model


def tune_sgd_learning_rate(
    data_manager: TransformerDataManager,
    lr_values: List[float],
    max_iters: int,
    eval_interval: int,
    device: torch.device,
    tune_dir: Path
) -> float:
    """
    Fine-tune the learning rate for SGD optimizer.
    
    Args:
        data_manager: Data manager for getting batches
        lr_values: List of learning rate values to try
        max_iters: Maximum number of training iterations
        eval_interval: Interval for evaluation
        device: Device to run training on
        tune_dir: Directory to save tuning results
        
    Returns:
        Best learning rate value
    """
    print("\n=== Tuning SGD Learning Rate ===")
    results = {}
    
    # Create directory for SGD tuning
    sgd_tune_dir = tune_dir / "sgd"
    sgd_tune_dir.mkdir(exist_ok=True, parents=True)
    
    # Try each learning rate
    for lr in lr_values:
        print(f"\nTrying SGD with lr={lr}")
        
        # Initialize model with same weights for fair comparison
        set_seed(data_manager.seed)
        model = GPTLanguageModel(data_manager).to(device)
        
        # Train model with this learning rate
        metrics, _ = train_with_optimizer(
            model=model,
            data_manager=data_manager,
            optimizer_name="SGD",
            optimizer_kwargs={"lr": lr},
            max_iters=max_iters,
            eval_interval=eval_interval
        )
        
        # Store results
        final_val_loss = metrics['val_losses'][-1]
        results[lr] = final_val_loss
        print(f"SGD with lr={lr}: Final val loss = {final_val_loss:.4f}")
    
    # Find best learning rate
    best_lr = min(results, key=results.get)
    best_loss = results[best_lr]
    
    # Save results
    with open(sgd_tune_dir / "results.json", "w") as f:
        json.dump({"lr_values": {str(lr): float(loss) for lr, loss in results.items()}, 
                  "best_lr": float(best_lr),
                  "best_loss": float(best_loss)}, f, indent=2)
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(list(results.keys()), list(results.values()), 'o-')
    plt.xscale('log')
    plt.xlabel('Learning Rate')
    plt.ylabel('Validation Loss')
    plt.title('SGD Learning Rate Tuning')
    plt.grid(True)
    plt.savefig(sgd_tune_dir / "lr_tuning.png")
    plt.close()
    
    print(f"\nBest SGD Learning Rate: {best_lr} (Validation Loss: {best_loss:.4f})")
    
    return best_lr


def tune_adam_hyperparams(
    data_manager: TransformerDataManager,
    lr_values: List[float],
    beta1_values: List[float],
    beta2_values: List[float],
    max_iters: int,
    eval_interval: int,
    device: torch.device,
    tune_dir: Path
) -> Tuple[float, Tuple[float, float]]:
    """
    Fine-tune the learning rate and beta parameters for Adam optimizer.
    
    Args:
        data_manager: Data manager for getting batches
        lr_values: List of learning rate values to try
        beta1_values: List of beta1 values to try
        beta2_values: List of beta2 values to try
        max_iters: Maximum number of training iterations
        eval_interval: Interval for evaluation
        device: Device to run training on
        tune_dir: Directory to save tuning results
        
    Returns:
        Tuple containing:
            - Best learning rate
            - Best beta values (beta1, beta2)
    """
    print("\n=== Tuning Adam Hyperparameters ===")
    results = {}
    
    # Create directory for Adam tuning
    adam_tune_dir = tune_dir / "adam"
    adam_tune_dir.mkdir(exist_ok=True, parents=True)
    
    # Try each combination of parameters
    for lr, beta1, beta2 in product(lr_values, beta1_values, beta2_values):
        param_key = f"lr={lr}, beta1={beta1}, beta2={beta2}"
        print(f"\nTrying Adam with {param_key}")
        
        # Initialize model with same weights for fair comparison
        set_seed(data_manager.seed)
        model = GPTLanguageModel(data_manager).to(device)
        
        # Train model with these parameters
        metrics, _ = train_with_optimizer(
            model=model,
            data_manager=data_manager,
            optimizer_name="Adam",
            optimizer_kwargs={"lr": lr, "betas": (beta1, beta2)},
            max_iters=max_iters,
            eval_interval=eval_interval
        )
        
        # Store results
        final_val_loss = metrics['val_losses'][-1]
        results[param_key] = {"loss": final_val_loss, "params": {"lr": lr, "beta1": beta1, "beta2": beta2}}
        print(f"Adam with {param_key}: Final val loss = {final_val_loss:.4f}")
    
    # Find best parameters
    best_param_key = min(results, key=lambda k: results[k]["loss"])
    best_params = results[best_param_key]["params"]
    best_loss = results[best_param_key]["loss"]
    
    # Save results
    with open(adam_tune_dir / "results.json", "w") as f:
        json.dump({
            "parameter_combinations": {k: {"loss": float(v["loss"]), "params": v["params"]} for k, v in results.items()},
            "best_params": best_params,
            "best_loss": float(best_loss)
        }, f, indent=2)
    
    # Plot results as heatmap for each beta2 value
    for beta2 in beta2_values:
        plt.figure(figsize=(10, 6))
        
        # Filter results for this beta2
        filtered_results = {(r["params"]["lr"], r["params"]["beta1"]): r["loss"] 
                           for _, r in results.items() if r["params"]["beta2"] == beta2}
        
        # Extract unique lr and beta1 values and sort them
        lr_list = sorted(set(lr for lr, _ in filtered_results.keys()))
        beta1_list = sorted(set(beta1 for _, beta1 in filtered_results.keys()))
        
        # Create 2D grid of loss values
        loss_grid = [[filtered_results.get((lr, beta1), float('nan')) for lr in lr_list] for beta1 in beta1_list]
        
        # Plot as heatmap
        plt.figure(figsize=(10, 6))
        plt.imshow(loss_grid, cmap='viridis')
        plt.colorbar(label='Validation Loss')
        plt.xticks(range(len(lr_list)), [f"{lr}" for lr in lr_list])
        plt.yticks(range(len(beta1_list)), [f"{beta1}" for beta1 in beta1_list])
        plt.xlabel('Learning Rate')
        plt.ylabel('Beta1')
        plt.title(f'Adam Hyperparameter Tuning (Beta2={beta2})')
        plt.savefig(adam_tune_dir / f"adam_tuning_beta2_{beta2}.png")
        plt.close()
    
    # Plot learning rate comparison for best beta values
    best_beta1 = best_params["beta1"]
    best_beta2 = best_params["beta2"]
    
    lr_results = {r["params"]["lr"]: r["loss"] 
                 for _, r in results.items() 
                 if r["params"]["beta1"] == best_beta1 and r["params"]["beta2"] == best_beta2}
    
    plt.figure(figsize=(10, 6))
    plt.plot(sorted(lr_results.keys()), [lr_results[lr] for lr in sorted(lr_results.keys())], 'o-')
    plt.xscale('log')
    plt.xlabel('Learning Rate')
    plt.ylabel('Validation Loss')
    plt.title(f'Adam Learning Rate (Best Betas: {best_beta1}, {best_beta2})')
    plt.grid(True)
    plt.savefig(adam_tune_dir / "adam_lr_tuning.png")
    plt.close()
    
    print(f"\nBest Adam Parameters: lr={best_params['lr']}, beta1={best_params['beta1']}, beta2={best_params['beta2']} (Validation Loss: {best_loss:.4f})")
    
    return best_params["lr"], (best_params["beta1"], best_params["beta2"])


def compare_optimizers(
    data_manager: TransformerDataManager,
    optimizers: Dict[str, Dict],
    max_iters: int,
    eval_interval: int,
    device: torch.device,
    model_save_dir: Path,
    early_stopping_config: Optional[Dict] = None,
    test_file: Optional[str] = None
) -> Dict[str, Dict]:
    """
    Compare different optimizers for transformer training.
    
    Args:
        data_manager: Data manager for getting batches
        optimizers: Dictionary mapping optimizer names to their parameters
        max_iters: Maximum number of training iterations
        eval_interval: Interval for evaluation
        device: Device to run training on
        model_save_dir: Directory to save model checkpoints
        early_stopping_config: Configuration for early stopping
        test_file: Optional test file path for final evaluation
        
    Returns:
        Dictionary with results for each optimizer
    """
    results = {}
    
    # Train with each optimizer
    for opt_name, opt_params in optimizers.items():
        print(f"\n=== Training with {opt_name} ===")
        
        # Initialize model with same weights for fair comparison
        set_seed(data_manager.seed)
        model = GPTLanguageModel(data_manager).to(device)
        
        # Create optimizer-specific save directory
        opt_save_dir = model_save_dir / opt_name
        opt_save_dir.mkdir(exist_ok=True, parents=True)
        
        # Train model with this optimizer
        metrics, best_model = train_with_optimizer(
            model=model,
            data_manager=data_manager,
            optimizer_name=opt_name,
            optimizer_kwargs=opt_params,
            max_iters=max_iters,
            eval_interval=eval_interval,
            early_stopping_config=early_stopping_config,
            model_save_dir=opt_save_dir
        )
        
        # Store results
        results[opt_name] = {
            'metrics': metrics,
            'best_model': best_model,
            'final_val_loss': metrics['val_losses'][-1],
            'best_val_loss': metrics.get('best_val_loss', metrics['val_losses'][-1]),
            'hyperparams': opt_params
        }
        
        # Evaluate on test data if provided
        if test_file:
            test_data = data_manager.load_test_dataset(test_file)
            test_loss = best_model.evaluate_test_loss(
                test_data=test_data,
                block_size=data_manager.block_size,
                batch_size=data_manager.batch_size,
                eval_iters=data_manager.eval_iters
            )
            results[opt_name]['test_loss'] = test_loss
            print(f"{opt_name} Test Loss: {test_loss:.4f}")
        
        # Print early stopping information if applicable
        if 'early_stopped' in metrics and metrics['early_stopped']:
            print(f"{opt_name}: Early stopped at iteration {metrics['early_stopping_iter']}")
    
    # Calculate and display baseline loss
    baseline_loss = calculate_baseline_loss(data_manager.vocab_size)
    print(f"\nBaseline Loss (uniform distribution): {baseline_loss:.4f}\n")
    
    # Compare final results
    print("\n=== Optimizer Comparison ===")
    for opt_name, result in results.items():
        early_stop_info = " (early stopped)" if result['metrics'].get('early_stopped', False) else ""
        print(f"{opt_name}{early_stop_info}: Best val loss = {result['best_val_loss']:.4f}")
        if test_file and 'test_loss' in result:
            print(f"  Test loss = {result['test_loss']:.4f}")
    
    # Save results to JSON
    results_file = model_save_dir / "optimizer_results.json"
    serializable_results = {}
    
    for opt_name, result in results.items():
        metrics_dict = {
            'train_losses': [float(x) for x in result['metrics']['train_losses']],
            'val_losses': [float(x) for x in result['metrics']['val_losses']],
            'iterations': result['metrics']['iterations'],
            'best_val_loss': float(result['best_val_loss'])
        }
        
        # Add early stopping info if present
        if 'early_stopped' in result['metrics']:
            metrics_dict['early_stopped'] = result['metrics']['early_stopped']
            if result['metrics']['early_stopped']:
                metrics_dict['early_stopping_iter'] = result['metrics']['early_stopping_iter']
        
        # Add step_sizes for Polyak if available
        if 'step_sizes' in result['metrics']:
            # Handle None values that might be in step_sizes
            metrics_dict['step_sizes'] = [float(x) if x is not None else 0.0 for x in result['metrics']['step_sizes']]
        
        serializable_results[opt_name] = {
            'metrics': metrics_dict,
            'final_val_loss': float(result['final_val_loss']),
            'best_val_loss': float(result['best_val_loss']),
            'hyperparams': result['hyperparams']
        }
        
        if 'test_loss' in result:
            serializable_results[opt_name]['test_loss'] = float(result['test_loss'])
    
    with open(results_file, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    return results


def plot_training_curves(results: Dict[str, Dict], log_dir: Path) -> None:
    """
    Plot training and validation curves for different optimizers.
    
    Args:
        results: Results dictionary from compare_optimizers
        log_dir: Directory to save figures
    """
    plt.figure(figsize=(12, 8))
    
    # Plot training loss
    plt.subplot(2, 1, 1)
    for opt_name, result in results.items():
        metrics = result['metrics']
        plt.plot(metrics['iterations'], metrics['train_losses'], label=f"{opt_name} (train)")
        
        # Mark early stopping point if applicable
        if 'early_stopped' in metrics and metrics['early_stopped']:
            early_stop_idx = metrics['iterations'].index(metrics['early_stopping_iter'])
            plt.plot(metrics['iterations'][early_stop_idx], metrics['train_losses'][early_stop_idx], 
                     'ro', markersize=8, label=f"{opt_name} early stop" if opt_name == list(results.keys())[0] else "")
    
    plt.title('Training Loss by Optimizer')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot validation loss
    plt.subplot(2, 1, 2)
    for opt_name, result in results.items():
        metrics = result['metrics']
        plt.plot(metrics['iterations'], metrics['val_losses'], label=f"{opt_name} (val)")
        
        # Mark early stopping point if applicable
        if 'early_stopped' in metrics and metrics['early_stopped']:
            early_stop_idx = metrics['iterations'].index(metrics['early_stopping_iter'])
            plt.plot(metrics['iterations'][early_stop_idx], metrics['val_losses'][early_stop_idx], 
                     'ro', markersize=8, label=f"{opt_name} early stop" if opt_name == list(results.keys())[0] else "")
    
    plt.title('Validation Loss by Optimizer')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    # Save the figure
    save_path = log_dir / 'optimizer_comparison.png'
    plt.savefig(save_path)
    plt.close()
    
    # Create individual plots for each optimizer
    for opt_name, result in results.items():
        plt.figure(figsize=(10, 6))
        metrics = result['metrics']
        
        plt.plot(metrics['iterations'], metrics['train_losses'], label="Training Loss")
        plt.plot(metrics['iterations'], metrics['val_losses'], label="Validation Loss")
        
        plt.title(f'{opt_name} Learning Curves')
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        save_path = log_dir / f'{opt_name}_learning_curve.png'
        plt.savefig(save_path)
        plt.close()
    
    # Plot generalization gap (NEW)
    plt.figure(figsize=(12, 6))
    for opt_name, result in results.items():
        metrics = result['metrics']
        # Calculate generalization gap (val_loss - train_loss)
        gen_gap = [val - train for val, train in zip(metrics['val_losses'], metrics['train_losses'])]
        plt.plot(metrics['iterations'], gen_gap, label=f"{opt_name}")
    
    plt.title('Generalization Gap by Optimizer')
    plt.xlabel('Iterations')
    plt.ylabel('Validation Loss - Training Loss')
    plt.legend()
    plt.grid(True)
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.3)
    
    save_path = log_dir / 'generalization_gap.png'
    plt.savefig(save_path)
    plt.close()
    
    # Plot normalized generalization gap (NEW)
    plt.figure(figsize=(12, 6))
    for opt_name, result in results.items():
        metrics = result['metrics']
        # Calculate normalized generalization gap (val_loss/train_loss - 1)
        norm_gen_gap = [(val/train - 1) for val, train in zip(metrics['val_losses'], metrics['train_losses'])]
        plt.plot(metrics['iterations'], norm_gen_gap, label=f"{opt_name}")
    
    plt.title('Normalized Generalization Gap by Optimizer')
    plt.xlabel('Iterations')
    plt.ylabel('(Validation Loss / Training Loss) - 1')
    plt.legend()
    plt.grid(True)
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.3)
    
    save_path = log_dir / 'normalized_generalization_gap.png'
    plt.savefig(save_path)
    plt.close()
    
    # Plot Polyak step size over time if available (NEW)
    if any('step_sizes' in results[opt]['metrics'] for opt in results if opt == 'Polyak'):
        plt.figure(figsize=(10, 6))
        for opt_name, result in results.items():
            if opt_name == 'Polyak' and 'step_sizes' in result['metrics']:
                metrics = result['metrics']
                # Filter out None values and replace with 0.0
                step_sizes = np.array([s if s is not None else 0.0 for s in metrics['step_sizes']])
                plt.plot(metrics['iterations'], step_sizes, 'o-')
                
                # Add exponential moving average line for clearer trend
                window = min(len(step_sizes), 5)  # Use smaller of 5 or length of data
                if window > 1:
                    weights = np.exp(np.linspace(-1., 0., window))
                    weights /= weights.sum()
                    smoothed = np.convolve(step_sizes, weights, mode='valid')
                    # Plot starting from window-1 to align with original data
                    plt.plot(metrics['iterations'][window-1:], smoothed, 'r-', 
                             label='Exponential Moving Average', linewidth=2)
        
        plt.title('Polyak Step Size Evolution')
        plt.xlabel('Iterations')
        plt.ylabel('Step Size')
        plt.grid(True)
        if 'smoothed' in locals():
            plt.legend()
        
        # Add log scale y-axis version if values vary by orders of magnitude
        plt.yscale('log')
        save_path = log_dir / 'polyak_step_size_log.png'
        plt.savefig(save_path)
        
        # Also create linear scale version
        plt.yscale('linear')
        save_path = log_dir / 'polyak_step_size.png'
        plt.savefig(save_path)
        plt.close()

    # Add early stopping summary if any optimizer was early stopped
    if any('early_stopped' in results[opt]['metrics'] and results[opt]['metrics']['early_stopped'] 
           for opt in results):
        plt.figure(figsize=(8, 4))
        
        # Create a bar chart showing training duration for each optimizer
        opt_names = []
        durations = []
        colors = []
        
        for opt_name, result in results.items():
            metrics = result['metrics']
            opt_names.append(opt_name)
            
            if 'early_stopped' in metrics and metrics['early_stopped']:
                durations.append(metrics['early_stopping_iter'])
                colors.append('orange')
            else:
                durations.append(metrics['iterations'][-1])
                colors.append('blue')
        
        plt.bar(opt_names, durations, color=colors)
        plt.title('Training Duration by Optimizer')
        plt.xlabel('Optimizer')
        plt.ylabel('Iterations')
        plt.xticks(rotation=45)
        
        # Add a legend for early stopping
        import matplotlib.patches as mpatches
        blue_patch = mpatches.Patch(color='blue', label='Completed Full Training')
        orange_patch = mpatches.Patch(color='orange', label='Early Stopped')
        plt.legend(handles=[blue_patch, orange_patch])
        
        plt.tight_layout()
        save_path = log_dir / 'early_stopping_summary.png'
        plt.savefig(save_path)
        plt.close()


def create_default_config() -> Dict[str, Any]:
    """
    Create default configuration if no config file is provided.
    
    Returns:
        Default configuration dictionary
    """
    return {
        "data": {
            "data_dir": "src/transformer-datasets",
            "dataset_file": "input_childSpeech_trainingSet.txt",
            "test_file": None,
            "batch_size": 64,
            "block_size": 256
        },
        "training": {
            "max_iters": 2000,
            "eval_interval": 200,
            "seed": 42,
            "tuning_iters": 500,
            "enable_tuning": False,
            "early_stopping": {
                "enabled": True,
                "patience": 3,
                "min_delta": 0.001,
                "min_epochs": 1
            }
        },
        "tuning": {
            "sgd": {
                "lr_values": [0.0001, 0.001, 0.01, 0.1]
            },
            "adam": {
                "lr_values": [0.0001, 0.001, 0.01],
                "beta1_values": [0.9, 0.95],
                "beta2_values": [0.999, 0.99]
            }
        },
        "optimizers": {
            "Adam": {"lr": 0.001, "betas": [0.9, 0.999]},
            "SGD": {"lr": 0.1},
            "Polyak": {"eps": 1e-8, "f_star": 0.0}
        },
        "logging": {
            "log_dir": "logs/transformer",
            "save_models": True
        }
    }


def main() -> None:
    """Main function to run the transformer training experiment."""
    args = parse_args()
    
    # Load configuration or create default
    try:
        config = load_config(args.config)
        print(f"Loaded configuration from {args.config}")
    except (FileNotFoundError, yaml.YAMLError) as e:
        print(f"Error loading config file: {e}")
        print("Using default configuration")
        config = create_default_config()
        
        # Save default config for reference
        os.makedirs(os.path.dirname(args.config), exist_ok=True)
        with open(args.config, "w") as f:
            yaml.dump(config, f, default_flow_style=False)
    
    # Override enable_tuning if --tune flag is passed
    if args.tune:
        config["training"]["enable_tuning"] = True
    
    # Set device
    device = get_device()
    print(f"Using device: {device}")
    
    # Set random seed
    seed = config["training"].get("seed", 42)
    set_seed(seed)
    
    # Create log directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_log_dir = Path(config["logging"]["log_dir"]) / f"week9_{timestamp}"
    base_log_dir.mkdir(exist_ok=True, parents=True)
    
    # Save configuration
    with open(base_log_dir / "config.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    
    # Initialize data manager
    data_manager = TransformerDataManager(
        data_dir=config["data"]["data_dir"],
        dataset_file=config["data"]["dataset_file"],
        batch_size=config["data"]["batch_size"],
        block_size=config["data"]["block_size"],
        max_iters=config["training"]["max_iters"],
        eval_interval=config["training"]["eval_interval"],
        seed=seed
    )
    
    # Print dataset info
    data_stats = data_manager.analyze_dataset()
    print(f"Dataset: {config['data']['dataset_file']}")
    print(f"Train size: {data_stats['train_size']} tokens")
    print(f"Val size: {data_stats['val_size']} tokens")
    print(f"Vocabulary size: {data_stats['vocab_size']} characters")
    
    # Create directories
    model_dir = base_log_dir / "models"
    model_dir.mkdir(exist_ok=True, parents=True)
    
    # Get optimizers config
    optimizers = config["optimizers"].copy()
    
    # Get early stopping config
    early_stopping_config = config["training"].get("early_stopping")
    if early_stopping_config and early_stopping_config.get("enabled", False):
        print(f"Early stopping enabled with patience={early_stopping_config.get('patience', 3)}, "
              f"min_delta={early_stopping_config.get('min_delta', 0.001)}")
    
    # Perform hyperparameter tuning if enabled
    if config["training"]["enable_tuning"]:
        print("\n=== Starting Hyperparameter Tuning ===")
        tuning_dir = base_log_dir / "tuning"
        tuning_dir.mkdir(exist_ok=True, parents=True)
        
        tuning_iters = config["training"]["tuning_iters"]
        eval_interval = config["training"]["eval_interval"]
        
        # Use early stopping during tuning but with shorter patience
        tuning_early_stopping = None
        if early_stopping_config and early_stopping_config.get("enabled", False):
            tuning_early_stopping = early_stopping_config.copy()
            # Use a shorter patience for tuning to speed things up
            tuning_early_stopping["patience"] = min(2, early_stopping_config.get("patience", 3))
        
        # Tune SGD learning rate
        sgd_lr = tune_sgd_learning_rate(
            data_manager=data_manager,
            lr_values=config["tuning"]["sgd"]["lr_values"],
            max_iters=tuning_iters,
            eval_interval=eval_interval,
            device=device,
            tune_dir=tuning_dir
        )
        
        # Update SGD optimizer config with best learning rate
        optimizers["SGD"]["lr"] = sgd_lr
        
        # Tune Adam hyperparameters
        adam_lr, adam_betas = tune_adam_hyperparams(
            data_manager=data_manager,
            lr_values=config["tuning"]["adam"]["lr_values"],
            beta1_values=config["tuning"]["adam"]["beta1_values"],
            beta2_values=config["tuning"]["adam"]["beta2_values"],
            max_iters=tuning_iters,
            eval_interval=eval_interval,
            device=device,
            tune_dir=tuning_dir
        )
        
        # Update Adam optimizer config with best parameters
        optimizers["Adam"]["lr"] = adam_lr
        optimizers["Adam"]["betas"] = adam_betas
        
        print("\n=== Hyperparameter Tuning Complete ===")
        print(f"Using SGD with lr={sgd_lr}")
        print(f"Using Adam with lr={adam_lr}, betas={adam_betas}")
    
    # Run comparison with optimized parameters
    results = compare_optimizers(
        data_manager=data_manager,
        optimizers=optimizers,
        max_iters=config["training"]["max_iters"],
        eval_interval=config["training"]["eval_interval"],
        device=device,
        model_save_dir=model_dir,
        early_stopping_config=early_stopping_config,
        test_file=config["data"].get("test_file")
    )
    
    # Plot results
    plot_training_curves(results, base_log_dir)
    
    print(f"Experiment completed. Results saved to {base_log_dir}")


if __name__ == "__main__":
    main() 