# Optimizations and Algorithms for Data Analysis

This directory contains implementations of various optimization algorithms and models for data analysis, focusing on synthetic regression, transformer models, and week6 experiments.

## Project Structure

```
final-assignment/
├── configs/                  # Configuration files for experiments
│   ├── synthetic_regression_config.yaml
│   ├── week6_experiment_config.yaml
│   └── week9_transformer_config.yaml
├── experiments/              # Experiment scripts
│   ├── synthetic_regression.py
│   ├── week6_experiment.py
│   └── week9_transformer.py
├── results/                  # Results from experiments
│   └── logs/
│       ├── synthetic_regression/
│       ├── transformer/
│       └── week6/
├── src/                      # Source code
│   ├── datasets.py           # Dataset implementations
│   ├── models/               # Model implementations
│   │   ├── linear_regression.py
│   │   ├── transformer_data.py
│   │   ├── transformer_week9.py
│   │   └── week6_model.py
│   ├── optim/                # Optimizer implementations
│   │   ├── polyak_adam.py
│   │   ├── polyak_sgd.py
│   │   └── sgd.py
│   ├── transformer-datasets/ # Datasets for transformer models
│   └── utils.py              # Utility functions
├── tests/                    # Unit tests
├── requirements.txt          # Python dependencies
```

## Installation

```bash
pip install -r requirements.txt
```

## Running Experiments

### 1. Synthetic Regression Experiment

This experiment compares different optimizers on synthetic linear regression data.

```bash
python -m experiments.synthetic_regression --config configs/synthetic_regression_config.yaml
```

Options:
- `--config`: Path to configuration file (default: `configs/synthetic_regression_config.yaml`)
- `--best_lr_path`: Path to previously saved best learning rates to skip learning rate search
- `--demo_mode`: Run only for specific noise and seed across different batch sizes
- `--demo_noise`: Noise level to use in demo mode (default: 1.0)
- `--demo_seed`: Seed to use in demo mode (default: 42)

### 2. Week 6 Experiment

This experiment focuses on specific optimization algorithms from week 6.

```bash
python -m experiments.week6_experiment --config configs/week6_experiment_config.yaml
```

Options:
- `--config`: Path to configuration file (default: `configs/week6_experiment_config.yaml`)

### 3. Transformer Experiment (Week 9)

This experiment trains transformer models with different optimizers.

```bash
python -m experiments.week9_transformer --config configs/week9_transformer_config.yaml
```

Options:
- `--config`: Path to configuration file (default: `configs/week9_transformer_config.yaml`)

## Configuration Files

Each experiment has a corresponding YAML configuration file in the `configs/` directory:

### Synthetic Regression Config

```yaml
data:
  n_samples: 1000  # Number of data points
  dimension: 20    # Feature dimension
  noise_std_values: [0.1, 1.0, 5.0]  # Noise standard deviations to test
  seeds: [0, 42, 1337]  # Random seeds for reproducibility

training:
  epochs: 20  # Number of training epochs
  batch_sizes: [1, 16, 64, 256, "full"]  # Batch sizes to test

optimizer:
  learning_rates: [0.001, 0.01, 0.1, 1.0]  # Learning rates for grid search
  
  polyak:
    eps: 1.0e-8  # Small constant for numerical stability
    f_star: 0.0  # Known minimum value of the loss function
    
  polyak_adam:
    alpha: 0.003  # Base learning rate
    polyak_factor: 0.3  # Factor for Polyak step size
    amsgrad: false  # Whether to use the AMSGrad variant

logging:
  log_dir: "results/logs/synthetic_regression"
  save_interval: 1  # Save logs every N epochs
```

Similar configuration structures exist for the week6 and transformer experiments.

## Results

Experiment results are logged in the `results/logs/` directory, organized by experiment type:

- `results/logs/synthetic_regression/`: Results from synthetic regression experiments
- `results/logs/transformer/`: Results from transformer experiments
- `results/logs/week6/`: Results from week6 experiments

Each experiment creates subdirectories with timestamps and experiment parameters for organization. Results include:
- Loss plots
- Performance metrics
- Optimizer comparisons
- Trained model parameters (in some cases)

## Models

### Linear Regression (`src/models/linear_regression.py`)
A simple linear regression model used in synthetic experiments.

### Week6 Model (`src/models/week6_model.py`)
Model implementation specific to the week6 experiments.

### Transformer Models (`src/models/transformer_week9.py`, `src/models/transformer_data.py`)
Transformer model implementations for sequence data processing.

## Optimizers

### SGD (`src/optim/sgd.py`)
Standard Stochastic Gradient Descent implementation.

### PolyakSGD (`src/optim/polyak_sgd.py`)
SGD with Polyak step size that adapts based on the current loss value and a target minimum.

### PolyakAdam (`src/optim/polyak_adam.py`)
Adam optimizer enhanced with Polyak step size calculations for potentially faster convergence.

## Utility Functions

Utility functions for seeding, device management, data loading, etc. are available in `src/utils.py`.

