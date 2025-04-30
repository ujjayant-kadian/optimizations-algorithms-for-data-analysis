# Optimizations and Algorithms for Data Analysis

A comprehensive repository containing weekly assignments and a final project for the "Optimizations and Algorithms for Data Analysis" (CS7DS2) course at Trinity College Dublin.

## Repository Overview

This repository is organized by weekly assignments and culminates in a final project that applies various optimization techniques to data analysis problems:

```
optimizations-algorithms-for-data-analysis/
├── week-2/                  # Numerical differentiation and gradient descent
├── week-4/                  # Advanced optimization algorithms
├── week-6/                  # Further optimization techniques
├── week-8/                  # Advanced topics in optimization
└── final-assignment/        # Comprehensive implementation of various optimizers and models
```

## Weekly Assignments

### Week 2: Numerical Differentiation and Gradient Descent
- Implementation of numerical differentiation techniques
- Basic gradient descent algorithms
- Convergence analysis and visualization

### Week 4: Advanced Optimization Algorithms
- Implementation of advanced gradient-based methods
- Comparative analysis of optimization techniques
- Performance evaluation across different problem settings

### Week 6: Further Optimization Techniques
- Specialized optimization algorithms
- Applications to specific machine learning problems
- Convergence analysis in challenging settings

### Week 8: Advanced Topics
- Advanced optimization concepts
- Specialized applications
- Performance analysis and technique comparison

## Final Assignment

The final assignment is a comprehensive implementation of various optimization algorithms and models for data analysis, focusing on:

- Synthetic regression experiments
- Transformer models
- Week 6 experiment extensions

### Project Structure

```
final-assignment/
├── configs/                  # Configuration files for experiments
├── experiments/              # Experiment scripts
├── results/                  # Results from experiments
│   └── logs/
│       ├── synthetic_regression/
│       ├── transformer/
│       └── week6/
├── src/                      # Source code
│   ├── datasets.py           # Dataset implementations
│   ├── models/               # Model implementations
│   └── optim/                # Optimizer implementations
├── tests/                    # Unit tests
└── requirements.txt          # Python dependencies
```

### Key Features

- **Multiple Optimizer Implementations:**
  - Standard SGD
  - Polyak SGD (adaptive step size)
  - Polyak Adam (combining Adam with Polyak step size)
  
- **Model Implementations:**
  - Linear regression for synthetic data
  - Transformer models for sequence data
  - Specialized models for week 6 experiments
  
- **Comprehensive Experiment Framework:**
  - Configuration-driven experiment setup
  - Systematic hyperparameter exploration
  - Thorough results logging and visualization

## Installation

```bash
# Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r final-assignment/requirements.txt
```

## Running Experiments

### Synthetic Regression Experiment

```bash
cd final-assignment
python -m experiments.synthetic_regression --config configs/synthetic_regression_config.yaml
```

### Week 6 Experiment

```bash
cd final-assignment
python -m experiments.week6_experiment --config configs/week6_experiment_config.yaml
```

### Transformer Experiment

```bash
cd final-assignment
python -m experiments.week9_transformer --config configs/week9_transformer_config.yaml
```

## Results

Experiment results are saved in the `final-assignment/results/logs/` directory, organized by experiment type and timestamp. Results include:
- Loss plots
- Performance metrics
- Optimizer comparisons
- Trained model parameters (except for PyTorch models, which are gitignored)

## Technical Requirements

- Python 3.10+
- PyTorch
- NumPy, Matplotlib, Pandas
- YAML for configuration

## Development Guidelines

- Follow PEP8 style guidelines
- Use type hints in all public function signatures
- Write docstrings in Google style format
- Set random seeds explicitly for reproducibility
- Include basic error handling

## Author

**Student:** Ujjayant Kadian  
**Student Number:** 22330954  
**Institution:** Trinity College Dublin

## License

This project is for academic purposes for the CS7DS2 module at Trinity College Dublin.
Feel free to reuse the code with proper attribution. 