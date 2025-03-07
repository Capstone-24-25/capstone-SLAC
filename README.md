# Capstone-SLAC

A deep learning framework for image classification using PyTorch, designed for the SLAC project.

## Project Structure

### Data
- Contains training, testing, and validation CSV files with image paths
- Organized to facilitate easy data loading and processing

### Images
- Sample images showing original and transformed versions
- Useful for visualizing preprocessing techniques

### Models
- Saved model checkpoints and performance metrics
- JSON files from model testing
- Performance visualization plots (PNG)

### SLAC25 Package
Core functionality organized into modular components:

#### Data Handling
- **dataloader.py**: Efficient batch loading with customizable sampling strategies
  - `DataLoaderFactory` class for creating and configuring data loaders
  - Various sampling methods (sequential, random, weighted)
  - Optimized for performance with multi-worker support

- **dataset.py**: Custom dataset implementation
  - Image loading and transformation pipeline
  - Efficient memory management for large datasets

#### Model Architecture
- **models.py**: Neural network model definitions
  - `BaselineCNN`: Simple convolutional neural network
  - `ResNet`: Residual network implementation
  - Easily extensible for new architectures

#### Training Infrastructure
- **network.py**: Training and evaluation framework
  - `Wrapper`: Base class for model training setup
  - `ModelWrapper`: High-level interface for model training, validation, and testing
  - Support for test mode with reduced dataset size for quick iterations

#### Utilities
- **sampler.py**: Custom sampling strategies
- **transform.py**: Image transformation and augmentation
- **utils.py**: Helper functions and classes
  - Model evaluation metrics
  - Performance visualization
  - Early stopping implementation
  - Additional utilities

### Main Script
- **__main__.py**: Entry point with command-line interface
  - Flexible argument parsing for training configuration
  - Easy model selection and hyperparameter tuning
  - Test mode for rapid prototyping and debugging

## Usage

Basic usage:
```bash
python __main__.py --nepoch 10 --batch_size 32
```

Test mode (for quick iterations):
```bash
python __main__.py --testmode
```

## Features

- **Modular Design**: Easily swap models, datasets, and training strategies
- **Test Mode**: Quickly validate code changes with smaller datasets
- **Comprehensive Logging**: Detailed training metrics and model checkpoints
- **Early Stopping**: Prevent overfitting with validation-based early stopping
- **Learning Rate Scheduling**: Adaptive learning rate for improved convergence

# Test commit for github workflows
