# Deep Learning Assignment 2: CNNs and RNNs

**Author:** Ahmad Agah
**Course:** CS 440/540 - Deep Learning

## Overview

This project implements Convolutional Neural Networks (CNNs) and Recurrent Neural Networks (RNNs) for image classification and sequence prediction tasks.

## Contents

| Notebook | Description |
|----------|-------------|
| `hw2_cnn_cifar10.ipynb` | CNN experiments on CIFAR-10 dataset |
| `hw2_rnn_sequences.ipynb` | RNN experiments on stock and weather data |

## Part 1 & 2: CNN on CIFAR-10

### What I Did
- Implemented LeNet architecture adapted for 3-channel RGB images
- Trained 12 combinations of hyperparameters (3 learning rates × 2 activations × 2 loss functions)
- Visualized feature maps at the last convolution layer
- Compared 5×5 vs 3×3 kernels with ReLU activation

### Key Results

| Configuration | Train Error | Test Error |
|---------------|-------------|------------|
| tanh + lr=0.1 + CrossEntropy | 34.4% | 45.4% |
| sigmoid (all configs) | ~90% | ~90% |

### Key Findings
- **Sigmoid failed** due to vanishing gradients (gradient becomes ~0.004 after 4 layers)
- **Tanh worked** because of larger gradients and zero-centered outputs
- **Cross-entropy outperformed MSE** for classification tasks
- **ReLU with 3×3 kernels** learns effectively without vanishing gradient issues

## Part 3: RNN for Sequence Prediction

### Part 3a: Stock Price Prediction
- Trained RNN to predict 101st value from 100 previous Dow Jones daily highs
- Compared with sin(x) prediction

**Finding:** RNN performs much better on sin(x) than stock data because:
- sin(x) is deterministic and periodic
- Stock prices are non-stationary and depend on external factors not in the price history

### Part 3b: Weather Prediction
- Applied DFT to Seattle temperature data - found clear annual periodicity (~365 days)
- Trained RNN on 2 years of data to predict year 3 temperatures

**Finding:** Model captures seasonal trends but cannot predict daily fluctuations caused by external factors (pressure systems, humidity, wind)

## How to Run

1. Open notebooks in Google Colab
2. Mount Google Drive when prompted
3. Run all cells sequentially

**Requirements:**
- PyTorch
- torchvision
- yfinance
- numpy, pandas, matplotlib

## Results

All plots and trained models are saved to `Google Drive/hw2/saved/`:
- `part1a_plots.png` - Training curves for 12 configurations
- `part1b_feature_maps.png` - Feature map visualizations
- `part2_comparison.png` - 3×3 vs 5×5 kernel comparison
- `p3a_comparison.png` - DJI vs sin(x) prediction
- `p3b_dft_1yr.png`, `p3b_dft_3yr.png` - DFT analysis
- `p3b_weather_pred.png` - Weather prediction results
