# EEG Signal Classification

Deep learning model for classifying EEG (electroencephalogram) signals using PyTorch. Built as a course project for UCLA CS 147 (Introduction to Machine Learning).

## Overview

EEG signals are electrical activity recordings from the brain. This project applies deep learning (CNN/RNN) to classify brain states from raw EEG data.

## Project Structure

```
├── EEG_loading.ipynb     # Data exploration and preprocessing notebook
├── train.py              # Training entry point
└── src/
    ├── dataloader.py     # Custom PyTorch Dataset for EEG data
    ├── model.py          # CNN/RNN model architecture
    └── trainer.py        # Training and evaluation loop
```

## Setup

```bash
pip install torch torchvision matplotlib
```

## Usage

```bash
python train.py
```

Or explore the data interactively:

```bash
jupyter notebook EEG_loading.ipynb
```

## Tech Stack

- Python, PyTorch
- Custom `torch.utils.data.Dataset` for EEG loading
- Matplotlib for visualization
