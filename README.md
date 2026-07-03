# EEG Signal Classification

A PyTorch-based pipeline for classifying EEG (electroencephalogram) signals from raw brain recordings. Built as a course project for UCLA CS 147 (Introduction to Machine Learning).

## Overview

EEG signals capture the electrical activity of the brain over time. This project implements a deep learning approach to classify brain states from raw EEG data, covering the full ML pipeline from data loading to model training and evaluation.

## Project Structure

```
├── EEG_loading.ipynb     # Data exploration and preprocessing notebook
├── train.py              # Training entry point
└── src/
    ├── dataloader.py     # Custom PyTorch Dataset for EEG data loading
    ├── model.py          # Neural network model architecture (CNN/RNN)
    └── trainer.py        # Training loop, validation, and evaluation
```

## Tech Stack

- Python 3
- PyTorch -- model definition, training, and data loading
- Matplotlib -- data visualization and exploratory analysis
- Jupyter Notebook -- interactive data exploration

## Setup

```bash
pip install torch torchvision matplotlib jupyter
```

## Usage

Run training from the command line:

```bash
python train.py
```

Explore and preprocess the data interactively:

```bash
jupyter notebook EEG_loading.ipynb
```

## Architecture

The pipeline uses a custom `torch.utils.data.Dataset` (`EEGData`) to load raw EEG recordings, feeds them through a neural network defined in `src/model.py`, and trains using the `Trainer` class in `src/trainer.py`. The notebook (`EEG_loading.ipynb`) provides data exploration and visualization ahead of training.

