## Description
This is the starter code for Assignment 2. Please edit the files while keeping the structure as similar as possible.

The python script main.py trains a model on CIFAR10 dataset using PyTorch. It allows you to specify the number of epochs, model_class, batch size, learning rate, and L2 regularization strength.

## Example Command
```python
python main.py --epochs 10 --model_class 'Plain-Old-CIFAR10-FC' --batch_size 128 --learning_rate 0.01 --l2_regularization 0.0001
```

## Options
- epochs (int): Number of epochs for training (default: 100).
- model_class (str): Model class name (default: 'Plain-Old-CIFAR10-FC'). Choices:
  - FC: `Plain-Old-CIFAR10-FC`, `D-shuffletruffle-FC`, `N-shuffletruffle-FC`
  - CNN: `Plain-Old-CIFAR10-CNN`, `D-shuffletruffle-CNN`, `N-shuffletruffle-CNN`
  - Attention: `Plain-Old-CIFAR10-Attention`, `D-shuffletruffle-Attention`, `N-shuffletruffle-Attention`
- batch_size (int): Batch size for training (default: 128).
- learning_rate (float): Learning rate for the optimizer (default: 0.01).
- l2_regularization (float): L2 regularization strength (default: 0.0).
