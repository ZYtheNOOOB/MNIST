# CS561-MNIST Project

This is the project repo for USC CS561 HW3 MNIST project. **Implement a pure numpy neural network for hand written digit recognition task.**

Training and testing shared a limit on total running time within 30 minutes, GPU acceleration and multiprocessing not allowed.

## Dataset

10000 train and 10000 test images selected from MNIST dataset. Extract by running mnist_csv3.py.

## Preprocessing

1. Normalize

2. Split training and validation sets on a ratio of 9:1

3. Augment training data by scale, rotation and translation

4. Shuffle and create batch

## Network Specs and Hyperparams

- Structure: 4-layer MLP
- Activation: Tanh
- Loss func: Cross entropy loss
- Optimizer: Adam

**Hyperparams**

- lr: 1e-4
- batch size: 128
- epoch: 20 (with early termination)
- beta1, beta2: 0.9, 0.999
- weight decay: None
- cross validation: None

## Result

Accuracy scoring on hidden train & test set on Vocareum: 0.9723
