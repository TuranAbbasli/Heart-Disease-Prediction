import numpy as np

def relu(x):
    """ReLU activation function"""
    return np.maximum(0, x)

def sigmoid(x):
    """Sigmoid activation function"""
    return 1 / (1 + np.exp(-x))



def reluDerivative(x):
    """Derivative of the relu activation function"""
    return x > 0

def sigmoidDerivative(x):
    """Derivative of the sigmoid activation function."""
    return sigmoid(x) * (1 - sigmoid(x))