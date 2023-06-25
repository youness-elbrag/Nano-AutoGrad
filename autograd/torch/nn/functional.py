"""Functional interface"""

import numpy as np


def exp(input):
    """
    Computes the element-wise exponential of the input tensor.

    Args:
        input: The input tensor.

    Returns:
        The tensor with exponential values.
    """
    return input.exp()


def log(input):
    """
    Computes the element-wise natural logarithm of the input tensor.

    Args:
        input: The input tensor.

    Returns:
        The tensor with logarithmic values.
    """
    return input.log()


def relu(input):
    """
    Computes the element-wise rectified linear activation (ReLU) of the input tensor.

    Args:
        input: The input tensor.

    Returns:
        The tensor with ReLU-applied values.
    """
    return input.relu()


def sigmoid(input):
    """
    Computes the element-wise sigmoid activation of the input tensor.

    Args:
        input: The input tensor.

    Returns:
        The tensor with sigmoid-applied values.
    """
    return input.sigmoid()


def tanh(input):
    """
    Computes the element-wise hyperbolic tangent of the input tensor.

    Args:
        input: The input tensor.

    Returns:
        The tensor with hyperbolic tangent values.
    """
    return input.tanh()


def log_softmax(input, dim=-1):
    """
    Computes the logarithm of softmax activations along a specified dimension of the input tensor.

    Args:
        input: The input tensor.
        dim: The dimension along which to compute the softmax.

    Returns:
        The tensor with log softmax values.
    """
    return input.log_softmax(dim=dim)


def binary_cross_entropy(input, target):
    """
    Computes the binary cross entropy loss between input and target tensors.

    Args:
        input: The input tensor.
        target: The target tensor.

    Returns:
        The binary cross entropy loss.
    """
    return -(target * input.log() + (1 - target) * (1 - input).log()).sum() / target.shape[0]


def nll_loss(input, target):
    """
    Computes the negative log likelihood loss between input and target tensors.

    Args:
        input: The input tensor.
        target: The target tensor.

    Returns:
        The negative log likelihood loss.
    """
    return -(input * target).sum() / target.shape[0]


def mse_loss(input, target):
    """
    Computes the mean squared error (MSE) loss between input and target tensors.

    Args:
        input: The input tensor.
        target: The target tensor.

    Returns:
        The mean squared error loss.
    """
    return ((input - target) ** 2).mean()


def huber_loss(input, target, delta=1.0):
    """
    Computes the Huber loss between input and target tensors.

    Args:
        input: The input tensor.
        target: The target tensor.
        delta: The threshold for the absolute error.

    Returns:
        The Huber loss.
    """
    error = input - target
    abs_error = abs(error)
    quadratic = 0.5 * (error ** 2)
    linear = delta * (abs_error - 0.5 * delta)
    return np.where(abs_error <= delta, quadratic, linear).mean()
