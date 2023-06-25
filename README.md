### Nano-AutoGrad

This project provides a lightweight Python micro-framework for building and training neural networks from scratch based on automatic differentiation and computational graph engine.

<div align="center">
  <img src="logo.png" alt="Nano-AutoGrad Logo" width="200">
</div>

### Installation

[![Documentation](https://img.shields.io/badge/Documentation-Read%20the%20Docs-blue.svg)](https://nano-autograd.readthedocs.io/en/latest/)
[![Examples](https://img.shields.io/badge/Examples-GitHub-green.svg)](https://nano-autograd.readthedocs.io/en/latest/README.html)

## Introduction

Nano-AutoGrad is a micro-framework that allows you to build and train neural networks from scratch based on automatic differentiation and computational graphs.

## Installation

You can install Nano-AutoGrad using pip:

```bash
pip install nano-autograds
```

### Features

1. Nano-AutoGrad offers the following features:

    * Automatic Differentiation: Nano-AutoGrad automatically * computes gradients, making it easy to perform gradient-based optimization.
    * Computational Graph Engine: It leverages a computational graph representation to efficiently compute gradients and perform backpropagation.
    * Lightweight and Efficient: Nano-AutoGrad is designed to be lightweight and efficient, suitable for small to medium-sized neural networks.
    * Easy-to-Use API: The framework provides a simple and intuitive API, allowing users to define and train neural networks with ease.
    * Integration with NumPy: Nano-AutoGrad seamlessly integrates with NumPy, enabling efficient array operations and computations.

### Usage

To get started with Nano-AutoGrad, refer to the documentation for detailed usage instructions, examples, and API reference. Here are some basic steps to build and train a neural network using Nano-AutoGrad:
*  examples 1 :

    ```python
    import numpy as np
    import autograd.core.nn as nn
    import autograd.torch.optim as nn

    class MyNeuralNetwork(na.Module):
        def __init__(self):
            self.linear = na.Linear(2, 1)

        def forward(self, x):
            return self.linear(x)

    network = MyNeuralNetwork()
    optimizer = na.SGD(network.parameters(), lr=0.1)

    ```
* Example 2 :
building 'Linear Model' using torch autograd engine 

    ```python  

    import autograd.torch.nn as nn 
    import autograd.torch.tensor as Tensor
    import autograd.torch.optim as SGD
    import autograd.functiona as F

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.l1 = nn.Linear(784, 1568, name='l1')
            self.l2 = nn.Linear(1568, 392, name='l2')
            self.l3 = nn.Linear(392, 10, name='l3')

        def forward(self, x):
            z = F.relu(self.l1(x))
            z = F.relu(self.l2(z))
            out = F.log_softmax(self.l3(z))
            return out

    model = Model()
    optimizer = autograd.optim.SGD(model.parameters(), lr=5e-2, weight_decay=1e-4)
    scheduler = autograd.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.75, total_iters=num_epochs)

    ```
### Examples

The Nano-AutoGrad repository provides various examples demonstrating the usage of the framework for different tasks, such as linear regression, classification, and more. You can explore the examples directory in the repository to gain a better understanding of how to use Nano-AutoGrad in practice.
Contributing please 

### Contributions 

Nano-AutoGrad are welcome! If you have any bug reports, feature requests, or want to contribute code, please open an issue or submit a pull request on the official GitHub repository.
License

Nano-AutoGrad is released under the MIT License. Please see the LICENSE file in the repository for more details.
Acknowledgements

We would like to thank the contributors and the open-source community for their valuable contributions to Nano-AutoGrad.
Contact

For any inquiries or further information, you can reach out to the project maintainer, Youness El Brag, via email at youness.elbrag@example.com.


Please note that you may need to update the contact email address with the appropriate one.


### Credits :

1. [micrograd](https://github.com/karpathy/micrograd) Andrej karpathy
2. [ugrad](https://github.com/conscell/ugrad/tree/main)  conscell 

