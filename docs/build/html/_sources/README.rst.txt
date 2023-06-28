Get started
===========

Nano-AutoGrad
=============

Nano-AutoGrad is a micro-framework that enables building and training neural networks from scratch based on an autodifferentiation (auto-diff) engine and computational graph. It implements backpropagation (reverse-mode autodiff) over a dynamically built Directed Acyclic Graph (DAG). The framework also includes a small neural networks library on top of the autodifferentiation engine, providing a PyTorch-like API. Both components are compact, with approximately 100 lines of code for the autodifferentiation engine and 50 lines of code for the neural networks library. Nano-AutoGrad is designed to be lightweight and potentially useful for educational purposes.

Installation
------------

To install Nano-AutoGrad, you can use pip:

.. code-block:: shell

   pip install nano-autogeads

Usage
-----

The core engine of Nano-AutoGrad provides the ability to build and train neural networks. Here are two examples of models you can create using Nano-AutoGrad:

MLP (Multi-Layer Perceptron)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The `MLP` class represents a multi-layer perceptron neural network model.

.. code-block:: python
    
   from autograd.core.nn import MLP , layer

   class MLP(Module):
   
       def __init__(self, nin, nouts):
           """
           Initialize an MLP (Multi-Layer Perceptron) module.
   
           Args:
               nin: The number of input features.
               nouts: A list of the number of output features for each layer.
           """
           sz = [nin] + nouts
           self.layers = [Layer(sz[i], sz[i + 1], nonlin=i != len(nouts) - 1) for i in range(len(nouts))]
   
       def __call__(self, x):
           """
           Compute the output of the MLP.
   
           Args:
               x: The input value.
   
           Returns:
               The output value of the MLP.
           """
           for layer in self.layers:
               x = layer(x)
           return x
   
       def parameters(self):
           """
           Return a list of all parameters in the MLP.
           """
           return [p for layer in self.layers for p in layer.parameters()]
   
       def __repr__(self):
           """
           Return a string representation of the MLP.
           """
           return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"

SparseMLP
~~~~~~~~~

The `SparseMLP` class represents a sparse multi-layer perceptron neural network model.

.. code-block:: python

   from autograd.core.Spares_nn import SparseMLP , SparseLayer

   class SparseMLP(Module):
   
       def __init__(self, nin, nouts, sparsities):
           """
           Initialize a SparseMLP module.
   
           Args:
               nin: The number of input features.
               nouts: A list of the number of output features for each layer.
               sparsities: A list of sparsity levels for each layer.
           """
           sz = [nin] + nouts
           self.layers = [SparseLayer(sz[i], sz[i + 1], sparsity=sparsities[i], nonlin=i != len(nouts) - 1) for i in range(len(nouts))]
   
       def __call__(self, x, dense_grad=False):
           """
           Compute the output of the sparse MLP.
   
           Args:
               x: The input value.
               dense_grad: A boolean indicating whether to compute gradients for all weights (dense gradients).
   
           Returns:
               The output value of the sparse MLP.
           """
           for layer in self.layers:
               x = layer(x, dense_grad=dense_grad)


Linear Model 
~~~~~~~~~~~~

building `Lieanr Model` using torch autograd engine 

.. code-block:: python   
    
    import autograd.torch.nn as nn 
    import autograd.torch.tensor as Tensor
    import autograd.torch.optim as SGD
    import autograd.functiona; as F

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

More Examples 
~~~~~~~~~~~~~
Visit Repo code  `Github` using torch autograd engine 

