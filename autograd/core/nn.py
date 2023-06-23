import random
from .engine import Value
import math 
class Module:

    def zero_grad(self):
        """
        Set the gradients of all parameters in the module to zero.
        """
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        """
        Return a list of all parameters in the module.
        """
        return []

class Neuron(Module):

    def __init__(self, nin, nonlin=True):
        """
        Initialize a Neuron module.

        Args:
            nin: The number of input features.
            nonlin: A boolean indicating whether to apply non-linearity (ReLU) to the neuron's output.
        """
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(0)
        self.nonlin = nonlin

    def __call__(self, x):
        """
        Compute the output of the neuron.

        Args:
            x: The input value.

        Returns:
            The output value of the neuron.
        """
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return act.relu() if self.nonlin else act

    def parameters(self):
        """
        Return a list of all parameters in the neuron.
        """
        return self.w + [self.b]

    def __repr__(self):
        """
        Return a string representation of the neuron.
        """
        return f"{'ReLU' if self.nonlin else 'Linear'}Neuron({len(self.w)})"

class Layer(Module):

    def __init__(self, nin, nout, **kwargs):
        """
        Initialize a Layer module.

        Args:
            nin: The number of input features.
            nout: The number of output features.
            **kwargs: Additional arguments to be passed to the Neuron constructor.
        """
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]

    def __call__(self, x):
        """
        Compute the output of the layer.

        Args:
            x: The input value.

        Returns:
            The output value of the layer.
        """
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        """
        Return a list of all parameters in the layer.
        """
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self):
        """
        Return a string representation of the layer.
        """
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"

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
