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

class SparseNeuron(Module):

    def __init__(self, nin, sparsity, nonlin=True):
        """
        Initialize a SparseNeuron module.

        Args:
            nin: The number of input features.
            sparsity: The sparsity level, a value between 0 and 1.
            nonlin: A boolean indicating whether to apply non-linearity (ReLU) to the neuron's output.
        """
        assert 0 <= sparsity < 1
        n_weights = math.ceil((1 - sparsity) * nin)
        w_indices = random.sample(range(n_weights), k=n_weights)
        self.w = {i: Value(random.uniform(-0.1, 0.1)) for i in w_indices}
        self.b = Value(0)
        self.nonlin = nonlin
        self.zero_ws = {}

    def __call__(self, x, dense_grad=False):
        """
        Compute the output of the sparse neuron.

        Args:
            x: The input value.
            dense_grad: A boolean indicating whether to compute gradients for all weights (dense gradients).

        Returns:
            The output value of the sparse neuron.
        """
        if dense_grad:
            self.zero_ws = {}
            results = []
            for i, xi in enumerate(x):
                if i in self.w:
                    results.append(self.w[i] * xi)
                else:
                    self.zero_ws[i] = Value(0)
                    results.append(self.zero_ws[i] * xi)

            act = sum(results, self.b)
            return act.relu() if self.nonlin else act
        else:
            act = sum((wi * x[i] for i, wi in self.w.items()), self.b)
            return act.relu() if self.nonlin else act

    def parameters(self):
        """
        Return a list of all parameters in the sparse neuron.
        """
        return list(self.w.values()) + [self.b]

    def __repr__(self):
        """
        Return a string representation of the sparse neuron.
        """
        return f"{'ReLU' if self.nonlin else 'Linear'}Neuron({len(self.w)})"

class SparseLayer(Module):

    def __init__(self, nin, nout, sparsity=0, **kwargs):
        """
        Initialize a SparseLayer module.

        Args:
            nin: The number of input features.
            nout: The number of output features.
            sparsity: The sparsity level for the neurons in the layer.
            **kwargs: Additional arguments to be passed to the SparseNeuron constructor.
        """
        self.neurons = [SparseNeuron(nin, sparsity, **kwargs) for _ in range(nout)]

    def __call__(self, x, dense_grad=False):
        """
        Compute the output of the sparse layer.

        Args:
            x: The input value.
            dense_grad: A boolean indicating whether to compute gradients for all weights (dense gradients).

        Returns:
            The output value of the sparse layer.
        """
        out = [n(x, dense_grad=dense_grad) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        """
        Return a list of all parameters in the sparse layer.
        """
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self):
        """
        Return a string representation of the sparse layer.
        """
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"

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
        return x

    def parameters(self):
        """
        Return a list of all parameters in the sparse MLP.
        """
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        """
        Return a string representation of the sparse MLP.
        """
        main_str = '\n'.join(str(layer) for layer in self.layers)
        return f"MLP of [\n{main_str}\n]"
