import numpy as np
import math
import random


class Value:
    """Stores a single scalar value and its gradient."""

    def __init__(self, data=None, _children=(), _op='', label=''):
        """
        Initialize a Value object.

        Args:
            data: The scalar value to store. If None, a random value between -1 and 1 is generated.
            _children: A tuple of child Value objects.
            _op: The operation that produced this node.
            label: A label for the Value object.
        """
        if data is None:
            data = random.uniform(-1, 1)
        self.data = data
        self.grad = 0
        self.label = label
        # Internal variables used for autograd graph construction
        self._backward = lambda *a, **k: None
        self._prev = set(_children)
        self._op = _op  # The op that produced this node, for graphviz/debugging/etc

    def __add__(self, other):
        """
        Perform addition between two Value objects.

        Args:
            other: The other Value object to add.

        Returns:
            A new Value object representing the sum.
        """
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward(keep_graph=False):
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward

        return out

    def __mul__(self, other):
        """
        Perform multiplication between two Value objects.

        Args:
            other: The other Value object to multiply.

        Returns:
            A new Value object representing the product.
        """
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward(keep_graph=False):
            if keep_graph:
                self.grad += other * out.grad
                other.grad += self * out.grad
            else:
                self.grad += other.data * out.grad
                other.grad += self.data * out.grad

        out._backward = _backward

        return out

    def __pow__(self, other):
        """
        Perform exponentiation between a Value object and a scalar.

        Args:
            other: The scalar value to raise the Value object to.

        Returns:
            A new Value object representing the exponentiation.
        """
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Value(self.data ** other, (self,), f'**{other}')

        def _backward(keep_graph=False):
            if keep_graph:
                self.grad += (other * self ** (other - 1)) * out.grad
            else:
                self.grad += (other * self.data ** (other - 1)) * out.grad

        out._backward = _backward

        return out

    def relu(self):
        """
        Apply the rectified linear unit (ReLU) activation function to the Value object.

        Returns:
            A new Value object representing the output of ReLU.
        """
        out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')

        def _backward(keep_graph=False):
            self.grad += (out.data > 0) * out.grad

        out._backward = _backward

        return out

    def softmax(self):
        """
        Apply the softmax activation function to the Value object.

        Returns:
            A new Value object representing the output of softmax.
        """
        softmax = np.exp(self.data) / np.sum(np.exp(self.data), axis=1)[:, None]
        out = Value(softmax, (self,), 'softmax')

        def _backward():
            self.grad += (out.grad - np.reshape(np.sum(out.grad * softmax, 1), [-1, 1])) * softmax

        out._backward = _backward

        return out

    def backward(self, keep_graph=False):
        """
        Perform backpropagation to compute the gradients.

        Args:
            keep_graph: Whether to keep the computational graph for higher-order gradients.
        """
        # Topological order all of the children in the graph
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                # Set grad to zero to prevent previous values affecting the result.
                v.grad = 0
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        # Making grad a `Value` type allows us to track backpropagation and do higher-order gradients.
        self.grad = Value(1) if keep_graph else 1
        for v in reversed(topo):
            v._backward(keep_graph=keep_graph)

    def __neg__(self):
        """Negation of the Value object."""
        return self * -1

    def __radd__(self, other):
        """Addition of the Value object with another object."""
        return self + other

    def __sub__(self, other):
        """Subtraction of another object from the Value object."""
        return self + (-other)

    def __rsub__(self, other):
        """Subtraction of the Value object from another object."""
        return other + (-self)

    def __rmul__(self, other):
        """Multiplication of the Value object with another object."""
        return self * other

    def __truediv__(self, other):
        """Division of the Value object by another object."""
        return self * other ** -1

    def __rtruediv__(self, other):
        """Division of another object by the Value object."""
        return other * self ** -1

    def __repr__(self):
        """Representation of the Value object."""
        return f"Value(data={self.data}, grad={self.grad})"

    def __float__(self):
        """Conversion of the Value object to a float."""
        return float(self.data)
