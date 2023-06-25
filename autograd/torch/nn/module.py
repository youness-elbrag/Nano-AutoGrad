from ..init import *
from ..tensor import Tensor


class Module:
    """
    Base class for neural network modules.
    """

    def __init__(self):
        """
        Initializes the Module.
        """
        self._parameters = {}
        self._modules = {}

    def __setattr__(self, name, value):
        """
        Overrides the default attribute setting behavior.

        Args:
            name: Name of the attribute.
            value: Value to be set.

        Raises:
            ValueError: If the value is not an instance of Tensor or Module.
        """
        if isinstance(value, Tensor):
            self._parameters[name] = value
        elif isinstance(value, Module):
            self._modules[name] = value
        else:
            super().__setattr__(name, value)

    def __getattr__(self, name):
        """
        Overrides the default attribute getting behavior.

        Args:
            name: Name of the attribute.

        Returns:
            The attribute value.

        Raises:
            AttributeError: If the attribute is not found.
        """
        if name in self._parameters:
            return self._parameters[name]
        elif name in self._modules:
            return self._modules[name]
        else:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __call__(self, *args, **kwargs):
        """
        Allows the Module instance to be called like a function.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            The result of the forward method.
        """
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        """
        Performs the forward pass through the module.

        This method needs to be implemented by subclasses to define the actual computation
        that takes place during the forward pass.

        Raises:
            NotImplementedError: If the forward method is not implemented in a subclass.
        """
        raise NotImplementedError("Subclasses of 'Module' must implement the 'forward' method.")

    def modules(self):
        """
        Returns an iterator over all sub-modules.

        Returns:
            An iterator over the sub-modules.
        """
        return iter(self._modules.values())

    def parameters(self):
        """
        Returns an iterator over all parameters in the module and its sub-modules.

        Returns:
            An iterator over the parameters.
        """
        for module in self._modules.values():
            yield from module.parameters()
        yield from self._parameters.values()


class Linear(Module):
    """
    Linear layer applies a linear transformation to the input: y = x W.T + b.
    """

    def __init__(self, in_features, out_features, name=""):
        """
        Initializes the Linear layer.

        Args:
            in_features: The number of input features.
            out_features: The number of output features.
            name: The name of the Linear layer (optional).
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.name = name

        bound = self.in_features ** -0.5
        self.weight = Tensor(
            np.random.uniform(-bound, bound, (out_features, in_features)),
            requires_grad=True,
            name="w_" + name
        )
        self.bias = Tensor(
            np.random.uniform(-bound, bound, (out_features,)),
            requires_grad=True,
            name="b_" + name
        )

    def forward(self, inp):
        """
        Performs a forward pass through the Linear layer.

        Args:
            inp: The input tensor.

        Returns:
            The result of the forward pass.
        """
        return inp @ self.weight.T + self.bias
