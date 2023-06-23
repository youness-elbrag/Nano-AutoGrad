from ..init import *
from ..tensor import Tensor


class Module:
    def __init__(self):
        """
        Base class for neural network modules.
        """
        self._parameters = {}
        self._modules = {}

    def __setattr__(self, name, value):
        """
        Overrides the default attribute setting behavior.

        This method is called when an attribute is set on an instance of the Module class.
        It handles special cases for parameters and sub-modules.
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

        This method is called when an attribute is accessed on an instance of the Module class.
        It handles special cases for parameters and sub-modules.
        """
        if name in self._parameters:
            return self._parameters[name]
        elif name in self._modules:
            return self._modules[name]
        else:
            return super().__getattr__(name)

    def __call__(self, *args, **kwargs):
        """
        Allows the Module instance to be called like a function.

        This method enables the Module instance to be called like a function
        by forwarding the call to the forward method. Any arguments passed to the call
        are forwarded to the forward method.

        Returns:
        - The result of the forward method.
        """
        return self.forward(*args, **kwargs)
    
    def forward(self, *args, **kwargs):
        """
        Performs the forward pass through the module.

        This method performs the forward pass through the module.
        It should be implemented by subclasses to define the actual computation
        that takes place during the forward pass.
        """
        raise NotImplementedError

    def modules(self):
        """
        Returns an iterator over all sub-modules.

        This method allows access to all the sub-modules contained within the current module.
        It returns an iterator that can be used to iterate over the sub-modules.
        """
        return iter(self._modules.values())  # Return an iterator over the sub-modules

    def parameters(self):
        """
        Returns an iterator over all parameters in the module and its sub-modules.

        This method provides access to all the parameters present in the module and its sub-modules.
        It returns an iterator that can be used to iterate over the parameters.
        """
        for module in self._modules.values():
            yield from module.parameters()  # Recursively yield parameters from sub-modules
        yield from self._parameters.values()  # Yield parameters from the current module


class Linear(Module):
    def __init__(self, in_features, out_features, name=""):
        """
        Linear layer applies a linear transformation to the input: y = x W.T + b
        input x [*, in_features]
        learnable weights W [out_features, in_features]
        bias b [out_features]
        output y [*, out_features]

        Args:
            in_features: The number of input features.
            out_features: The number of output features.
            name: The name of the Linear layer (optional).
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.name = name

        # Initialize weight and bias with random values
        bound = self.in_features ** -0.5
        self.weight = Tensor(np.random.uniform(-bound, bound, (out_features, in_features)), 
                             requires_grad=True, 
                             name="w_" + name)
        self.bias = Tensor(np.random.uniform(-bound, bound, (out_features, )), 
                           requires_grad=True, 
                           name="b_" + name)

    def forward(self, inp):
        """
        Perform a forward pass through the Linear layer.

        Args:
            inp: The input tensor.

        Returns:
            The result of the forward pass.
        """
        return (inp @ self.weight.T + self.bias)


