from .init import *


class Tensor:
    grad_enabled = True

    def __init__(self, data, name="", requires_grad=False):
        """
        A class representing a tensor object.
        This class provides functionality for tensor operations and gradient computations.

        Args:
            data: The data array or value.
            name: The name of the tensor (optional).
            requires_grad: Whether to compute gradients for this tensor (default: False).
        """
        self.data = data if isinstance(data, np.ndarray) else np.array(data)
        self.shape = self.data.shape
        self.name = name
        self.requires_grad = requires_grad
        self.grad, self.grad_fn = (
            (np.zeros_like(self.data), Node(grad_fn=self.accum_grad, 
                                            next_functions=(), 
                                            name="accum")) if requires_grad and self.grad_enabled 
            else (None, None))
    
    def accum_grad(self, grad):
        """ 
        Gradient accumulation function.
        """
        self.grad += grad
        return ()
    
    @staticmethod
    def broadcast_axis(shape_left, shape_right):
        """
        Determine the axes along which broadcasting occurs between two shapes.

        Args:
            shape_left: Shape of the left tensor.
            shape_right: Shape of the right tensor.

        Returns:
            A tuple of two tuples representing the axes along which broadcasting occurs.
        """
        if shape_left == shape_right:
            return ((), ())
        
        # Determine the maximum number of dimensions between the two shapes
        left_dim = len(shape_left)
        right_dim = len(shape_right)
        result_ndim = max(left_dim, right_dim)
        
        # Pad the shapes with 1s to match the maximum number of dimensions
        left_padded = (1, ) * (result_ndim - left_dim) + shape_left
        right_padded = (1, ) * (result_ndim - right_dim) + shape_right
        
        # Store the axes along which broadcasting occurs
        left_axes = []
        right_axes = []

        # Iterate over padded shapes and compare corresponding axes
        for axis_idx, (left_axis, right_axis) in enumerate(zip(left_padded, right_padded)):
            if right_axis > left_axis:  # If the right axis is greater, broadcasting occurs for the left tensor
                left_axes.append(axis_idx)
            elif left_axis > right_axis:  # Broadcasting occurs for the right tensor
                right_axes.append(axis_idx)
        
        return tuple(left_axes), tuple(right_axes)

    @property
    def T(self):
        """
        Return the transpose of the Tensor.

        Returns:
            The transposed Tensor object.
        """
        result = Tensor(self.data.T, name=".T")

        if self.requires_grad and self.grad_enabled:
            # Define the gradient function for the transpose operation
            result.grad_fn = Node(grad_fn=lambda grad: (grad.T, ),
                                  next_functions=(self.grad_fn, ),
                                  name=".T")
            result.requires_grad = True
        
        return result

    def __add__(self, other):
        """
        Add two Tensor objects element-wise.

        Args:
            other: The Tensor object or constant value to be added.

        Returns:
            The resulting Tensor object after the addition.
        """
        other_data, other_requires_grad, other_grad_fn = (
            (other.data, other.requires_grad, other.grad_fn) if isinstance(other, Tensor)
            else (other, False, None))
        other_shape = () if isinstance(other, (int, float)) else other.shape

        result = Tensor(self.data + other_data, name="+")
        
        if (self.requires_grad or other_requires_grad) and self.grad_enabled:
            if self.shape == other_shape:
                # Gradient function for element-wise addition of tensors with same shape
                def grad_fn(grad): return (
                    grad if self.requires_grad else None, 
                    grad if other_requires_grad else None)
            else:
                # Determine the axes along which broadcasting occurs
                axis_self, axis_other = self.broadcast_axis(self.shape, other_shape)
                
                # Define the gradient function for element-wise addition
                def grad_fn(grad): return (
                    np.reshape(np.sum(grad, axis=axis_self), self.shape) if self.requires_grad else None,
                    np.reshape(np.sum(grad, axis=axis_other), other_shape) if other_requires_grad else None)
                
            result.grad_fn = Node(grad_fn=grad_fn,
                                  next_functions=(self.grad_fn, other_grad_fn),
                                  name="+")
            result.requires_grad = True

        return result

    def __mul__(self, other):
        """
        Multiply two Tensor objects element-wise.

        Args:
            other: The Tensor object or constant value to be multiplied.

        Returns:
            The resulting Tensor object after the multiplication.
        """
        other_data, other_requires_grad, other_grad_fn = (
            (other.data, other.requires_grad, other.grad_fn) if isinstance(other, Tensor)
            else (other, False, None))
        other_shape = () if isinstance(other, (int, float)) else other.shape
        
        result = Tensor(self.data * other_data, name="*")

        if (self.requires_grad or other_requires_grad) and self.grad_enabled:
            if self.shape == other_shape:
                # Gradient function for element-wise multiplication of tensors with same shape
                def grad_fn(grad): return (
                    other_data * grad if self.requires_grad else None, 
                    self.data * grad if other_requires_grad else None)
            else:
                # Determine the axes along which broadcasting occurs
                axis_self, axis_other = self.broadcast_axis(self.shape, other_shape)

                # Define the gradient function for element-wise multiplication
                def grad_fn(grad): return (
                    np.reshape(np.sum(other_data * grad, axis=axis_self), self.shape) if self.requires_grad else None, 
                    np.reshape(np.sum(self.data * grad, axis=axis_other), other_shape) if other_requires_grad else None)
                
            result.grad_fn = Node(grad_fn=grad_fn,
                                  next_functions=(self.grad_fn, other_grad_fn),
                                  name="*")
            result.requires_grad = True

        return result
    
    def __matmul__(self, other):
        """
        Perform matrix multiplication between two Tensor objects.

        Args:
            other: The Tensor object or constant value to be multiplied.

        Returns:
            The resulting Tensor object after the matrix multiplication.
        """
        other_data, other_requires_grad, other_grad_fn = (
            (other.data, other.requires_grad, other.grad_fn) if isinstance(other, Tensor)
            else (other, False, None))
        
        result = Tensor(self.data @ other_data, name="@")

        if (self.requires_grad or other_requires_grad) and self.grad_enabled:
            if self.data.ndim == other_data.ndim == 2:
                # Gradient function for matrix multiplication of 2D tensors
                def grad_fn(grad): return (
                    grad @ other_data.T if self.requires_grad else None, 
                    self.data.T @ grad if other_requires_grad else None)
            else:
                # Other cases
                if self.data.ndim == 1:
                    # Handling broadcasting for self when it is 1D
                    self_expand_axis = (0, )
                    self_expanded_shape = (1, ) + self.shape 
                else:
                    self_expand_axis = ()
                    self_expanded_shape = self.shape
                
                if other_data.ndim == 1:
                    # Handling broadcasting for other when it is 1D
                    other_expand_axis = (-1, )
                    other_expanded_shape = other.shape + (1, )
                else:
                    other_expand_axis = ()
                    other_expanded_shape = other.shape
                
                # Determine the axes for broadcasting and reduction
                result_expand_axis = self_expand_axis + other_expand_axis
                axis_self, axis_other = self.broadcast_axis(self_expanded_shape[:-2], other_expanded_shape[:-2])

                # Gradient function for matrix multiplication
                def grad_fn(grad): return (
                    np.reshape(np.sum(np.squeeze(np.expand_dims(grad, axis=result_expand_axis) @ 
                                                 np.expand_dims(other_data, axis=other_expand_axis).swapaxes(-1, -2),
                                                 axis=self_expand_axis), 
                                      axis=axis_self), self.shape) if self.requires_grad else None, 
                    np.reshape(np.sum(np.squeeze(np.expand_dims(self.data, axis=self_expand_axis).swapaxes(-1, -2) @ 
                                                 np.expand_dims(grad, axis=result_expand_axis),
                                                 axis=other_expand_axis), 
                                      axis=axis_other), other.shape) if other_requires_grad else None)
                    
            result.grad_fn = Node(grad_fn=grad_fn,
                                  next_functions=(self.grad_fn, other_grad_fn),
                                  name="@")
            result.requires_grad = True

        return result

    def __pow__(self, other):
        """
        Raise elements of Tensor object to the power of another element-wise.
        Note: The exponent 'other' is considered as a constant and not a variable, so no gradient is computed with respect to 'other'.

        Args:
            other: The exponent value or Tensor object containing the exponents.

        Returns:
            The resulting Tensor object contains bases in self raised to the exponents in other. 
        """
        other_data = other.data if isinstance(other, Tensor) else other

        result = Tensor(self.data ** other_data, name="**")

        if self.requires_grad and self.grad_enabled:
            # Define the gradient function for ** operation
            result.grad_fn = Node(grad_fn=lambda grad: (other_data * self.data ** (other_data - 1) * grad, ),
                                  next_functions=(self.grad_fn, ),
                                  name="**")
            result.requires_grad = True

        return result
    
    def sum(self, dim=None, keepdim=False):
        """
        Compute the sum of elements in the Tensor.

        Args:
            dim: The dimension or dimensions to reduce. If None, all dimensions are reduced (optional).
            keepdim: If True the axes which are reduced are left in the result as dimensions with size 1 (default: False).
        Returns:
            The resulting Tensor object representing the sum.
        """
        result = Tensor(np.sum(self.data, axis=dim, keepdims=keepdim), name="sum")

        if self.requires_grad and self.grad_enabled:
            expand_axis = dim if dim and not keepdim else ()
            # Define the gradient function for summation
            result.grad_fn = Node(grad_fn=lambda grad: (np.ones_like(self.data) * np.expand_dims(grad, axis=expand_axis), ),
                                  next_functions=(self.grad_fn, ),
                                  name="sum")
            result.requires_grad = True
        
        return result

    def exp(self):
        """
        Compute the exponential of each element in the Tensor.

        Returns:
            The resulting Tensor object representing the exponential.
        """
        result = Tensor(np.exp(self.data), name="exp")

        if self.requires_grad and self.grad_enabled:
            # Define the gradient function for exponent
            result.grad_fn = Node(grad_fn=lambda grad: (result.data * grad, ),
                                  next_functions=(self.grad_fn, ),
                                  name="exp")
            result.requires_grad = True

        return result
    
    def log(self):
        """
        Compute the natural logarithm of each element in the Tensor.

        Returns:
            The resulting Tensor object representing the logarithm.
        """
        result = Tensor(np.log(self.data), name="log")

        if self.requires_grad and self.grad_enabled:
            # Define the gradient function for logarithm
            result.grad_fn = Node(grad_fn=lambda grad: (grad / self.data, ),
                                  next_functions=(self.grad_fn, ),
                                  name="log")
            result.requires_grad = True

        return result
    
    def relu(self):
        """
        Apply the Rectified Linear Unit (ReLU) activation function element-wise.

        Returns:
            The resulting Tensor object after applying ReLU.
        """
        result = Tensor(np.maximum(0, self.data), name="relu")
        
        if self.requires_grad and self.grad_enabled:
            # Define the gradient function for ReLU
            result.grad_fn = Node(grad_fn=lambda grad: ((self.data > 0) * grad, ),
                                  next_functions=(self.grad_fn, ),
                                  name="relu")
            result.requires_grad = True

        return result

    def sigmoid(self):
        """
        Apply the sigmoid activation function element-wise.

        Returns:
            The resulting Tensor object after applying the sigmoid function.
        """
        result = Tensor(np.tanh(self.data * 0.5) * 0.5 + 0.5, name="sigmoid")
        
        if self.requires_grad and self.grad_enabled:
            # Define the gradient function for sigmoid
            result.grad_fn = Node(grad_fn=lambda grad: (result.data * (1 - result.data) * grad, ),
                                  next_functions=(self.grad_fn, ),
                                  name="sigmoid")
            result.requires_grad = True

        return result

    def tanh(self):
        """
        Apply the hyperbolic tangent (tanh) activation function element-wise.

        Returns:
            The resulting Tensor object after applying the tanh function.
        """
        result = Tensor(np.tanh(self.data), name="tanh")
        
        if self.requires_grad and self.grad_enabled:
            # Define the gradient function for tanh
            result.grad_fn = Node(grad_fn=lambda grad: ((1 - result.data**2) * grad, ),
                                  next_functions=(self.grad_fn, ),
                                  name="tanh")
            result.requires_grad = True

        return result
    
    def log_softmax(self, dim=-1):
        """
        Applies a softmax function followed by a logarithm.

        Args:
            dim: The dimension along which to compute the log-softmax (default: -1).

        Returns:
            The resulting Tensor object after applying the log-softmax function.
        """
        logits_off = self - np.max(self.data, axis=dim, keepdims=True)
        result = logits_off - logits_off.exp().sum(dim=dim, keepdim=True).log()
        result.name = "log_softmax"
        return result
    
    # Other operations
    def __neg__(self):
        return self * (-1)

    def __sub__(self, other):
        return self + (-other)
    
    def __radd__(self, other):
        return self + other
    
    def __rsub__(self, other):
        return (-self) + other

    def __rmul__(self, other):
        return self * other
    
    def __truediv__(self, other):
        return self * other ** (-1)

    def __rtruediv__(self, other):
        return self ** (-1) * other

    def __repr__(self):
        """
        Return a string representation of the Tensor object.

        Returns:
            The string representation of the Tensor object.
        """
        return f"Tensor(data={self.data}, grad={self.grad}, name={self.name} requires_grad={self.requires_grad})"

    def backward(self, gradient=None, retain_graph=False):
        """
        Perform backpropagation to compute gradients for all Node objects involved in the computation graph
        and store the gradients in the leaf Tensor nodes requiring gradient computation.

        This method traverses the computation graph, starting from the Node object associated with the current 
        Tensor object and propagating gradients to the previous Node objects using their respective gradient functions.

        Args:
            gradient: Gradient w.r.t. the tensor. 
            retain: If True, the graph used to compute the grads will be retained, otherwise it will be freed (default: False).

        Returns:
            None
        """
        stack = [(self.grad_fn, gradient if gradient is not None else np.ones_like(self.data))]
        visited = []
        while stack:
            grad_fn, grad = stack.pop()
            if not grad_fn:
                continue
            next_fns = grad_fn.next_functions
            res_for_next = grad_fn(grad)
            if grad_fn.name != "accum":
                visited.append(grad_fn)
            if next_fns == ():
                continue
            for n, r in zip(next_fns, res_for_next):
                stack.append((n, r))
        if not retain_graph:
            for grad_fn in visited:
                grad_fn.grad_fn = None


class Node:
    def __init__(self, grad_fn, next_functions, name=""):
        """
        A class representing a gradient function node in the computational graph.
        Gradient function nodes encapsulate the gradient computation and propagation
        for a specific operation in the graph.

        Args:
            grad_fn: The gradient function.
            next_functions: A tuple of next gradient function nodes.
            name: The name of the gradient function node (optional).
        """
        self.grad_fn = grad_fn
        self.next_functions = next_functions
        self.name = name

    def __call__(self, grad):
        """
        Call the gradient function with the given gradient.

        Args:
            grad: The gradient to be passed to the gradient function.

        Returns:
            The result of the gradient function.
        """
        if self.grad_fn:
            return self.grad_fn(grad)
        else:
            raise RuntimeError("Trying to backward through the graph a second time.")
    
    def __repr__(self):
        """
        Return a string representation of the gradient function node.

        Returns:
            A string representation of the gradient function node.
        """
        return f"Node={self.name}"


class no_grad:
    def __init__(self):
        """
        Initialize the 'no_grad' context manager.
        """
        self.prev = False
    
    def __enter__(self):
        """
        Disable gradient computation.
        
        This method is called when entering a 'with' block.
        It temporarily disables the computation of gradients by setting the 'grad_enabled' flag to False.
        """
        self.prev = Tensor.grad_enabled
        Tensor.grad_enabled = False

    def __exit__(self, *args):
        """
        Enable gradient computation.
        
        This method is called when exiting a 'with' block.
        It restores the previous value of the 'grad_enabled' flag, allowing gradient computation to resume.
        """
        Tensor.grad_enabled = self.prev