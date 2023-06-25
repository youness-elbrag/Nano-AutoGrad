class Optimizer:
    def __init__(self, parameters, lr):
        """
        Base class for optimization algorithms.

        Args:
            parameters: Model parameters to optimize.
            lr: Learning rate for the optimizer.
        """
        self.parameters = list(parameters)
        self.lr = lr

    def step(self):
        """
        Perform a single optimization step.
        This method should be implemented by subclasses.
        """
        raise NotImplementedError

    def zero_grad(self):
        """
        Set gradients of all parameters to zero.
        """
        for p in self.parameters:
            p.grad *= 0.


class SGD(Optimizer):
    def __init__(self, parameters, lr, weight_decay=0):
        """
        Stochastic Gradient Descent (SGD) optimizer.

        Args:
            parameters: Model parameters to optimize.
            lr: Learning rate for the optimizer.
            weight_decay: L2 regularization weight decay factor.
        """
        super().__init__(parameters, lr)
        self.weight_decay = weight_decay

    def step(self):
        """
        Perform a single optimization step using SGD.
        """
        for p in self.parameters:
            if self.weight_decay:
                p.data -= self.lr * (p.grad + p.data * self.weight_decay)
            else:
                p.data -= self.lr * p.grad
