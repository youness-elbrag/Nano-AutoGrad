class LRScheduler:
    def __init__(self, optimizer, last_epoch=-1):
        """
        Learning rate scheduler base class.

        Args:
            optimizer: The optimizer for which to adjust the learning rate.
            last_epoch: The index of the last epoch (default: -1).
        """
        self.optimizer = optimizer
        self.base_lr = optimizer.lr
        self.last_epoch = last_epoch + 1

    def step(self):
        """
        Update the learning rate based on the current epoch.
        """
        self.last_epoch += 1
        lr = self.get_lr()
        self.optimizer.lr = lr

    def get_lr(self):
        """
        Calculate the learning rate.
        This method should be implemented by subclasses.
        """
        raise NotImplementedError


class LinearLR(LRScheduler):
    def __init__(self, optimizer, start_factor=1.0, end_factor=1.0, total_iters=1, last_epoch=-1):
        """
        Linear learning rate scheduler.

        Args:
            optimizer: The optimizer for which to adjust the learning rate.
            start_factor: The initial learning rate factor.
            end_factor: The final learning rate factor.
            total_iters: The total number of iterations to reach the final learning rate.
            last_epoch: The index of the last epoch.
        """
        super().__init__(optimizer, last_epoch)
        self.start_factor = start_factor
        self.end_factor = end_factor
        self.total_iters = total_iters

    def get_lr(self):
        """
        Calculate the learning rate based on the current epoch.

        Returns:
            The updated learning rate.
        """
        if self.last_epoch > self.total_iters:
            return self.optimizer.lr

        return self.base_lr * (
            self.start_factor + (self.end_factor - self.start_factor) * self.last_epoch / self.total_iters
        )
