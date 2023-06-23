"""Functional interface"""


def exp(input):
    return input.exp()


def log(input):
    return input.log()


def relu(input):
    return input.relu()


def sigmoid(input):
    return input.sigmoid()


def tanh(input):
    return input.tanh()


def log_softmax(input, dim=-1):
    return input.log_softmax(dim=dim)


def binary_cross_entropy(input, target):
    return -(target * input.log() + (1 - target) * (1 - input).log()).sum() / target.shape[0]


def nll_loss(input, target):
    return -(input * target).sum() / target.shape[0]