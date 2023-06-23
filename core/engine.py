import numpy as np 
import math 
import random


class Value:
    """ stores a single scalar value and its gradient """

    def __init__(self, data=None, _children=(), _op='',label=''):
        if data is None:
          data = random.uniform(-1,1)
        self.data = data
        self.grad = 0
        self.label=label
        # internal variables used for autograd graph construction
        self._backward = lambda *a, **k: None
        self._prev = set(_children)
        self._op = _op # the op that produced this node, for graphviz / debugging / etc

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward(keep_graph=False):
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward

        return out

    def __mul__(self, other):
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
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Value(self.data**other, (self,), f'**{other}')

        def _backward(keep_graph=False):
            if keep_graph:
                self.grad += (other * self**(other-1)) * out.grad
            else:
                self.grad += (other * self.data**(other-1)) * out.grad
        out._backward = _backward

        return out

    def relu(self):
        out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')

        def _backward(keep_graph=False):
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward

        return out
    
    def softmax(self):

        out =  Value(np.exp(self.data) / np.sum(np.exp(self.data), axis=1)[:, None], (self,), 'softmax')
        softmax = out.data
        def _backward():
            self.grad += (out.grad - np.reshape(
            np.sum(out.grad * softmax, 1),
            [-1, 1]
              )) * softmax
        out._backward = _backward

        return out

    def backward(self, keep_graph=False):
        # topological order all of the children in the graph
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                # Set to grad to zero to prevent previous values effecting the
                # result.
                v.grad = 0
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        # Making grad a `Value` type allows us to track backprop and do higher
        # order gradients.
        self.grad = Value(1) if keep_graph else 1
        for v in reversed(topo):
            v._backward(keep_graph=keep_graph)

    def __neg__(self): # -self
        return self * -1

    def __radd__(self, other): # other + self
        return self + other

    def __sub__(self, other): # self - other
        return self + (-other)

    def __rsub__(self, other): # other - self
        return other + (-self)

    def __rmul__(self, other): # other * self
        return self * other

    def __truediv__(self, other): # self / other
        return self * other**-1

    def __rtruediv__(self, other): # other / self
        return other * self**-1

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"
    
    def __float__(self): return float(self.data)


        
    

   
   