import random
from core.engine import Value

class Module:

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        return []


class SparseNeuron(Module):

    def __init__(self, nin, sparsity, nonlin=True):
        assert 0 <= sparsity < 1
        n_weights = math.ceil((1 - sparsity) * nin)
        w_indices = random.sample(range(n_weights), k=n_weights)
        self.w = {i: Value(random.uniform(-0.1, 0.1)) for i in w_indices}
        self.b = Value(0)
        self.nonlin = nonlin
        self.zero_ws = {}

    def __call__(self, x, dense_grad=False):
        if dense_grad:
            # We need to calculate all gradients therefore introduce zeros.
            self.zero_ws = {}
            results = []
            for i, xi in enumerate(x):
                if i in self.w:
                    results.append(self.w[i]*xi)
                else:
                    self.zero_ws[i] = Value(0)
                    results.append(self.zero_ws[i]*xi)

            act = sum(results, self.b)
            return act.relu() if self.nonlin else act
        else:
            act = sum((wi*x[i] for i, wi in self.w.items()), self.b)
            return act.relu() if self.nonlin else act        
    
    def parameters(self):
        return list(self.w.values()) + [self.b]

    def __repr__(self):
        return f"{'ReLU' if self.nonlin else 'Linear'}Neuron({len(self.w)})"


class SparseLayer(Module):

    def __init__(self, nin, nout, sparsity=0, **kwargs):
        self.neurons = [SparseNeuron(nin, sparsity, **kwargs) for _ in range(nout)]

    def __call__(self, x, dense_grad=False):
        out = [n(x, dense_grad=dense_grad) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"

class SparseMLP(Module):

    def __init__(self, nin, nouts, sparsities):
        sz = [nin] + nouts
        self.layers = [SparseLayer(sz[i], sz[i+1], sparsity=sparsities[i],
                                   nonlin=i!=len(nouts)-1) for i in range(len(nouts))]

    def __call__(self, x, dense_grad=False):
        for layer in self.layers:
            x = layer(x, dense_grad=dense_grad)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        main_str = '\n'.join(str(layer) for layer in self.layers)
        return f"MLP of [\n{main_str}\n]"