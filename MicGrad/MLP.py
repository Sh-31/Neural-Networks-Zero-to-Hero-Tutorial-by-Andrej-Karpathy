


class Neuron:
    def __init__(self,nin) -> None:
        self.weights =  []
        for _ in range(nin):
            self.weights.append(Value(np.random.uniform(-1,1)))
        self.bias = Value(0)
    def __call__(self, X,actvtion='tanh'):
        act = self.bias
        for wi,xi in zip(self.weights,X):
            act += wi * xi
        if actvtion == "tanh":  
            out = act.tanh()    
        else:
            out = act # no actvtion function    
        return out 
    
    def parameters(self):
        return self.weights + [self.bias]

    def __repr__(self) -> str:
        return f"Neuron(wegihts={self.weights} ,bias={self.bias})"    

class Layer:

    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]
    
class MLP:
  
  def __init__(self, nin, nouts):
    sz = [nin] + nouts
    self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]
  
  def __call__(self, x):
    for layer in self.layers:
      x = layer(x)
    return x
  
  def parameters(self):
    return [p for layer in self.layers for p in layer.parameters()]

    

