import numpy as np
from ..neuron import Neuron
from ..ensemble import Ensemble

class WeightedNeuron(Neuron):
    def __init__(self, weights, bias):
        super().__init__()
        self.inputs = []
        self.grad_inputs = []

        self.weights = weights
        self.grad_weights = np.zeros_like(weights)

        self.bias = bias
        self.grad_bias = np.zeros_like(bias)

    def forward(self):
        for i, input_idx in enumerate_mapping(self.inputs):
            self.value += self.inputs[input_idx] * self.weights[i]
        self.value += self.bias[0]

    def backward(self):
        for i, input_idx in enumerate_mapping(self.inputs):
            self.grad_inputs[input_idx] += self.grad * self.weights[i]
        for i, input_idx in enumerate_mapping(self.inputs):
            self.grad_weights[i] += self.grad * self.inputs[input_idx]
        self.grad_bias[0] += self.grad

def FullyConnectedLayer(net, input_ensemble, num_outputs):
    fan_in = len(input_ensemble)
    scale = np.sqrt(3.0 / fan_in)
    weights = np.random.rand(num_outputs, fan_in).astype(np.float32) * (2 * scale) - scale
    bias = np.zeros((num_outputs, 1), dtype=np.float32)
    neurons = np.array([WeightedNeuron(weights[i], bias[i]) for i in range(num_outputs)])
    ens = Ensemble(neurons)
    net.add_ensemble(ens)
    input_shape = input_ensemble.shape
    def mapping(x):
        return prod([range(d) for d in input_shape])
    net.add_connections(input_ensemble, ens, mapping)
    return ens
