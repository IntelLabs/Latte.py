import numpy as np
from ..neuron import WeightedNeuron, BiasNeuron
from ..ensemble import Ensemble, EnsembleGroup
import latte.core

class FCNeuron(WeightedNeuron):
    def forward(self):
        for i in range_dim(self.inputs, 0):
            self.value += self.inputs[i] * self.weights[i]

    def backward(self):
        for i in range_dim(self.inputs, 0):
            self.grad_inputs[i] += self.grad * self.weights[i]

    def update_internal(self):
        for i in range_dim(self.inputs, 0):
            self.grad_weights[i] += self.grad * self.inputs[i]

def FullyConnectedLayer(net, input_ensemble, num_outputs):
    fan_in = len(input_ensemble)
    scale = np.sqrt(3.0 / fan_in)
    weights = np.random.rand(num_outputs, fan_in).astype(np.float32) * (2 * scale) - scale
    weights_grad = np.zeros_like(weights)

    neurons = np.array([FCNeuron(weights[i], weights_grad[i]) for i in range(num_outputs)])

    ens = net.init_ensemble(neurons)

    input_shape = input_ensemble.shape
    flattened = np.prod(input_shape)

    def mapping(x):
        return (range(flattened), )

    net.add_connections(input_ensemble, ens, mapping, reshape=(flattened, ))

    bias = np.zeros((num_outputs, 1), dtype=np.float32)
    grad_bias = np.zeros_like(bias)

    bias_neurons = np.array([BiasNeuron(bias[i], grad_bias[i]) for i in range(num_outputs)])

    bias_ens = net.init_activation_ensemble(bias_neurons, ens)

    if "value" in input_ensemble.tiling_info:
        ens.tile('weights', dim=1, factor=latte.core.SIMDWIDTH)

    return EnsembleGroup(ens, bias_ens)
