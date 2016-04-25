import numpy as np
from ..neuron import WeightedNeuron
from ..ensemble import Ensemble

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
