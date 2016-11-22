import numpy as np
from ..neuron import Neuron
from ..ensemble import Ensemble
import latte
import itertools

class TanhNeuron(Neuron):
    def __init__(self):
        super().__init__()
        self.inputs = []
        self.grad_inputs = []

    def forward(self):
        self.value = tanh(self.input)

    def backward(self):
            self.grad_input = (self.value * (1.0 - self.value)) * self.grad

    def update_internal(self):
        pass

def TanhLayer(net, input_ensemble):
    
    tanh_neurons = np.empty(input_ensemble.shape, dtype='object')

    for i in range(len(tanh_neurons)):
        tanh_neurons[i] = TanhNeuron()

    tanh_ens = net.init_activation_ensemble(tanh_neurons, input_ensemble)
    
    tanh_ens.parallelize(phase="forward", loop_var="_neuron_index_0")
    tanh_ens.parallelize(phase="backward", loop_var="_neuron_index_0")
    
    if "value" in input_ensemble.tiling_info:
        tanh_ens.parallelize(phase="forward", loop_var="_neuron_index_1_outer")
        tanh_ens.parallelize(phase="backward", loop_var="_neuron_index_1_outer")
    else:
        tanh_ens.parallelize(phase="forward", loop_var="_neuron_index_1")
        tanh_ens.parallelize(phase="backward", loop_var="_neuron_index_1")
    
    return tanh_ens
