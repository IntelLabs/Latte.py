import numpy as np
from ..neuron import Neuron
from ..ensemble import Ensemble
import itertools

class ReLUNeuron(Neuron):
    def __init__(self):
        self.inputs = []
        self.grad_inputs = []

    def forward(self):
        self.value = max(self.input, 0.0)

    def backward(self):
        if self.input > 0.0:
            self.grad_input = self.grad
        else:
            self.grad_input = 0.0

def ReLULayer(net, input_ensemble):
    
    relu_neurons = np.empty(input_ensemble.shape, dtype='object')

    for i in range(len(relu_neurons)):
        relu_neurons[i] = ReLUNeuron()

    relu_ens = net.init_activation_ensemble(relu_neurons)
    net.add_one_to_one_connections(input_ensemble, relu_ens)

    return relu_ens
