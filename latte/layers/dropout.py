import numpy as np
from ..neuron import Neuron
from ..ensemble import Ensemble
import itertools

class DropoutNeuron(Neuron):
    batch_fields = Neuron.batch_fields + ["randval"]
    zero_init_fields = Neuron.zero_init_fields + ["randval"]

    def __init__(self, dropout_ratio):
        super().__init__()
        self.inputs = []
        self.grad_inputs = []

        self.ratio = np.array([dropout_ratio], dtype=np.float32)
        self.randval = np.zeros((1,), dtype=np.float32)

    def forward(self):
        self.randval[0] = rand()
        if self.randval[0] < self.ratio[0]:
            self.value = 0.0
        
    def backward(self):
        if self.randval[0] < self.ratio[0]:
            self.grad_input = 0.0 
        
def DropoutLayer(net, input_ensemble, ratio=0.5):
    
    neurons = np.empty(input_ensemble.shape, dtype='object')

    for i in range(neurons.size):
        neurons.flat[i] = DropoutNeuron(ratio)

    dropout_ens = net.init_activation_ensemble(neurons, input_ensemble)

    return dropout_ens
