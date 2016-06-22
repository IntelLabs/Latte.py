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

        self.ratio = dropout_ratio
        self.randval = 0.0

    def forward(self):
        self.randval = rand()
        if self.randval < self.ratio:
            self.value = 0.0
        
    def backward(self):
        if self.randval < self.ratio:
            self.grad_input = 0.0 
        
def DropoutLayer(net, input_ensemble, ratio=0.5):
    
    # neurons = np.array(
    #     [(0.0, 0.0, [], [], ratio, 0.0)],
    #     dtype=[
    #         ('value', float),
    #         ('grad', float),
    #         ('inputs', list),
    #         ('grad_inputs', list),
    #         ('ratio', float),
    #         ('randval', float)
    #         ]
    # )
    neurons = np.array([DropoutNeuron(ratio) for _ in range(np.prod(input_ensemble.shape))])
    neurons = neurons.reshape(input_ensemble.shape)

    dropout_ens = net.init_activation_ensemble(neurons, input_ensemble)

    return dropout_ens
