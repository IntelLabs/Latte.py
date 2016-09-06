import numpy as np
from ..neuron import Neuron
from ..ensemble import Ensemble
import itertools
import latte

class ConcatNeuron(Neuron):
    #batch_fields     = Neuron.batch_fields + ["mask_j", "mask_k"]
    #zero_init_fields = Neuron.zero_init_fields + ["mask_j", "mask_k"]

    def __init__(self):
        super().__init__()
        self.inputs = []
        self.grad_inputs = []

        #self.mask_j = 0
        #self.mask_k = 0

    def forward(self):
        self.value = self.inputs[0,0,0]

    def backward(self):
        self.grad_inputs[0,0,0] += self.grad

    def update_internal(self):
        pass

def ConcatLayer(net, input_ensemble):
    
    assert len(input_ensemble) > 1
    total_channels = input_ensemble[0].shape[0]

    for i in range(1, len(input_ensemble) - 1):
        assert input_ensemble[i].ndim == 3 and \
               input_ensemble[i].ndim == input_ensemble[i-1].ndim and \
               input_ensemble[i].shape[1] == input_ensemble[i-1].shape[1] and\
               input_ensemble[i].shape[2] == input_ensemble.shape[2] 
        total_channels += input_ensemble[i].shape[0]
    #assert input_ensemble.ndim == 3, "PoolingLayer only supports 3-d input"



    shape = (total_channels,input_ensemble[0].shape[1], input_ensemble[0].shape[2])
    neurons = np.empty(shape, dtype='object')
    neurons[:] = ConvNeuron()

    concat_ens = net.init_ensemble(neurons)

    #input_shape = input_ensemble.shape


    for i in range(len(input_ensemble)):
        
        offset = 0   
        if(i > 0)
            offset = input_ensemble[i-1].shape[0]
       
        def mapping(c, y, x):
            out_c = c + offset + 1
            return range(c, out_c), range(y, y+1), range(x, x+1)
        net.add_connections(input_ensemble[i], concat_ens, mapping)

    return concat_ens
