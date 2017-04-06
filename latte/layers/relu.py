import numpy as np
from ..neuron import Neuron
from ..ensemble import Ensemble
import latte
import itertools

class ReLUNeuron(Neuron):
    def __init__(self):
        super().__init__()
        self.inputs = []
        self.grad_inputs = []

    def forward(self):
        self.value = max(self.input, float(0.0))

    def backward(self):
        # self.grad_input = ifelse(self.input > 0.0, self.grad, 0.0)
        if self.input > 0.0:
            self.grad_input = self.grad
        else:
            self.grad_input = 0.0
        #self.grad_input = max(self.grad, float(0.0))
    def update_internal(self):
        pass

def ReLULayer(net, input_ensemble):
    
    relu_neurons = np.empty(input_ensemble.shape, dtype='object')

    for i in range(len(relu_neurons)):
        relu_neurons[i] = ReLUNeuron()

    relu_ens = net.init_activation_ensemble(relu_neurons, input_ensemble)
    relu_ens.parallelize(phase="forward", loop_var="_neuron_index_0")
    relu_ens.parallelize(phase="backward", loop_var="_neuron_index_0")
    if "value" in input_ensemble.tiling_info:
        relu_ens.parallelize(phase="forward", loop_var="_neuron_index_1_outer")
        relu_ens.parallelize(phase="backward", loop_var="_neuron_index_1_outer")
        relu_ens.simd(phase="forward", loop_var="_neuron_index_1_inner")
    else:
        relu_ens.parallelize(phase="forward", loop_var="_neuron_index_1")
        relu_ens.parallelize(phase="backward", loop_var="_neuron_index_1")
    '''
    h_unroll_factor=7
    w_unroll_factor=4
    if relu_ens.shape[1] % h_unroll_factor == 0 and relu_ens.shape[2] % w_unroll_factor == 0:
      relu_ens.unroll(phase="forward", loop_var="_neuron_index_2", factor=h_unroll_factor, unroll_type= 1)
      relu_ens.unroll(phase="forward", loop_var="_neuron_index_3", factor=w_unroll_factor, unroll_type=1)
    '''
    if net.cbr_fusion:
      net.fuse_cbr(input_ensemble, relu_ens)

    return relu_ens
