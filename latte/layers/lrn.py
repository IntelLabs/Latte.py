import numpy as np
from ..neuron import Neuron
from ..ensemble import Ensemble
import itertools
import latte


class LRNNeuron(Neuron):
    batch_fields     = Neuron.batch_fields + ["sum_value", "n", "alpha", "beta"]
    #zero_init_fields = Neuron.zero_init_fields + ["mask_j", "mask_k"]

    def __init__(self, k,n, alpha,beta):
        super().__init__()
        self.inputs = []
        self.grad_inputs = []
        self.sum_value = k
        #self.k = k
        self.n = n
        self.alpha = alpha
        self.beta = beta
        #self.mask_j = 0`
        #self.mask_k = 0

    def forward(self):
        #sum_value = 1.0
        #initialize sum to parameter k
        #for i in range_dim(self.inputs,0):
        #for j in range_dim(self.inputs, 1):
        #for k in range_dim(self.inputs, 2):
        #   sum_value[i,j,k] = 1.0

        #update sum on 0th channel
        for i in range_dim(self.inputs, 0):
            #for j in range_dim(self.inputs, 1):
            #for k in range_dim(self.inputs, 2):
            #computing square might be more optimal to use pow(self.inputs, 2)
            self.sum_value  += self.inputs[i,0,0]*self.inputs[i,0,0]

        self.sum_value *= self.alpha/n
            #sum_value =  sum_value^(-beta)
        self.value = self.inputs[0,0,0]*self.sum_value^(-self.beta)


            #update sum on channels 1 to n
            #for i in [1 , range_dim(self.inputs, 0)):
            #for j in range_dim(self.inputs, 1):
            #for k in range_dim(self.inputs, 2):
            #computing square might be more optimal to use pow(self.inputs, 2)
            #sum_value[i,j,k] += sum_value[i-1,j,k]
            #sum_value[i,j,k] -= self.inputs[i-1,j,k]*self.inputs[i-1,j,k]
            #sum_value[i,j,k] -= self.inputs[i+n-1,j,k]*self.inputs[i+n-1,j,k]


    def backward(self):
        value = 0.0

        #for i in range_dim(self.inputs, 0):
        self.grad_inputs[0,0,0] += self.grad/self.sum_value^(beta)

        for i in range_dim(self.inputs, 0):
            self.grad_inputs[i,0,0] -= (2/self.n)*self.alpha*self.beta*self.grad*self.value/self.sum_value
        #j = self.mask_j
        #k = self.mask_k
        #val = self.grad/self.inputs.size
        #for j in range_dim(self.grad_inputs, 1):
        #for k in range_dim(self.grad_inputs, 2):
        #self.grad_inputs[0,j,k] += val

    def update_internal(self):
        pass



def LRNLayer(net, input_ensemble, n = 5, beta = 0.75 , alpha =0.0001, k = 1.0 ):
    assert input_ensemble.ndim == 3, "PoolingLayer only supports 3-d input"
    input_channels, input_height, input_width = input_ensemble.shape


    shape = (input_channels, input_height, input_width)
    neurons = np.empty(shape, dtype='object')
    neurons[:] = LRNNeuron(k,n,alpha,beta)

    pooling_ens = net.init_ensemble(neurons)

    input_shape = input_ensemble.shape

    def mapping(c, y, x):
        in_y = y
        in_x = x
        return range(c, c+n), range(in_y, in_y+1), range(in_x, in_x + 1)

    net.add_connections(input_ensemble, pooling_ens, mapping)

    #pooling_ens.parallelize(phase="forward", loop_var="_neuron_index_0")
    #pooling_ens.parallelize(phase="backward", loop_var="_neuron_index_0")

    #if "value" in input_ensemble.tiling_info:
    #tiled_dims = input_ensemble.tiling_info["value"]
    #for dim, factor in tiled_dims:
    #pooling_ens.tile('inputs', dim=dim, factor=factor)
    #pooling_ens.parallelize(phase="forward", loop_var="_neuron_index_1_outer")
    #pooling_ens.parallelize(phase="backward", loop_var="_neuron_index_1_outer")
    #pooling_ens.tile('value', dim=0, factor=latte.config.SIMDWIDTH)
    #pooling_ens.tile('mask_j', dim=0, factor=latte.config.SIMDWIDTH)
    #pooling_ens.tile('mask_k', dim=0, factor=latte.config.SIMDWIDTH)
    #else:
    #pooling_ens.parallelize(phase="forward", loop_var="_neuron_index_1")
    #pooling_ens.parallelize(phase="backward", loop_var="_neuron_index_1")
    #if "grad" in input_ensemble.tiling_info:
    #tiled_dims = input_ensemble.tiling_info["grad"]
    #for dim, factor in tiled_dims:
    #pooling_ens.tile('grad_inputs', dim=dim, factor=factor)
    #pooling_ens.tile('grad', dim=0, factor=latte.config.SIMDWIDTH)

    return pooling_ens
