import numpy as np
from ..neuron import Neuron
from ..ensemble import LRNEnsemble
import itertools
import latte


class LRNNeuron(Neuron):
    batch_fields     = Neuron.batch_fields + ["sum_value","n", "alpha", "beta"]

    def __init__(self, k,n, alpha,beta):
        super().__init__()
        self.inputs = []
        self.grad_inputs = []
        self.sum_value = 0.0
        self.k = k
        self.n = n
        self.alpha = alpha
        self.beta = beta

    def forward(self):
        index = self.n/2    
        #update sum on 0th channel
        for i in range_dim(self.inputs, 0):
            self.sum_value  = self.sum_value + (self.inputs[i,0,0]*self.inputs[i,0,0])

        self.sum_value = self.sum_value*(self.alpha/self.n)
        self.sum_value += self.k    
        index = self.n/2
        self.value = self.inputs[index,0,0]/pow(self.sum_value,self.beta)

    def backward(self):
        index = self.n/2
        self.grad_inputs[index,0,0]  += (self.grad/pow(self.sum_value,self.beta))
        for i in range_dim(self.inputs, 0):
            self.grad_inputs[i,0,0] -= (((((2.0/self.n)*self.alpha)*(self.beta*self.grad))*self.inputs[index,0,0])*self.inputs[i,0,0]/pow(self.sum_value,self.beta+1))
    
    def update_internal(self):
        pass



def LRNLayer(net, input_ensemble, n = 5, beta = 0.75 , alpha =0.0001, k = 1.0 ):
    assert input_ensemble.ndim == 3, "LRNLayer only supports 3-d input"
    input_channels, input_height, input_width = input_ensemble.shape


    #k_  = np.zeros((1), dtype=np.float32)
    #n_  = np.zeros((1), dtype=np.float32)
    #alpha_  = np.zeros((1), dtype=np.float32)
    #beta_  = np.zeros((1), dtype=np.float32)

    #k_[0] = float(k)
    #n_[0] = float(n)
    #alpha_[0] = float(alpha)
    #beta_[0] = float(beta)


    #sum_value = np.zeros(input_ensemble.shape, dtype=np.float32)



    shape = (input_channels, input_height, input_width)
    neurons = np.empty(shape, dtype='object')
    neurons[:,:,:] = LRNNeuron(float(k),int(n),float(alpha),float(beta))

 
    lrn_ens = LRNEnsemble(neurons)
    net.add_ensemble(lrn_ens)
    input_shape = input_ensemble.shape

    extend = latte.config.SIMDWIDTH
    pad = (n-1)//2       
    if (input_channels + pad)% extend == 0:    
        efective_pad = pad
    else:
        effective_pad  = extend - (input_channels + pad)%extend
        effective_pad += pad
    
    #ANAND REFACTOR
    #mapping expression too complicated
    def mapping(c, y, x):
        in_y = y
        in_x = x
        return range(c -pad+effective_pad, c+ pad +effective_pad +1), range(in_y, in_y+1), range(in_x, in_x + 1)
    
    input_ensemble.set_padding((effective_pad,effective_pad),(0,0), (0,0))
    net.add_connections(input_ensemble, lrn_ens, mapping)

    lrn_ens.parallelize(phase="forward", loop_var="_neuron_index_0")
    lrn_ens.parallelize(phase="backward", loop_var="_neuron_index_0")



    if "value" in input_ensemble.tiling_info:
        tiled_dims = input_ensemble.tiling_info["value"]
        for dim, factor in tiled_dims:
            lrn_ens.tile('inputs', dim=dim, factor=factor)
        lrn_ens.tile('value', dim=0, factor=latte.config.SIMDWIDTH)
        lrn_ens.tile('sum_value', dim=0, factor=latte.config.SIMDWIDTH)
        lrn_ens.tile('alpha', dim=0, factor=latte.config.SIMDWIDTH)
        lrn_ens.tile('beta', dim=0, factor=latte.config.SIMDWIDTH)
        lrn_ens.tile('n', dim=0, factor=latte.config.SIMDWIDTH)
        lrn_ens.tile('k', dim=0, factor=latte.config.SIMDWIDTH)
        #if "OPENCL" not in latte.config.parallel_strategy: 
        #    lrn_ens.vectorize(phase="forward", loop_var="_neuron_index_1_inner", factor=latte.config.SIMDWIDTH)

    if "grad" in input_ensemble.tiling_info:
        tiled_dims = input_ensemble.tiling_info["grad"]
        for dim, factor in tiled_dims:
            lrn_ens.tile('grad_inputs', dim=dim, factor=factor)
        lrn_ens.tile('grad', dim=0, factor=latte.config.SIMDWIDTH)
        #if "OPENCL" not in latte.config.parallel_strategy:
        #    lrn_ens.vectorize(phase="backward", loop_var="_neuron_index_1_inner", factor=latte.config.SIMDWIDTH)
    return lrn_ens
