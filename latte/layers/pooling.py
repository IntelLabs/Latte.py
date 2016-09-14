import numpy as np
from ..neuron import Neuron
from ..ensemble import Ensemble
import itertools
import latte

class MaxNeuron(Neuron):
    batch_fields     = Neuron.batch_fields + ["mask_j", "mask_k"]
    zero_init_fields = Neuron.zero_init_fields + ["mask_j", "mask_k"]

    def __init__(self):
        super().__init__()
        self.inputs = []
        self.grad_inputs = []

        self.mask_j = 0
        self.mask_k = 0

    def forward(self):
        max_value = -INFINITY
        for j in range_dim(self.inputs, 1):
            for k in range_dim(self.inputs, 2):
                if self.inputs[0,j,k] > max_value:
                    max_value = self.inputs[0,j,k]
                    self.mask_j = j
                    self.mask_k = k
        self.value = max_value

    def backward(self):
        j = self.mask_j
        k = self.mask_k
        self.grad_inputs[0,j,k] += self.grad

    def update_internal(self):
        pass

def MaxPoolingLayer(net, input_ensemble, kernel=2, stride=2, pad=0):
    assert input_ensemble.ndim == 3, "PoolingLayer only supports 3-d input"

    if isinstance(kernel, tuple):
        assert len(kernel) == 2, "kernel as a tuple must be of length 2"
        kernel_h, kernel_w = kernel
    else:
        kernel_h, kernel_w = kernel, kernel

    if isinstance(stride, tuple):
        assert len(stride) == 2, "stride as a tuple must be of length 2"
        stride_h, stride_w = stride
    else:
        stride_h, stride_w = stride, stride

    if isinstance(pad, tuple):
        assert len(pad) == 2, "pad as a tuple must be of length 2"
        pad_h, pad_w = pad
    else:
        pad_h, pad_w = pad, pad

    input_channels, input_height, input_width = input_ensemble.shape
    output_width = ((input_width - kernel_w + 2 * pad_w) // stride_w) + 1
    output_height = ((input_height - kernel_h + 2 * pad_h) // stride_h) + 1

    shape = (input_channels, output_height, output_width)
    neurons = np.empty(shape, dtype='object')
    neurons[:] = MaxNeuron()

    pooling_ens = net.init_ensemble(neurons)

    input_shape = input_ensemble.shape

    def mapping(c, y, x):
        in_y = y*stride_h - pad 
        in_x = x*stride_w - pad
        return range(c, c+1), range(in_y, in_y+kernel_h), range(in_x, in_x+kernel_w)

    net.add_connections(input_ensemble, pooling_ens, mapping, clamp=True)

    pooling_ens.parallelize(phase="forward", loop_var="_neuron_index_0")
    pooling_ens.parallelize(phase="backward", loop_var="_neuron_index_0")

    if "value" in input_ensemble.tiling_info:
        tiled_dims = input_ensemble.tiling_info["value"]
        for dim, factor in tiled_dims:
            pooling_ens.tile('inputs', dim=dim, factor=factor)
        pooling_ens.parallelize(phase="forward", loop_var="_neuron_index_1_outer")
        pooling_ens.parallelize(phase="backward", loop_var="_neuron_index_1_outer")
        pooling_ens.tile('value', dim=0, factor=latte.config.SIMDWIDTH)
        pooling_ens.tile('mask_j', dim=0, factor=latte.config.SIMDWIDTH)
        pooling_ens.tile('mask_k', dim=0, factor=latte.config.SIMDWIDTH)
    else:
        pooling_ens.parallelize(phase="forward", loop_var="_neuron_index_1")
        pooling_ens.parallelize(phase="backward", loop_var="_neuron_index_1")

    if "grad" in input_ensemble.tiling_info:
        tiled_dims = input_ensemble.tiling_info["grad"]
        for dim, factor in tiled_dims:
            pooling_ens.tile('grad_inputs', dim=dim, factor=factor)
        pooling_ens.tile('grad', dim=0, factor=latte.config.SIMDWIDTH)

    return pooling_ens

class MeanNeuron(Neuron):
    batch_fields     = Neuron.batch_fields + ["sum_value", "kernel"]
    #zero_init_fields = Neuron.zero_init_fields + ["sum_value"]
 
    def __init__(self, height, width):
        super().__init__()
        self.inputs = []
        self.grad_inputs = []
        #self.sum_value = 0 
        #self.kernel_w = width
        #self.kernel_h = height        
        #self.mask_j = 0`
        #self.mask_k = 0
        self.value = 0  
        self.kernel = width*height    

    def forward(self):
        #sum_value = 0
        for j in range_dim(self.inputs, 1):
            for k in range_dim(self.inputs, 2):
                self.value +=  self.inputs[0,j,k]
 
        self.value = self.value/self.kernel#self.sum_value
    

    def backward(self):
        #j = self.mask_j
        #k = self.mask_k
        val = self.grad/self.kernel

        for j in range_dim(self.grad_inputs, 1):
            for k in range_dim(self.grad_inputs, 2):
                self.grad_inputs[0,j,k] += val
 
    def update_internal(self):
        pass
 
 



def MeanPoolingLayer(net, input_ensemble, kernel=2, stride=2, pad=0):
    assert input_ensemble.ndim == 3, "PoolingLayer only supports 3-d input"
 
    if isinstance(kernel, tuple):
        assert len(kernel) == 2, "kernel as a tuple must be of length 2"
        kernel_h, kernel_w = kernel
    else:
        kernel_h, kernel_w = kernel, kernel
 
    if isinstance(stride, tuple):
        assert len(stride) == 2, "stride as a tuple must be of length 2"
        stride_h, stride_w = stride
    else:
        stride_h, stride_w = stride, stride
 
    if isinstance(pad, tuple):
        assert len(pad) == 2, "pad as a tuple must be of length 2"
        pad_h, pad_w = pad
    else:
        pad_h, pad_w = pad, pad
 
    input_channels, input_height, input_width = input_ensemble.shape
    output_width = ((input_width - kernel_w + 2 * pad_w) // stride_w) + 1
    output_height = ((input_height - kernel_h + 2 * pad_h) // stride_h) + 1
 
    shape = (input_channels, output_height, output_width)
    neurons = np.empty(shape, dtype='object')
    neurons[:] = MeanNeuron(kernel_h, kernel_w)
 
    pooling_ens = net.init_ensemble(neurons)
 
    input_shape = input_ensemble.shape
 
    def mapping(c, y, x):
        in_y = y*stride_h - pad
        in_x = x*stride_w - pad
        return range(c, c+1), range(in_y, in_y+kernel_h), range(in_x, in_x+kernel_w)
 
    net.add_connections(input_ensemble, pooling_ens, mapping)
 
    pooling_ens.parallelize(phase="forward", loop_var="_neuron_index_0")
    pooling_ens.parallelize(phase="backward", loop_var="_neuron_index_0")
 
    if "value" in input_ensemble.tiling_info:
        tiled_dims = input_ensemble.tiling_info["value"]
        for dim, factor in tiled_dims:
            pooling_ens.tile('inputs', dim=dim, factor=factor)
        pooling_ens.parallelize(phase="forward", loop_var="_neuron_index_1_outer")
        pooling_ens.parallelize(phase="backward", loop_var="_neuron_index_1_outer")
        pooling_ens.tile('value', dim=0, factor=latte.config.SIMDWIDTH)
        pooling_ens.tile('mask_j', dim=0, factor=latte.config.SIMDWIDTH)
        pooling_ens.tile('mask_k', dim=0, factor=latte.config.SIMDWIDTH)
    else:
        pooling_ens.parallelize(phase="forward", loop_var="_neuron_index_1")
        pooling_ens.parallelize(phase="backward", loop_var="_neuron_index_1")
 
    if "grad" in input_ensemble.tiling_info:
        tiled_dims = input_ensemble.tiling_info["grad"]
        for dim, factor in tiled_dims:
            pooling_ens.tile('grad_inputs', dim=dim, factor=factor)
        pooling_ens.tile('grad', dim=0, factor=latte.config.SIMDWIDTH)
 
    return pooling_ens

