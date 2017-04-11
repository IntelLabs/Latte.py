import numpy as np
from ..neuron import Neuron
from ..ensemble import Ensemble
import itertools
import latte
import math
class MaxNeuron(Neuron):
    batch_fields     = Neuron.batch_fields + ["mask_j", "mask_k"]
    zero_init_fields = Neuron.zero_init_fields + ["mask_j", "mask_k"]

    def __init__(self, j,k):
        super().__init__()
        self.inputs = []
        self.grad_inputs = []

        self.mask_j = j
        self.mask_k = k

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
    #ANAND : Changing output volume calculation to use ceil instead of floor, matches caffe
    #intepretation 
    output_width = int(math.ceil((input_width - kernel_w + 2 * pad_w) / stride_w)) + 1
    output_height = int(math.ceil((input_height - kernel_h + 2 * pad_h) / stride_h)) + 1


    max_j = np.zeros((5,), dtype=np.int32)
    
    shape = (input_channels, output_height, output_width)
    max_j = np.zeros(shape, dtype=np.int32)
    max_k = np.zeros(shape, dtype=np.int32)

    neurons = np.empty(shape, dtype='object')


    for i in range(input_channels):
      for j in range(output_height):
         for k in range(output_width): 
            neurons[i,j,k] = MaxNeuron(max_j[i,j,k],max_k[i,j,k])



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
        #print("tiling on\n")
        for dim, factor in tiled_dims:
            pooling_ens.tile('inputs', dim=dim, factor=factor)
        pooling_ens.parallelize(phase="forward", loop_var="_neuron_index_1_outer")
        pooling_ens.parallelize(phase="backward", loop_var="_neuron_index_1_outer")
        pooling_ens.simd(phase="forward", loop_var="_neuron_index_1_inner")
        pooling_ens.tile('value', dim=0, factor=latte.config.SIMDWIDTH)
        pooling_ens.tile('mask_j', dim=0, factor=latte.config.SIMDWIDTH)
        pooling_ens.tile('mask_k', dim=0, factor=latte.config.SIMDWIDTH)
        '''
        unroll_factor = 4
        if pooling_ens.shape[1] % unroll_factor == 0 and pooling_ens.shape[2] % unroll_factor == 0:
          pooling_ens.unroll(phase="forward", loop_var="_neuron_index_2", factor=unroll_factor, unroll_type= 1)
          pooling_ens.unroll(phase="forward", loop_var="_neuron_index_3", factor=unroll_factor, unroll_type=1)
        '''

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
    #ANAND : Changing output volume calculation to use ceil instead of floor, matches caffe
    #intepretation
    output_width = int(math.ceil((input_width - kernel_w + 2 * pad_w) / stride_w)) + 1
    output_height = int(math.ceil((input_height - kernel_h + 2 * pad_h) / stride_h)) + 1
 
    shape = (input_channels, output_height, output_width)
    neurons = np.empty(shape, dtype='object')
    
    h = np.full(shape, kernel_h, dtype=np.int32)
    w = np.full(shape, kernel_w, dtype=np.int32) 
 

    
    for i in range(input_channels):
      for j in range(output_height):
         for k in range(output_width):
            neurons[i,j,k] = MeanNeuron(h[i,j,k],w[i,j,k])

 
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
        #print("tiling on\n")
        for dim, factor in tiled_dims:
            pooling_ens.tile('inputs', dim=dim, factor=factor)
        pooling_ens.parallelize(phase="forward", loop_var="_neuron_index_1_outer")
        pooling_ens.parallelize(phase="backward", loop_var="_neuron_index_1_outer")
        pooling_ens.tile('value', dim=0, factor=latte.config.SIMDWIDTH)
        pooling_ens.tile('kernel', dim=0, factor=latte.config.SIMDWIDTH)
    else:
        pooling_ens.parallelize(phase="forward", loop_var="_neuron_index_1")
        pooling_ens.parallelize(phase="backward", loop_var="_neuron_index_1")
 
    if "grad" in input_ensemble.tiling_info:
        tiled_dims = input_ensemble.tiling_info["grad"]
        for dim, factor in tiled_dims:
            pooling_ens.tile('grad_inputs', dim=dim, factor=factor)
        pooling_ens.tile('grad', dim=0, factor=latte.config.SIMDWIDTH)
        pooling_ens.tile('kernel', dim=0, factor=latte.config.SIMDWIDTH)

    return pooling_ens

