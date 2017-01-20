import numpy as np
from ..neuron import WeightedNeuron, BiasNeuron
from ..ensemble import Ensemble, EnsembleGroup
import itertools
import latte.core

SIMDWIDTH = latte.config.SIMDWIDTH

class ConvNeuron(WeightedNeuron):
    def forward(self):
        for j in range_dim(self.inputs, 2):
            for k in range_dim(self.inputs, 1):
                for i in range_dim(self.inputs, 0):
                    self.value += self.inputs[i, j, k] * self.weights[i, j, k]

    def backward(self):
        for i in range_dim(self.inputs, 0):
            for j in range_dim(self.inputs, 1):
                for k in range_dim(self.inputs, 2):
                    self.grad_inputs[i, j, k] += self.grad * self.weights[i, j, k]

    def update_internal(self):
        for i in range_dim(self.inputs, 0):
            for j in range_dim(self.inputs, 1):
                for k in range_dim(self.inputs, 2):
                    self.grad_weights[i, j, k] += self.grad * self.inputs[i, j, k]


def compute_output_shape(input_shape, kernel, pad, stride):
    width, height, channels = input_shape
    return width_out, height_out

def ConvLayer(net, input_ensemble, num_filters=0, kernel=3, stride=1, pad=1, dilation=1):
    conv_ens = ConvLayerNoBias(net, input_ensemble, num_filters, kernel, stride, pad, dilation)



    input_channels, input_height, input_width = input_ensemble.shape

    kernel_eff = kernel + (kernel-1) * (dilation-1)

    output_width = ((input_width + 2 * pad - kernel_eff) // stride) + 1
    output_height = ((input_height + 2 * pad - kernel_eff) // stride) + 1

    #output_width = ((input_width - kernel * dilation + 2 * pad) // stride) + 1
    #output_height = ((input_height - kernel * dilation + 2 * pad) // stride) + 1

    if num_filters % latte.config.SIMDWIDTH != 0:
        num_filters_pad = latte.config.SIMDWIDTH - (num_filters % latte.config.SIMDWIDTH)
    else:
        num_filters_pad = 0
   
    bias = np.zeros((num_filters, 1), dtype=np.float32)
 
    bias = np.lib.pad(bias, ((0, num_filters_pad), (0, 0)), 'constant')
    grad_bias = np.zeros_like(bias)

    num_filters += num_filters_pad
    bias_neurons = np.empty((num_filters, output_height, output_width), dtype='object')
    for o in range(num_filters):
        bias_neurons[o, :, :] = BiasNeuron(bias[o], grad_bias[o])

    bias_ens = net.init_activation_ensemble(bias_neurons, conv_ens)

    # Begin Optimizations

    # bias_ens.privatize('grad_bias')
    bias_ens.tile('bias', dim=0, factor=SIMDWIDTH)
    bias_ens.tile('grad_bias', dim=0, factor=SIMDWIDTH)
    bias_ens.parallelize(phase="forward", loop_var="_neuron_index_0")
    bias_ens.parallelize(phase="forward", loop_var="_neuron_index_1_outer")
    bias_ens.swap_loops(phase="update_internal", loop_vars=("_neuron_index_0", "_neuron_index_1_outer"))
    bias_ens.parallelize(phase="update_internal", loop_var="_neuron_index_1_outer")
    # End Optimizations

    return EnsembleGroup(conv_ens, bias_ens)

def ConvLayerNoBias(net, input_ensemble, num_filters=0, kernel=3, stride=1, pad=1, dilation=1):
    assert num_filters > 0, "num_filters must be specified and greater than 0"
    assert input_ensemble.ndim == 3, "ConvLayer only supports 3-d input"

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

    if input_ensemble.shape[0] < latte.config.SIMDWIDTH:
        input_channel_pad = latte.config.SIMDWIDTH - input_ensemble.shape[0]
    elif input_ensemble.shape[0] % latte.config.SIMDWIDTH != 0:
        input_channel_pad = latte.config.SIMDWIDTH - (input_ensemble.shape[0] % latte.config.SIMDWIDTH)
    else:
        input_channel_pad = 0

    if num_filters % latte.config.SIMDWIDTH != 0:
        num_filters_pad = latte.config.SIMDWIDTH - (num_filters % latte.config.SIMDWIDTH)
    else:
        num_filters_pad = 0

    input_channels, input_height, input_width = input_ensemble.shape

    kernel_h_eff = kernel_h + (kernel_h-1) * (dilation - 1)
    kernel_w_eff = kernel_w + (kernel_w-1) * (dilation - 1)

    output_width = ((input_width + 2 * pad_w - kernel_w_eff) // stride_w) + 1
    output_height = ((input_height + 2 * pad_h - kernel_h_eff) // stride_h) + 1
 
    #output_width = ((input_width - kernel_w * dilation + 2 * pad_w) // stride_w) + 1
    #output_height = ((input_height - kernel_h * dilation + 2 * pad_h) // stride_h) + 1

    scale = np.sqrt(3.0 / (input_channels * kernel_h * kernel_w))
    weights = np.random.uniform(-scale, scale, (num_filters, input_channels, kernel_h,
            kernel_w)).astype(np.float32)

    weights = np.lib.pad(weights, ((0, num_filters_pad), (0, input_channel_pad), (0, 0), (0, 0)), 'constant', constant_values=(0,))
    input_channels += input_channel_pad
    num_filters += num_filters_pad
    grad_weights = np.zeros_like(weights)

    neurons = np.empty((num_filters, output_height, output_width), dtype='object')
    for o in range(num_filters):
        neurons[o, :, :] = ConvNeuron(weights[o], grad_weights[o])
    
    conv_ens = net.init_ensemble(neurons)

    def mapping(c, y, x):
        in_y = y*stride_h
        in_x = x*stride_w
        return (range(input_channels),
                range(in_y,in_y+kernel_h_eff,dilation),
                range(in_x,in_x+kernel_w_eff,dilation))

    input_ensemble.set_padding((0, input_channel_pad), (pad, pad), (pad, pad))

    net.add_connections(input_ensemble, conv_ens, mapping)
    
    # Begin Optimizations
    

    input_ensemble.tile('value', dim=0, factor=SIMDWIDTH)
    input_ensemble.tile('grad', dim=0, factor=SIMDWIDTH)
    conv_ens.tile('inputs', dim=0, factor=SIMDWIDTH)
    conv_ens.tile('grad_inputs', dim=0, factor=SIMDWIDTH)

    conv_ens.tile('weights', dim=0, factor=SIMDWIDTH)
    conv_ens.tile('weights', dim=1, factor=SIMDWIDTH)
    conv_ens.transpose('weights', -2, -1)

    conv_ens.tile('grad_weights', dim=0, factor=SIMDWIDTH)
    conv_ens.tile('grad_weights', dim=1, factor=SIMDWIDTH)
    # conv_ens.privatize('grad_weights')
    # conv_ens.transpose('grad_weights', -2, -1)

    conv_ens.tile('value', dim=0, factor=SIMDWIDTH)
    conv_ens.tile('grad', dim=0, factor=SIMDWIDTH)

    if "OPENCL" not in latte.config.parallel_strategy:
        conv_ens.vectorize(phase="forward", loop_var="_neuron_index_1_inner", factor=SIMDWIDTH)
    conv_ens.parallelize(phase="forward", loop_var="_neuron_index_0")
    conv_ens.parallelize(phase="forward", loop_var="_neuron_index_1_outer")
    if "OPENCL" not in latte.config.parallel_strategy:
        conv_ens.vectorize(phase="backward", loop_var="i_inner", factor=SIMDWIDTH)
    conv_ens.parallelize(phase="backward", loop_var="_neuron_index_0")
    conv_ens.parallelize(phase="backward", loop_var="i_outer")
    conv_ens.swap_loops(phase="backward", loop_vars=("_neuron_index_1_inner", "j"))
    conv_ens.swap_loops(phase="backward", loop_vars=("_neuron_index_1_inner", "k"))
    if "OPENCL" not in latte.config.parallel_strategy:
        conv_ens.vectorize(phase="update_internal", loop_var="i_inner", factor=SIMDWIDTH)
    conv_ens.parallelize(phase="update_internal", loop_var="_neuron_index_1_outer")
    conv_ens.parallelize(phase="update_internal", loop_var="i_outer")
    conv_ens.swap_loops(phase="update_internal", loop_vars=("_neuron_index_0", "_neuron_index_1_outer"))
    conv_ens.swap_loops(phase="update_internal", loop_vars=("_neuron_index_1_inner", "j"))
    conv_ens.swap_loops(phase="update_internal", loop_vars=("_neuron_index_1_inner", "k"))

    
    if "OPENCL" not in latte.config.parallel_strategy:
        outer_unroll_factor = 16
        #factor = 8
        while output_width % outer_unroll_factor != 0:
           outer_unroll_factor -= 1
        conv_ens.unroll(phase="forward", loop_var="_neuron_index_3", factor=outer_unroll_factor)
        inner_unroll_factor = 4
        conv_ens.unroll_2(phase="forward", loop_var="i_inner", factor=inner_unroll_factor)
        #syntax [enclose_loop, dimension, forall, prefetch_loop_var, prefetch_multiplier, prefetch_constant, dest_loop, cacheline_hint]
        if outer_unroll_factor == output_width:
          #3350
          conv_ens.prefetch(phase="forward", prefetch_dict_list={'value': [1, "_neuron_index_3", -3, outer_unroll_factor, "_neuron_index_2", 1, 1, 0], 'inputs': [2, "i_inner", -3, 1, 64, "i_inner", "_neuron_index_2", "_neuron_index_2", stride_h, 1, 0]  })
          #3563GF
          #conv_ens.prefetch(phase="forward", prefetch_dict_list={'value': [1, "_neuron_index_3", -3, outer_unroll_factor, "_neuron_index_2", 1, 1, 0]})
        else:
          conv_ens.prefetch(phase="forward", prefetch_dict_list={'value': [1, "_neuron_index_3", -1, outer_unroll_factor, "_neuron_index_3", 1, outer_unroll_factor, 0], 'inputs': [2, "i_inner", -2, inner_unroll_factor, 64, "i_inner", "_neuron_index_2", "_neuron_index_3", 1, inner_unroll_factor, 0]  })
        #backward  
        conv_ens.unroll(phase="backward", loop_var="_neuron_index_3", factor=outer_unroll_factor)
        conv_ens.unroll(phase="update_internal", loop_var="_neuron_index_3", factor=outer_unroll_factor)
         
    # End Optimizations
    # Added by Raj/Anand
    if "LIBXSMM" in latte.config.codegen_strategy:
      conv_ens.use_libxsmm(1)
      conv_ens.stride = stride

    return conv_ens
