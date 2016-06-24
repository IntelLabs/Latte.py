import numpy as np
from ..neuron import WeightedNeuron, BiasNeuron
from ..ensemble import Ensemble, EnsembleGroup
import itertools
import latte.core

SIMDWIDTH = latte.core.SIMDWIDTH

class ConvNeuron(WeightedNeuron):
    def forward(self):
        for j in range_dim(self.inputs, 1):
            for k in range_dim(self.inputs, 2):
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

    if input_ensemble.shape[0] < latte.core.SIMDWIDTH:
        input_channel_pad = latte.core.SIMDWIDTH - input_ensemble.shape[0]
    elif input_ensemble.shape[0] % latte.core.SIMDWIDTH != 0:
        input_channel_pad = latte.core.SIMDWIDTH - (input_ensemble.shape[0] % latte.core.SIMDWIDTH)
    else:
        input_channel_pad = 0


    # remainder = num_filters % SIMDWIDTH
    # num_filters += remainder

    input_channels, input_height, input_width = input_ensemble.shape
    output_width = ((input_width - kernel_w * dilation + 2 * pad_w) // stride_w) + 1
    output_height = ((input_height - kernel_h * dilation + 2 * pad_h) // stride_h) + 1

    scale = np.sqrt(3.0 / (input_channels * kernel_h * kernel_w))
    weights = np.random.rand(num_filters, input_channels, kernel_h,
            kernel_w).astype(np.float32) * (2 * scale) - scale
    if input_channel_pad > 0:
        weights = np.lib.pad(weights, ((0, 0), (0, input_channel_pad), (0, 0), (0, 0)), 'constant')
        input_channels += input_channel_pad
    grad_weights = np.zeros_like(weights)

    neurons = np.empty((num_filters, output_height, output_width), dtype='object')
    for o in range(num_filters):
        neurons[o, :, :] = ConvNeuron(weights[o], grad_weights[o])

    conv_ens = net.init_ensemble(neurons)

    def mapping(c, y, x):
        in_y = y*stride_h
        in_x = x*stride_w
        return (range(input_channels),
                range(in_y,in_y+kernel_h*dilation,dilation),
                range(in_x,in_x+kernel_w*dilation,dilation))

    input_ensemble.set_padding((0, input_channel_pad), (pad, pad), (pad, pad))
    net.add_connections(input_ensemble, conv_ens, mapping)

    bias = np.zeros((num_filters, 1), dtype=np.float32)
    grad_bias = np.zeros_like(bias)

    bias_neurons = np.empty((num_filters, output_height, output_width), dtype='object')
    for o in range(num_filters):
        bias_neurons[o, :, :] = BiasNeuron(bias[o], grad_bias[o])

    bias_ens = net.init_activation_ensemble(bias_neurons, conv_ens)

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
    conv_ens.privatize('grad_weights')
    conv_ens.transpose('grad_weights', -2, -1)

    conv_ens.tile('value', dim=0, factor=SIMDWIDTH)
    conv_ens.tile('grad', dim=0, factor=SIMDWIDTH)

    conv_ens.vectorize(direction="forward", loop_var="_neuron_index_1_inner", factor=SIMDWIDTH)
    bias_ens.vectorize(direction="forward", loop_var="_neuron_index_1_inner", factor=SIMDWIDTH)
    conv_ens.parallelize(direction="forward", loop_var="_neuron_index_1")
    conv_ens.vectorize(direction="backward", loop_var="i_inner", factor=SIMDWIDTH)
    bias_ens.vectorize(direction="backward", loop_var="_neuron_index_1_inner", factor=SIMDWIDTH)
    conv_ens.parallelize(direction="backward", loop_var="i")
    conv_ens.swap_loops(direction="backward", loop_vars=("_neuron_index_1_inner", "j"))
    conv_ens.swap_loops(direction="backward", loop_vars=("_neuron_index_1_inner", "k"))

    factor = 8
    while output_width % factor != 0:
        factor -= 1
    conv_ens.unroll(direction="forward", loop_var="_neuron_index_3", factor=factor)
    bias_ens.unroll(direction="forward", loop_var="_neuron_index_3", factor=factor)
    bias_ens.privatize('grad_bias')

    factor = 4
    while output_width % factor != 0:
        factor -= 1
    conv_ens.unroll(direction="backward", loop_var="_neuron_index_3", factor=factor)
    bias_ens.unroll(direction="backward", loop_var="_neuron_index_3", factor=factor)

    bias_ens.tile('bias', dim=0, factor=SIMDWIDTH)
    bias_ens.tile('grad_bias', dim=0, factor=SIMDWIDTH)
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

    if input_ensemble.shape[0] < latte.core.SIMDWIDTH:
        input_channel_pad = latte.core.SIMDWIDTH - input_ensemble.shape[0]
    elif input_ensemble.shape[0] % latte.core.SIMDWIDTH != 0:
        input_channel_pad = latte.core.SIMDWIDTH - (input_ensemble.shape[0] % latte.core.SIMDWIDTH)
    else:
        input_channel_pad = 0


    # remainder = num_filters % SIMDWIDTH
    # num_filters += remainder

    input_channels, input_height, input_width = input_ensemble.shape
    output_width = ((input_width - kernel_w * dilation + 2 * pad_w) // stride_w) + 1
    output_height = ((input_height - kernel_h * dilation + 2 * pad_h) // stride_h) + 1

    scale = np.sqrt(3.0 / (input_channels * kernel_h * kernel_w))
    weights = np.random.rand(num_filters, input_channels, kernel_h,
            kernel_w).astype(np.float32) * (2 * scale) - scale
    if input_channel_pad > 0:
        weights = np.lib.pad(weights, ((0, 0), (0, input_channel_pad), (0, 0), (0, 0)), 'constant')
        input_channels += input_channel_pad
    grad_weights = np.zeros_like(weights)

    neurons = np.empty((num_filters, output_height, output_width), dtype='object')
    for o in range(num_filters):
        neurons[o, :, :] = ConvNeuron(weights[o], grad_weights[o])

    conv_ens = net.init_ensemble(neurons)

    def mapping(c, y, x):
        in_y = y*stride_h
        in_x = x*stride_w
        return (range(input_channels),
                range(in_y,in_y+kernel_h*dilation,dilation),
                range(in_x,in_x+kernel_w*dilation,dilation))

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
    conv_ens.privatize('grad_weights')
    conv_ens.transpose('grad_weights', -2, -1)

    conv_ens.tile('value', dim=0, factor=SIMDWIDTH)
    conv_ens.tile('grad', dim=0, factor=SIMDWIDTH)

    conv_ens.vectorize(direction="forward", loop_var="_neuron_index_1_inner", factor=SIMDWIDTH)
    conv_ens.parallelize(direction="forward", loop_var="_neuron_index_1")
    conv_ens.vectorize(direction="backward", loop_var="i_inner", factor=SIMDWIDTH)
    conv_ens.parallelize(direction="backward", loop_var="i")
    conv_ens.swap_loops(direction="backward", loop_vars=("_neuron_index_1_inner", "j"))
    conv_ens.swap_loops(direction="backward", loop_vars=("_neuron_index_1_inner", "k"))

    factor = 8
    while output_width % factor != 0:
        factor -= 1
    conv_ens.unroll(direction="forward", loop_var="_neuron_index_3", factor=factor)

    factor = 4
    while output_width % factor != 0:
        factor -= 1
    conv_ens.unroll(direction="backward", loop_var="_neuron_index_3", factor=factor)

    # End Optimizations

    return conv_ens
