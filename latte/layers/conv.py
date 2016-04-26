import numpy as np
from ..neuron import Neuron
from ..ensemble import Ensemble
import itertools

class ConvNeuron(Neuron):
    def __init__(self, weights, grad_weights, bias, grad_bias):
        super().__init__()
        self.inputs = []
        self.grad_inputs = []

        self.weights = weights
        self.grad_weights = grad_weights

        self.bias = bias
        self.grad_bias = grad_bias

    def forward(self):
        for i, p in enumerate_dim(self.inputs, 0):
            for j, q in enumerate_dim(self.inputs, 1):
                for k, r in enumerate_dim(self.inputs, 2):
                    self.value += self.inputs[p, q, r] * self.weights[i, j, k]
        self.value += self.bias[0]

    def backward(self):
        for i, p in enumerate_dim(self.inputs, 0):
            for j, q in enumerate_dim(self.inputs, 1):
                for k, r in enumerate_dim(self.inputs, 2):
                    self.grad_inputs[p, q, r] += self.grad * self.weights[i, j, k]
        for i, p in enumerate_dim(self.inputs, 0):
            for j, q in enumerate_dim(self.inputs, 1):
                for k, r in enumerate_dim(self.inputs, 2):
                    self.grad_weights[i, j, k] += self.grad * self.inputs[p, q, r]
        self.grad_bias[0] += self.grad


def compute_output_shape(input_shape, kernel, pad, stride):
    width, height, channels = input_shape
    return width_out, height_out

def ConvLayer(net, input_ensemble, num_filters=0, kernel=3, stride=1, pad=1):
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

    input_channels, input_height, input_width = input_ensemble.shape
    output_width = ((input_width - kernel_w + 2 * pad_w) // stride_w) + 1
    output_height = ((input_height - kernel_h + 2 * pad_h) // stride_h) + 1

    scale = np.sqrt(3.0 / (input_channels * kernel_h * kernel_w))
    weights = np.random.rand(num_filters, input_channels, kernel_h,
            kernel_w).astype(np.float32) * (2 * scale) - scale
    grad_weights = np.zeros_like(weights)

    bias = np.zeros((num_filters, 1), dtype=np.float32)
    grad_bias = np.zeros_like(bias)
    neurons = np.empty((num_filters, output_height, output_width), dtype='object')
    for o, y, x in itertools.product(range(num_filters), range(output_height), range(output_width)):
        neurons[o, y, x] = ConvNeuron(weights[o], grad_weights[o], bias[o], grad_bias[o])

    ens = net.init_ensemble(neurons)

    input_shape = input_ensemble.shape

    def mapping(c, y, x):
        in_y = fmax(y*stride_h - pad, 0)
        in_x = fmax(x*stride_w - pad, 0)
        out_y = fmin(in_y + kernel_h, output_height)
        out_x = fmin(in_x + kernel_w, output_width)
        return range(input_channels), range(in_y,out_y), range(in_x,out_x)

    net.add_connections(input_ensemble, ens, mapping)

    return ens
