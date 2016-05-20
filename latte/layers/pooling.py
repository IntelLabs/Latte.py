import numpy as np
from ..neuron import Neuron
from ..ensemble import Ensemble
import itertools

class MaxNeuron(Neuron):
    batch_fields     = Neuron.batch_fields + ["mask"]
    zero_init_fields = Neuron.zero_init_fields + ["mask"]

    def __init__(self):
        super().__init__()
        self.inputs = []
        self.grad_inputs = []

        self.mask = np.zeros((2,), dtype=np.int32)

    def forward(self):
        max_value = -INFINITY
        for j in range_dim(self.inputs, 1):
            for k in range_dim(self.inputs, 2):
                if self.inputs[0,j,k] > max_value:
                    max_value = self.inputs[0,j,k]
                    self.mask[0] = j
                    self.mask[1] = k
        self.value = max_value

    def backward(self):
        j = self.mask[0]
        k = self.mask[1]
        self.grad_inputs[0,j,k] += self.grad

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
    neurons = np.array([MaxNeuron() for _ in range(np.prod(shape))]).reshape(shape)

    pooling_ens = net.init_ensemble(neurons)

    input_shape = input_ensemble.shape

    def mapping(c, y, x):
        in_y = y*stride_h - pad
        in_x = x*stride_w - pad
        return range(c, c+1), range(in_y, in_y+kernel_h), range(in_x, in_x+kernel_w)

    net.add_connections(input_ensemble, pooling_ens, mapping)

    return pooling_ens
