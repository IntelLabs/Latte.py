import numpy as np
from ..neuron import Neuron
from ..ensemble import Ensemble
import itertools
import math
from math import floor

class InterpolatedNeuron(Neuron):

    def __init__(self, delta_r, delta_c):
        super().__init__()
        self.inputs = []
        self.grad_inputs = []
    
        self.delta_r = delta_r
        self.delta_c = delta_c

    def forward(self):
      
        self.value = \
            self.inputs[0,0,0] * (1-self.delta_r) * (1-self.delta_c) + \
            self.inputs[0,1,0] * self.delta_r     * (1-self.delta_c) + \
            self.inputs[0,0,1] * (1-self.delta_r) * self.delta_c     + \
            self.inputs[0,1,1] * self.delta_r     * self.delta_c 

    def backward(self):

        self.grad_inputs[0,0,0] += (1-self.delta_r) * (1-self.delta_c) * self.grad
        self.grad_inputs[0,1,0] += self.delta_r     * (1-self.delta_c) * self.grad
        self.grad_inputs[0,0,1] += (1-self.delta_r) * self.delta_c     * self.grad
        self.grad_inputs[0,1,1] += self.delta_r     * self.delta_c     * self.grad

    def update_internal(self):
        pass

def InterpolationLayer(net, input_ensemble, pad=0, resize_factor=1.0):
    assert input_ensemble.ndim == 3, "InterpolationLayer only supports 3-d input"
    
    if isinstance(pad, tuple):
        assert len(pad) == 2, "pad as a tuple must be of length 2"
        pad_h, pad_w = pad
    else:
        pad_h, pad_w = pad, pad

    input_channels, input_height, input_width = input_ensemble.shape
    output_width = math.floor(input_width * resize_factor) - pad
    output_height = math.floor(input_height * resize_factor) - pad

    shape = (input_channels, output_height, output_width)
    
    neurons = np.zeros(shape, dtype='object')
    
    for c in range(input_channels):
        for h in range(output_height):
            for w in range(output_width):
                r_f = h * (1/resize_factor)
                c_f = w * (1/resize_factor)  
                delta_r = r_f - math.floor(r_f)
                delta_c = c_f - math.floor(c_f)
                neurons[c,h,w] = InterpolatedNeuron(delta_r, delta_c)

    interpolation_ens = net.init_ensemble(neurons)

    input_shape = input_ensemble.shape

    def mapping(c, y, x):
        in_y = floor(y / resize_factor) - pad
        in_x = floor(x / resize_factor) - pad
        return range(c, c+1), range(in_y, in_y+1), range(in_x, in_x+1)

    net.add_connections(input_ensemble, interpolation_ens, mapping, clamp=True)
    interpolation_ens.parallelize(phase="forward", loop_var="_neuron_index_0")
    interpolation_ens.parallelize(phase="backward", loop_var="_neuron_index_0")
    if "value" in input_ensemble.tiling_info:
        interpolation_ens.parallelize(phase="forward", loop_var="_neuron_index_1_outer")
        # interpolation_ens.parallelize(phase="backward", loop_var="_neuron_index_1_outer")
    else:
        interpolation_ens.parallelize(phase="forward", loop_var="_neuron_index_1")
        # interpolation_ens.parallelize(phase="backward", loop_var="_neuron_index_1")

    return interpolation_ens
