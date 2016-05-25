import numpy as np
from ..neuron import Neuron
from ..ensemble import Ensemble
import itertools
import math

class InterpolatedNeuron(Neuron):

    def __init__(self, ih, iw, resize_factor):
        super().__init__()
        self.inputs = []
        self.grad_inputs = []
    
        self.ih = ih
        self.iw = iw
        self.resize_factor = resize_factor

    def forward(self):
        r_f = self.ih * self.resize_factor
        c_f = self.iw * self.resize_factor  
        delta_r = r_f - floor(r_f)
        delta_c = c_f - floor(c_f)        
    
        j = 0
        k = 0
        
        self.value = self.inputs[0,j,k]*(1-delta_r)*(1-delta_c) + self.inputs[0,j+1,k]*delta_r*(1-delta_c) + self.inputs[0,j,k+1]*(1-delta_r)*delta_c + self.inputs[0,j+1,k+1]*delta_r*delta_c 

    def backward(self):
        r_f = self.ih * self.resize_factor
        c_f = self.iw * self.resize_factor  
        delta_r = r_f - floor(r_f)
        delta_c = c_f - floor(c_f)

        j = 0
        k = 0

        self.grad_inputs[0,j,k] += (1-delta_r)*(1-delta_c)*self.grad
        self.grad_inputs[0,j+1,k] += delta_r*(1-delta_c)*self.grad
        self.grad_inputs[0,j,k+1] += (1-delta_r)*delta_c*self.grad
        self.grad_inputs[0,j+1,k+1] += delta_r*delta_c*self.grad

def InterpolationLayer(net, input_ensemble, kernel=2, pad=0, resize_factor=1.0):
    assert input_ensemble.ndim == 3, "InterpolationLayer only supports 3-d input"

    if isinstance(kernel, tuple):
        assert len(kernel) == 2, "kernel as a tuple must be of length 2"
        kernel_h, kernel_w = kernel
    else:
        kernel_h, kernel_w = kernel, kernel


    if isinstance(pad, tuple):
        assert len(pad) == 2, "pad as a tuple must be of length 2"
        pad_h, pad_w = pad
    else:
        pad_h, pad_w = pad, pad

    input_channels, input_height, input_width = input_ensemble.shape
    output_width = math.floor(input_width * resize_factor)
    output_height = math.floor(input_height * resize_factor)

    shape = (input_channels, output_height, output_width)
    
    neurons = np.zeros(shape, dtype='object')
    
    for c in range(input_channels):
        for h in range(output_height):
            for w in range(output_width):
                neurons[c,h,w] = InterpolatedNeuron(h,w,resize_factor)

    interpolation_ens = net.init_ensemble(neurons)

    input_shape = input_ensemble.shape

    def mapping(c, y, x):
        in_y = math.floor(y/resize_factor) - pad
        in_x = math.floor(x/resize_factor) - pad
        return range(c, c+1), range(in_y, in_y+kernel_h), range(in_x, in_x+kernel_w)

    net.add_connections(input_ensemble, interpolation_ens, mapping)

    return interpolation_ens
