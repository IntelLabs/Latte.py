'''
Copyright (c) 2015, Intel Corporation

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright notice,
      this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of Intel Corporation nor the names of its contributors
      may be used to endorse or promote products derived from this software
      without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
'''
import numpy as np
from ..neuron import Neuron
from ..ensemble import Ensemble
import latte
import itertools

class TanhNeuron(Neuron):
    def __init__(self):
        super().__init__()
        self.inputs = []
        self.grad_inputs = []

    def forward(self):
        self.value = tanh(self.input)

    def backward(self):
            self.grad_input = (self.value * (1.0 - self.value)) * self.grad

    def update_internal(self):
        pass

def TanhLayer(net, input_ensemble):
    
    tanh_neurons = np.empty(input_ensemble.shape, dtype='object')

    for i in range(len(tanh_neurons)):
        tanh_neurons[i] = TanhNeuron()

    tanh_ens = net.init_activation_ensemble(tanh_neurons, input_ensemble)
    
    tanh_ens.parallelize(phase="forward", loop_var="_neuron_index_0")
    tanh_ens.parallelize(phase="backward", loop_var="_neuron_index_0")
    
    if "value" in input_ensemble.tiling_info:
        tanh_ens.parallelize(phase="forward", loop_var="_neuron_index_1_outer")
        tanh_ens.parallelize(phase="backward", loop_var="_neuron_index_1_outer")
    else:
        tanh_ens.parallelize(phase="forward", loop_var="_neuron_index_1")
        tanh_ens.parallelize(phase="backward", loop_var="_neuron_index_1")
    
    return tanh_ens
