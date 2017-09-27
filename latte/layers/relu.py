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

class ReLUNeuron(Neuron):
    def __init__(self):
        super().__init__()
        self.inputs = []
        self.grad_inputs = []

    def forward(self):
        #ANAND:temporarily commenting out following for fmax  7/5/17
        #self.value = max(self.input, float(0.0))
        #fmax(self.input, 0.0)
        self.value = fmax(self.input, 0.0)
    def backward(self):
        # self.grad_input = ifelse(self.input > 0.0, self.grad, 0.0)
        if self.input > 0.0:
            self.grad_input = self.grad
        else:
            self.grad_input = 0.0
        #self.grad_input = max(self.grad, float(0.0))
    def update_internal(self):
        pass

def ReLULayer(net, input_ensemble):
    
    relu_neurons = np.empty(input_ensemble.shape, dtype='object')

    for i in range(len(relu_neurons)):
        relu_neurons[i] = ReLUNeuron()

    relu_ens = net.init_activation_ensemble(relu_neurons, input_ensemble)
    relu_ens.parallelize(phase="forward", loop_var="_neuron_index_0")
    relu_ens.parallelize(phase="backward", loop_var="_neuron_index_0")
    if "value" in input_ensemble.tiling_info:
        relu_ens.parallelize(phase="forward", loop_var="_neuron_index_1_outer")
        relu_ens.parallelize(phase="backward", loop_var="_neuron_index_1_outer")
        #relu_ens.simd(phase="forward", loop_var="_neuron_index_1_inner")
        relu_ens.vectorize(phase="forward", loop_var="_neuron_index_1_inner", factor=latte.config.SIMDWIDTH)
    else:
        relu_ens.parallelize(phase="forward", loop_var="_neuron_index_1")
        relu_ens.parallelize(phase="backward", loop_var="_neuron_index_1")
    '''
    h_unroll_factor=7
    w_unroll_factor=4
    if relu_ens.shape[1] % h_unroll_factor == 0 and relu_ens.shape[2] % w_unroll_factor == 0:
      relu_ens.unroll(phase="forward", loop_var="_neuron_index_2", factor=h_unroll_factor, unroll_type= 1)
      relu_ens.unroll(phase="forward", loop_var="_neuron_index_3", factor=w_unroll_factor, unroll_type=1)
    '''
    if net.cbr_fusion or "ON" in latte.config.AUTO_FUSION:
      #print("FUSION ENABLED")
      net.fuse_cbr(input_ensemble, relu_ens)

    return relu_ens
