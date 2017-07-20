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
import itertools
import latte.config

class DropoutNeuron(Neuron):
    batch_fields = Neuron.batch_fields + ["randval"]
    zero_init_fields = Neuron.zero_init_fields + ["randval"]

    def __init__(self, dropout_ratio):
        super().__init__()
        self.inputs = []
        self.grad_inputs = []

        self.ratio = dropout_ratio
        self.randval = 0.0

    def forward(self):
        self.randval = rand()
        scale = 1./(1.-self.ratio)
        if self.randval < self.ratio:
            self.value = 0.0
        else:
            self.value *= scale
        
    def backward(self):
        scale = 1./(1.-self.ratio)
        if self.randval < self.ratio:
            self.grad_input = 0.0
        else:
            self.grad_input *= scale 

    def update_internal(self):
        pass
        
def DropoutLayer(net, input_ensemble, ratio=0.5):
    
    # neurons = np.array(
    #     [(0.0, 0.0, [], [], ratio, 0.0)],
    #     dtype=[
    #         ('value', float),
    #         ('grad', float),
    #         ('inputs', list),
    #         ('grad_inputs', list),
    #         ('ratio', float),
    #         ('randval', float)
    #         ]
    # )
    neurons = np.empty(input_ensemble.shape, dtype='object')
    neurons[:] = DropoutNeuron(ratio)
    # neurons = np.array([DropoutNeuron(ratio) for _ in range(np.prod(input_ensemble.shape))])
    # neurons = neurons.reshape(input_ensemble.shape)

    dropout_ens = net.init_activation_ensemble(neurons, input_ensemble)
    dropout_ens.parallelize(phase="forward", loop_var="_neuron_index_0")
    dropout_ens.parallelize(phase="backward", loop_var="_neuron_index_0")
    if "value" in input_ensemble.tiling_info:
        tiled_dims = input_ensemble.tiling_info["value"]
        for dim, factor in tiled_dims:
            dropout_ens.tile('inputs', dim=dim, factor=factor)
            dropout_ens.tile('grad_inputs', dim=dim, factor=factor)
        dropout_ens.tile('value', dim=0, factor=latte.config.SIMDWIDTH)
        dropout_ens.tile('grad', dim=0, factor=latte.config.SIMDWIDTH)
        dropout_ens.tile('randval', dim=0, factor=latte.config.SIMDWIDTH)
        dropout_ens.tile('ratio', dim=0, factor=latte.config.SIMDWIDTH)
        dropout_ens.parallelize(phase="forward", loop_var="_neuron_index_1_outer")
        dropout_ens.parallelize(phase="backward", loop_var="_neuron_index_1_outer")
    else:
        dropout_ens.parallelize(phase="forward", loop_var="_neuron_index_1")
        dropout_ens.parallelize(phase="backward", loop_var="_neuron_index_1")

    return dropout_ens
