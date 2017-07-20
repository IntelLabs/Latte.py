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
from ..neuron import WeightedNeuron, BiasNeuron
from ..ensemble import Ensemble, EnsembleGroup
import latte.core

class FCNeuron(WeightedNeuron):
    pass
    # def forward(self):
    #     for i in range_dim(self.inputs, 0):
    #         self.value += self.inputs[i] * self.weights[i]

    # def backward(self):
    #     for i in range_dim(self.inputs, 0):
    #         self.grad_inputs[i] += self.grad * self.weights[i]

    # def update_internal(self):
    #     for i in range_dim(self.inputs, 0):
    #         self.grad_weights[i] += self.grad * self.inputs[i]

def FullyConnectedLayerNoBias(net, input_ensemble, num_outputs):
    fan_in = len(input_ensemble)
    scale = np.sqrt(3.0 / fan_in)
    weights = np.random.rand(num_outputs, *input_ensemble.shape).astype(np.float32) * (2 * scale) - scale
    weights_grad = np.zeros_like(weights)

    neurons = np.array([FCNeuron(weights[i], weights_grad[i]) for i in range(num_outputs)])

    ens = net.init_ensemble(neurons)

    input_shape = input_ensemble.shape
    flattened = np.prod(input_shape)
    mapping_range = tuple(range(d) for d in input_ensemble.shape)

    def mapping(x):
        return mapping_range

    net.add_connections(input_ensemble, ens, mapping)

    input_ensemble.tile('value', dim=0, factor=latte.config.SIMDWIDTH)
    if "value" in input_ensemble.tiling_info:
        input_ensemble.tile('value', dim=0, factor=latte.config.SIMDWIDTH)
        input_ensemble.tile('grad', dim=0, factor=latte.config.SIMDWIDTH)
        ens.tile('inputs', dim=0, factor=latte.config.SIMDWIDTH)
        ens.tile('grad_inputs', dim=0, factor=latte.config.SIMDWIDTH)
    # ens.privatize('grad_weights')

    ens.tile('weights', dim=0, factor=latte.config.SIMDWIDTH)
    ens.tile('grad_weights', dim=0, factor=latte.config.SIMDWIDTH)
    if "value" in input_ensemble.tiling_info:
        ens.tile('weights', dim=1, factor=latte.config.SIMDWIDTH)
        ens.tile('grad_weights', dim=1, factor=latte.config.SIMDWIDTH)
    ens.tile('value', dim=0, factor=latte.config.SIMDWIDTH)
    ens.tile('grad', dim=0, factor=latte.config.SIMDWIDTH)
    if "OPENCL" not in latte.config.parallel_strategy:
        ens.vectorize(phase="forward", loop_var="_neuron_index_1_inner", factor=latte.config.SIMDWIDTH)
        #factor = 8
        factor = 16
        while net.batch_size % factor != 0:
            factor -= 1
            
        if latte.config.parallel_strategy != "FLOWGRAPH_LOOP":
          ens.unroll(phase="forward", loop_var="_neuron_index_0", factor=factor)
    ens.parallelize(phase="forward", loop_var="_neuron_index_0")
    ens.parallelize(phase="forward", loop_var="_neuron_index_1_outer")
    ens.parallelize(phase="backward", loop_var="_neuron_index_0")
    ens.parallelize(phase="backward", loop_var="__unique_loopvar0_outer")
    ens.parallelize(phase="update_internal", loop_var="_neuron_index_1_outer")
    ens.parallelize(phase="update_internal", loop_var="__unique_loopvar0_outer")
    ens.swap_loops(phase="update_internal", loop_vars=("_neuron_index_0", "_neuron_index_1_outer"))
    for i in range(1, input_ensemble.ndim):
        ens.swap_loops(phase="backward", loop_vars=("_neuron_index_1_inner", "__unique_loopvar{}".format(i)))

    if "value" in input_ensemble.tiling_info:
        if "OPENCL" not in latte.config.parallel_strategy:
            ens.vectorize(phase="backward", loop_var="__unique_loopvar0_inner", factor=latte.config.SIMDWIDTH)
            #factor = 8
            factor = 16
            while net.batch_size % factor != 0:
                factor -= 1
            if latte.config.parallel_strategy != "FLOWGRAPH_LOOP":
              ens.unroll(phase="backward", loop_var="_neuron_index_0", factor=factor)
              ens.unroll(phase="update_internal", loop_var="_neuron_index_0", factor=factor)
            ens.vectorize(phase="update_internal", loop_var="__unique_loopvar0_inner", factor=latte.config.SIMDWIDTH)
    # Added by Raj/Anand
    #ens.use_libxsmm(1)
    ens.stride=1
    return ens

def FullyConnectedLayer(net, input_ensemble, num_outputs):
    ens = FullyConnectedLayerNoBias(net, input_ensemble, num_outputs)

    bias = np.zeros((num_outputs, 1), dtype=np.float32)
    grad_bias = np.zeros_like(bias)

    bias_neurons = np.array([BiasNeuron(bias[i], grad_bias[i]) for i in range(num_outputs)])

    bias_ens = net.init_activation_ensemble(bias_neurons, ens)
    bias_ens.privatize('grad_bias')

    bias_ens.tile('bias', dim=0, factor=latte.config.SIMDWIDTH)
    bias_ens.tile('grad_bias', dim=0, factor=latte.config.SIMDWIDTH)


    bias_ens.parallelize(phase="forward", loop_var="_neuron_index_0")
    bias_ens.parallelize(phase="forward", loop_var="_neuron_index_1_outer")
    bias_ens.parallelize(phase="update_internal", loop_var="_neuron_index_0")
    bias_ens.parallelize(phase="update_internal", loop_var="_neuron_index_1_outer")
    if "OPENCL" not in latte.config.parallel_strategy:
        factor = 16
        while net.batch_size % factor != 0:
            factor -= 1
        if latte.config.parallel_strategy != "FLOWGRAPH_LOOP":
          bias_ens.unroll(phase="forward", loop_var="_neuron_index_0", factor=factor)

    return EnsembleGroup(ens, bias_ens)
