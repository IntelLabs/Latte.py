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
import math
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
        factor = 28
        while net.batch_size % factor != 0:
            factor -= 1
        if latte.config.parallel_strategy != "FLOWGRAPH_LOOP":
            factor = 28
            while math.ceil(num_outputs/latte.config.SIMDWIDTH) % factor != 0: 
                factor -= 1
 
            if (int(latte.config.nthreads)*4 > num_outputs//latte.config.SIMDWIDTH):
                factor2 = 16
            else:
                factor2 = 8


            while net.batch_size % factor2 != 0:
                factor2 -= 1
 
            #if len(input_shape) > 2:
            ens.unroll(phase="forward", loop_var="_neuron_index_0", factor=factor2)
            ens.unroll(phase="forward", loop_var="__unique_loopvar0_inner", factor=1)

            #ens.unroll_no_jam(phase="forward", loop_var="__unique_loopvar2", factor=3)
            # Anand experiment with different parallelization and loop order 9/8/17
            #ens.unroll(phase="forward", loop_var="_neuron_index_1_outer", factor=8, unroll_type=1)
            if "AVX-512" in latte.config.vec_config and len(input_shape) > 2: 
                    ens.prefetch(phase="forward", prefetch_dict_list={'weights': [1, "__unique_loopvar0_inner", -4,2,"__unique_loopvar1",1,1,0]})
            #ens.prefetch(phase="forward", prefetch_dict_list={'weights': [1, "__unique_loopvar0_inner", -4,2,"__unique_loopvar1",1,1,0],'value': [1, "_neuron_index_0", -3,32,"_neuron_index_0",1,32,0]})
  #'inputs': [2, "__unique_loopvar0_inner", -3,32, 16,"__unique_loopvar1","__unique_loopvar1","__unique_loopvar1", 1,1,2,0]})
            #ens.prefetch(phase="forward", prefetch_dict_list={'value': [1, "_neuron_index_0", -3,32,"_neuron_index_0",1,32,0]})
            # 'inputs': [2, "i_inner", -3, fp_pf_factor, fp_cache_line, fp_pf_loop, "_neuron_index_2", "_neuron_index_2", stride_h, 1, 2, 0]}) 
            #ens.swap_loops(phase="forward", loop_vars=("_neuron_index_1_outer", "__unique_loopvar0_outer"))
            if len(input_shape) > 1:
               ens.swap_loops(phase="forward", loop_vars=( "__unique_loopvar1",  "__unique_loopvar0_inner"))
            if len(input_shape) > 2:
               ens.swap_loops(phase="forward", loop_vars=( "__unique_loopvar2",  "__unique_loopvar0_inner"))
               #ens.prefetch(phase="forward", prefetch_dict_list={'inputs': [1, "__unique_loopvar2", -4, 1, "__unique_loopvar0_outer", 1,1, 0]})
               #ens.unroll(phase="forward", loop_var="__unique_loopvar2", factor=3)



    if (int(latte.config.nthreads)*4 > num_outputs//latte.config.SIMDWIDTH):
        ens.parallelize(phase="forward", loop_var="_neuron_index_0")
    ens.parallelize(phase="forward", loop_var="_neuron_index_1_outer")#ANAND 9/8/17 
    ens.parallelize(phase="backward", loop_var="_neuron_index_0")
    ens.parallelize(phase="backward", loop_var="__unique_loopvar0_outer")
    ens.parallelize(phase="update_internal", loop_var="_neuron_index_1_outer")
    ens.parallelize(phase="update_internal", loop_var="__unique_loopvar0_outer")
    ens.swap_loops(phase="update_internal", loop_vars=("_neuron_index_0", "_neuron_index_1_outer"))
    ens.swap_loops(phase="forward", loop_vars=( "_neuron_index_0",  "_neuron_index_1_outer")) 

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
    
    ens.vectorize(phase="forward", loop_var="_neuron_index_1_inner", factor=latte.config.SIMDWIDTH)


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
    #bias_ens.unroll(phase="forward", loop_var="_neuron_index_0", factor=32)


    if (int(latte.config.nthreads)*4 > num_outputs//latte.config.SIMDWIDTH):
        bias_ens.parallelize(phase="forward", loop_var="_neuron_index_0")
    

    bias_ens.parallelize(phase="forward", loop_var="_neuron_index_1_outer")
    bias_ens.swap_loops(phase="forward", loop_vars=( "_neuron_index_0",  "_neuron_index_1_outer"))  
    bias_ens.parallelize(phase="update_internal", loop_var="_neuron_index_0")
    bias_ens.parallelize(phase="update_internal", loop_var="_neuron_index_1_outer")
  

    bias_ens.vectorize(phase="forward", loop_var="_neuron_index_1_inner", factor=latte.config.SIMDWIDTH)
 

    if "OPENCL" not in latte.config.parallel_strategy:
        if (int(latte.config.nthreads)*4 > num_outputs//latte.config.SIMDWIDTH):
            factor = 16
        else:
            factor = 8

        while net.batch_size % factor != 0:
            factor -= 1
        if latte.config.parallel_strategy != "FLOWGRAPH_LOOP":
          bias_ens.unroll(phase="forward", loop_var="_neuron_index_0", factor=factor)#factor=factor


    grp_ens = EnsembleGroup(ens, bias_ens)
    net.fuse_cbr(grp_ens, relu_ens=None, is_fc=True)


    return grp_ens
