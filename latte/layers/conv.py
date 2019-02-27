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
import itertools
import latte.core
import math

SIMDWIDTH = latte.config.SIMDWIDTH

class ConvNeuron(WeightedNeuron):
    def forward(self):
        for j in range_dim(self.inputs, 2):
            for k in range_dim(self.inputs, 1):
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
    conv_ens = ConvLayerNoBias(net, input_ensemble, num_filters, kernel, stride, pad, dilation)



    input_channels, input_height, input_width = input_ensemble.shape

    kernel_eff = kernel + (kernel-1) * (dilation-1)

    output_width = ((input_width + 2 * pad - kernel_eff) // stride) + 1
    output_height = ((input_height + 2 * pad - kernel_eff) // stride) + 1

    #output_width = ((input_width - kernel * dilation + 2 * pad) // stride) + 1
    #output_height = ((input_height - kernel * dilation + 2 * pad) // stride) + 1

    if num_filters % latte.config.SIMDWIDTH != 0:
        num_filters_pad = latte.config.SIMDWIDTH - (num_filters % latte.config.SIMDWIDTH)
    else:
        num_filters_pad = 0
   
    bias = np.zeros((num_filters, 1), dtype=np.float32)
 
    bias = np.lib.pad(bias, ((0, num_filters_pad), (0, 0)), 'constant')
    grad_bias = np.zeros_like(bias)

    num_filters += num_filters_pad
    bias_neurons = np.empty((num_filters, output_height, output_width), dtype='object')
    for o in range(num_filters):
        bias_neurons[o, :, :] = BiasNeuron(bias[o], grad_bias[o])

    bias_ens = net.init_activation_ensemble(bias_neurons, conv_ens)

    # Begin Optimizations

    # bias_ens.privatize('grad_bias')
    bias_ens.tile('bias', dim=0, factor=SIMDWIDTH)
    bias_ens.tile('grad_bias', dim=0, factor=SIMDWIDTH)
    bias_ens.parallelize(phase="forward", loop_var="_neuron_index_0")
    bias_ens.parallelize(phase="forward", loop_var="_neuron_index_1_outer")
    bias_ens.swap_loops(phase="update_internal", loop_vars=("_neuron_index_0", "_neuron_index_1_outer"))
    
    #bias_ens.parallelize(phase="update_internal", loop_var="_neuron_index_0")

    bias_ens.parallelize(phase="update_internal", loop_var="_neuron_index_1_outer")
    bias_ens.vectorize(phase="forward", loop_var="_neuron_index_1_inner", factor=SIMDWIDTH)
    bias_ens.vectorize(phase="update_internal", loop_var="_neuron_index_1_inner", factor=SIMDWIDTH)
    '''
    h_unroll_factor = 7
    w_unroll_factor = 4
    if bias_ens.shape[1] % h_unroll_factor == 0 and bias_ens.shape[2] % w_unroll_factor == 0:
      bias_ens.unroll(phase="forward", loop_var="_neuron_index_2", factor=h_unroll_factor, unroll_type = 1)
      bias_ens.unroll(phase="forward", loop_var="_neuron_index_3", factor=w_unroll_factor, unroll_type =1)
    '''
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

    if input_ensemble.shape[0] < latte.config.SIMDWIDTH:
        input_channel_pad = latte.config.SIMDWIDTH - input_ensemble.shape[0]
    elif input_ensemble.shape[0] % latte.config.SIMDWIDTH != 0:
        input_channel_pad = latte.config.SIMDWIDTH - (input_ensemble.shape[0] % latte.config.SIMDWIDTH)
    else:
        input_channel_pad = 0

    if num_filters % latte.config.SIMDWIDTH != 0:
        num_filters_pad = latte.config.SIMDWIDTH - (num_filters % latte.config.SIMDWIDTH)
    else:
        num_filters_pad = 0

    input_channels, input_height, input_width = input_ensemble.shape

    kernel_h_eff = kernel_h + (kernel_h-1) * (dilation - 1)
    kernel_w_eff = kernel_w + (kernel_w-1) * (dilation - 1)

    output_width = ((input_width + 2 * pad_w - kernel_w_eff) // stride_w) + 1
    output_height = ((input_height + 2 * pad_h - kernel_h_eff) // stride_h) + 1
 
    #output_width = ((input_width - kernel_w * dilation + 2 * pad_w) // stride_w) + 1
    #output_height = ((input_height - kernel_h * dilation + 2 * pad_h) // stride_h) + 1

    scale = np.sqrt(3.0 / (input_channels * kernel_h * kernel_w))
    weights = np.random.uniform(-scale, scale, (num_filters, input_channels, kernel_h,
            kernel_w)).astype(np.float32)

    weights = np.lib.pad(weights, ((0, num_filters_pad), (0, input_channel_pad), (0, 0), (0, 0)), 'constant', constant_values=(0,))
    
    if  input_ensemble.shape[0] > latte.config.SIMDWIDTH:
        input_channels += input_channel_pad
    num_filters += num_filters_pad
    grad_weights = np.zeros_like(weights)

    neurons = np.empty((num_filters, output_height, output_width), dtype='object')
    for o in range(num_filters):
        neurons[o, :, :] = ConvNeuron(weights[o], grad_weights[o])
    
    conv_ens = net.init_ensemble(neurons)

    def mapping(c, y, x):
        in_y = y*stride_h
        in_x = x*stride_w
        return (range(input_channels),
                range(in_y,in_y+kernel_h_eff,dilation),
                range(in_x,in_x+kernel_w_eff,dilation))

    #input_ensemble.set_padding((0, input_channel_pad), (pad, pad), (pad, pad))

    if  input_ensemble.shape[0] > latte.config.SIMDWIDTH:
        input_ensemble.set_padding((0, input_channel_pad), (pad, pad), (pad, pad))
    else:
        input_ensemble.set_padding((0, 0), (pad, pad), (pad, pad))


    net.add_connections(input_ensemble, conv_ens, mapping)
    
    # Begin Optimizations
    

    #input_ensemble.tile('value', dim=0, factor=SIMDWIDTH)
    #input_ensemble.tile('grad', dim=0, factor=SIMDWIDTH)
    


    if  input_ensemble.shape[0] > latte.config.SIMDWIDTH:
        input_ensemble.tile('value', dim=0, factor=SIMDWIDTH)
        input_ensemble.tile('grad', dim=0, factor=SIMDWIDTH)
        conv_ens.tile('inputs', dim=0, factor=SIMDWIDTH)
        conv_ens.tile('grad_inputs', dim=0, factor=SIMDWIDTH)
 
    else:
        '''
        input_ensemble.tile('value', dim=0, factor= latte.config.SIMDWIDTH)
        input_ensemble.tile('grad', dim=0, factor=latte.config.SIMDWIDTH)
        print("input ensemble shape\n")
        print(input_ensemble.shape[0])
        conv_ens.tile('grad_inputs', dim=0, factor=latte.config.SIMDWIDTH)
        conv_ens.tile('inputs', dim=0, factor= latte.config.SIMDWIDTH)
        '''


        input_ensemble.tile('value', dim=0, factor=input_ensemble.shape[0])
        input_ensemble.tile('grad', dim=0, factor=input_ensemble.shape[0])
        conv_ens.tile('grad_inputs', dim=0, factor=input_ensemble.shape[0])
        conv_ens.tile('inputs', dim=0, factor=input_ensemble.shape[0])


        #conv_ens.tile_loop(var='i',factor=input_ensemble.shape[0], phase="forward")
        #$conv_ens.unroll(phase="forward", loop_var ="i_inner",factor=input_ensemble.shape[0])
        #conv_ens.tile_loop(var='i',factor=input_ensemble.shape[0], phase="forward") 







    #conv_ens.tile('inputs', dim=0, factor=SIMDWIDTH)
    #conv_ens.tile('grad_inputs', dim=0, factor=SIMDWIDTH)

    conv_ens.tile('weights', dim=0, factor=SIMDWIDTH)
    conv_ens.tile('weights', dim=1, factor=SIMDWIDTH)
    conv_ens.transpose('weights', -2, -1)

    conv_ens.tile('grad_weights', dim=0, factor=SIMDWIDTH)
    conv_ens.tile('grad_weights', dim=1, factor=SIMDWIDTH)
    # conv_ens.privatize('grad_weights')
    # conv_ens.transpose('grad_weights', -2, -1)

    conv_ens.tile('value', dim=0, factor=SIMDWIDTH)
    conv_ens.tile('grad', dim=0, factor=SIMDWIDTH)

    conv_ens.stride = stride

    # Added by Raj/Anand
    if "LIBXSMM" in latte.config.codegen_strategy:
      conv_ens.use_libxsmm(1)
      return conv_ens

    if "OPENCL" not in latte.config.parallel_strategy:
        conv_ens.vectorize(phase="forward", loop_var="_neuron_index_1_inner", factor=SIMDWIDTH)
    conv_ens.parallelize(phase="forward", loop_var="_neuron_index_0")
    conv_ens.parallelize(phase="forward", loop_var="_neuron_index_1_outer")
    if "OPENCL" not in latte.config.parallel_strategy:
        #if input_ensemble.shape[0] > SIMDWIDTH:
        conv_ens.vectorize(phase="backward", loop_var="i_inner", factor=SIMDWIDTH)
    conv_ens.parallelize(phase="backward", loop_var="_neuron_index_0")
    conv_ens.parallelize(phase="backward", loop_var="i_outer")
    conv_ens.swap_loops(phase="backward", loop_vars=("_neuron_index_1_inner", "j"))
    conv_ens.swap_loops(phase="backward", loop_vars=("_neuron_index_1_inner", "k"))
    #conv_ens.unroll_no_jam(phase="backward", loop_var="k", factor=kernel, unroll_type=1)
    conv_ens.unroll_no_jam(phase="backward", loop_var="j", factor=kernel, unroll_type=1)
    if "OPENCL" not in latte.config.parallel_strategy:
        #if input_ensemble.shape[0] > SIMDWIDTH:
        conv_ens.vectorize(phase="update_internal", loop_var="i_inner", factor=SIMDWIDTH)
    #conv_ens.parallelize(phase="update_internal", loop_var="_neuron_index_1_outer")
    #conv_ens.parallelize(phase="update_internal", loop_var="i_outer")
    conv_ens.swap_loops(phase="update_internal", loop_vars=("_neuron_index_0", "_neuron_index_1_outer"))
    conv_ens.swap_loops(phase="update_internal", loop_vars=("_neuron_index_1_inner", "j"))
    conv_ens.swap_loops(phase="update_internal", loop_vars=("_neuron_index_1_inner", "k"))
    
    if input_ensemble.shape[0] <= SIMDWIDTH and kernel >= 3:# and stride==1 :
            parallelism = conv_ens.shape[0]//latte.config.SIMDWIDTH
            if parallelism*kernel_h*kernel_w < 4*int(latte.config.nthreads):
                conv_ens.swap_loops(phase="update_internal", loop_vars=("i_outer", "k"))
                #conv_ens.swap_loops(phase="update_internal", loop_vars=("i_outer", "k"))

                conv_ens.swap_loops(phase="update_internal", loop_vars=("_neuron_index_3", "j"))
                #conv_ens.swap_loops(phase="update_internal", loop_vars=("_neuron_index_3", "k"))
                conv_ens.swap_loops(phase="update_internal", loop_vars=("_neuron_index_2", "j"))
                #conv_ens.swap_loops(phase="update_internal", loop_vars=("_neuron_index_2", "k"))
                conv_ens.swap_loops(phase="update_internal", loop_vars=("_neuron_index_0", "j"))
                conv_ens.swap_loops(phase="update_internal", loop_vars=("j", "k"))


                #conv_ens.swap_loops(phase="update_internal", loop_vars=("_neuron_index_2", "k"))

                #conv_ens.reset_parallelize(phase="update_internal")
                conv_ens.parallelize(phase="update_internal", loop_var="_neuron_index_1_outer")
                conv_ens.parallelize(phase="update_internal", loop_var="j")
                conv_ens.parallelize(phase="update_internal", loop_var="k")

            elif parallelism*kernel_h < 4*int(latte.config.nthreads):
                conv_ens.parallelize(phase="update_internal", loop_var="_neuron_index_1_outer")
                conv_ens.swap_loops(phase="update_internal", loop_vars=("i_outer", "j"))
                conv_ens.parallelize(phase="update_internal", loop_var="j")
            else:
                conv_ens.parallelize(phase="update_internal", loop_var="_neuron_index_1_outer")
                conv_ens.parallelize(phase="update_internal", loop_var="i_outer")  
    else:
            conv_ens.parallelize(phase="update_internal", loop_var="_neuron_index_1_outer")
            conv_ens.parallelize(phase="update_internal", loop_var="i_outer")


  
    '''  
    if input_ensemble.shape[0] < SIMDWIDTH and kernel > 3:
        conv_ens.swap_loops(phase="update_internal", loop_vars=("i_outer", "j"))
        conv_ens.swap_loops(phase="update_internal", loop_vars=("i_outer", "k"))
        conv_ens.parallelize(phase="update_internal", loop_var="j")
        conv_ens.parallelize(phase="update_internal", loop_var="k")
    #else:
    #    conv_ens.unroll_no_jam(phase="update_internal", loop_var="j", factor=kernel, unroll_type=1)   
    '''
    if "OPENCL" not in latte.config.parallel_strategy:
      '''
      h_unroll_factor = 7
      w_unroll_factor = 4
      if conv_ens.shape[1] % h_unroll_factor == 0 and conv_ens.shape[2] % w_unroll_factor == 0:
        conv_ens.swap_loops(phase="forward", loop_vars=("i_outer", "_neuron_index_2"))
        conv_ens.swap_loops(phase="forward", loop_vars=("i_outer", "_neuron_index_3"))
        conv_ens.unroll(phase="forward", loop_var="_neuron_index_2", factor=h_unroll_factor, unroll_type= 1)
        conv_ens.unroll(phase="forward", loop_var="_neuron_index_3", factor=w_unroll_factor, unroll_type=1)
        outer_unroll_factor = 28
        #outer_unroll_factor = 8
        #factor = 8
        while output_width % outer_unroll_factor != 0:
          outer_unroll_factor -= 1
        if (kernel_h == 1 and kernel_w == 1) or (stride_h >1): 
          inner_unroll_factor = 2
        else:
          inner_unroll_factor = 4
        if inner_unroll_factor > 1:
            conv_ens.unroll(phase="forward", loop_var="i_inner", factor=inner_unroll_factor)
            conv_ens.unroll(phase="backward", loop_var="_neuron_index_1_inner", factor=inner_unroll_factor)
        #update-internal
        conv_ens.unroll(phase="update_internal", loop_var="_neuron_index_3", factor=outer_unroll_factor)
      else:
      '''
      inner_unroll_factor=1
      if "AVX-512" in latte.config.vec_config:
        outer_unroll_factor = 28
        while output_width % outer_unroll_factor != 0:
          outer_unroll_factor -= 1
        if outer_unroll_factor > 1:
          conv_ens.unroll(phase="forward", loop_var="_neuron_index_3", factor=outer_unroll_factor)

        if (kernel_h == 1 and kernel_w == 1) or (stride_h >1): 
          inner_unroll_factor = 2
        else:
          #inner_unroll_factor = 4
          inner_unroll_factor = min(4, 32 - outer_unroll_factor)
          if inner_unroll_factor > 16:
            inner_unroll_factor = 16
          while 16 % inner_unroll_factor != 0:
            inner_unroll_factor -= 1
        if inner_unroll_factor > 1 and input_ensemble.shape[0] > latte.config.SIMDWIDTH:
          conv_ens.unroll(phase="forward", loop_var="i_inner", factor=inner_unroll_factor)
        #else:
        # conv_ens.unroll(phase="forward", loop_var="i_inner", factor=input_ensemble.shape[0])

        # a new approach is
        cache_line = 64
        fp_cache_line = int(cache_line/4)
        l1_size = 32768
        half_l1_size = 0.5 * l1_size
        data_needed_by_each_ifh = (output_width*cache_line) + (kernel_h*input_width*cache_line) + (kernel_h*kernel_w*SIMDWIDTH*cache_line)
        #syntax [prefetch_type=1, enclosing_loop_var, dim, prefetch_count, prefetch_loop_var, prefetch_multiplier, prefetch_constant, cacheline_hint]
        #syntax [prefetch_type=2, enclosing_loop_var, dim, prefetch_count, prefetch_offset, prefetch_dest_loop, prefetch_init_loop, prefetch_loop_var, prefetch_multiplier, prefetch_constant, cacheline_hint]
        #print("data=", data_needed_by_each_ifh, " half_li=", half_l1_size)
        if data_needed_by_each_ifh < l1_size:
          fp_pf_factor = ((kernel_h*(input_width))/((output_width/outer_unroll_factor) * kernel_h*kernel_w*(SIMDWIDTH/inner_unroll_factor)))
          #print ("fp_pf_factor=", fp_pf_factor)
          if fp_pf_factor > 1.0:
            fp_pf_factor = math.ceil(fp_pf_factor)
            fp_pf_loop = "i_inner"
          else:
            fp_pf_factor = ((kernel_h*(input_width))/((output_width/outer_unroll_factor) * kernel_h*kernel_w))
            #print ("fp_pf_factor=", fp_pf_factor)
            if fp_pf_factor > 1.0:
              fp_pf_factor = math.ceil(fp_pf_factor)
              fp_pf_loop = "k"
            else:
              fp_pf_factor = math.ceil((kernel_h*(input_width))/((output_width/outer_unroll_factor) * kernel_h))
              #print ("fp_pf_factor=", fp_pf_factor)
              fp_pf_loop = "j"
          #print("input_pf_factor:", fp_pf_factor, "input_pf_loop:", fp_pf_loop)
          if kernel_h == 1 and kernel_w == 1:
            #conv_ens.prefetch(phase="forward", prefetch_dict_list={'value': [1, "_neuron_index_3", -3, outer_unroll_factor, "_neuron_index_2", 1, 1, 0], 'inputs': [2, "i_inner", -3, fp_pf_factor, fp_cache_line, fp_pf_loop, "_neuron_index_2", "_neuron_index_2", stride_h, 1, 0],'weight':[1, "i_inner", -5, inner_unroll_factor, "i_outer", 1, 1, 1]})
            if outer_unroll_factor == output_width:
              conv_ens.prefetch(phase="forward", prefetch_dict_list={'value': [1, "_neuron_index_3", -3, outer_unroll_factor, "_neuron_index_2", 1, 1, 0], 'inputs': [2, "i_inner", -3, fp_pf_factor, fp_cache_line, fp_pf_loop, "_neuron_index_2", "_neuron_index_2", stride_h, 1, 2, 0]})
            else:
              conv_ens.prefetch(phase="forward", prefetch_dict_list={'value': [1, "_neuron_index_3", -2, outer_unroll_factor, "_neuron_index_3", 1, outer_unroll_factor, 0], 'inputs': [2, "i_inner", -3, fp_pf_factor, fp_cache_line, fp_pf_loop, "_neuron_index_2", "_neuron_index_2", stride_h, 1, 2, 0]})
          elif outer_unroll_factor == output_width:
            #3563GF
            conv_ens.prefetch(phase="forward", prefetch_dict_list={'value': [1, "_neuron_index_3", -3, outer_unroll_factor, "_neuron_index_2", 1, 1, 0]})
            #3400
            #conv_ens.prefetch(phase="forward", prefetch_dict_list={'value': [1, "_neuron_index_3", -3, outer_unroll_factor, "_neuron_index_2", 1, 1, 0], 'inputs': [2, "i_inner", -3, fp_pf_factor, fp_cache_line, fp_pf_loop, "_neuron_index_2", "_neuron_index_2", stride_h, 1, 0]})
          #elif input_channels <= SIMDWIDTH: 
          #  conv_ens.prefetch(phase="forward", prefetch_dict_list={'value': [1, "_neuron_index_3", -2, outer_unroll_factor, "_neuron_index_3", 1, outer_unroll_factor, 0], 'inputs': [2, "i_inner", -3, fp_pf_factor, fp_cache_line, fp_pf_loop, "_neuron_index_2", "_neuron_index_2", stride_h, 1, 0]})
          else:
            #conv_ens.prefetch(phase="forward", prefetch_dict_list={'value': [1, "_neuron_index_3", -3, outer_unroll_factor, "_neuron_index_2", 1, 1, 0], 'inputs': [2, "i_inner", -3, fp_pf_factor, fp_cache_line, fp_pf_loop, "_neuron_index_2", "_neuron_index_2", stride_h, 1, 0]})
            conv_ens.prefetch(phase="forward", prefetch_dict_list={'value': [1, "_neuron_index_3", -2, outer_unroll_factor, "_neuron_index_3", 1, outer_unroll_factor, 0], 'inputs': [2, "i_inner", -3, fp_pf_factor, fp_cache_line, fp_pf_loop, "_neuron_index_2", "_neuron_index_2", stride_h, 1, 2, 0]})
            #conv_ens.prefetch(phase="forward", prefetch_dict_list={'value': [1, "_neuron_index_3", -2, outer_unroll_factor, "_neuron_index_3", 1, outer_unroll_factor, 0], 'inputs': [2, "i_inner", -3, fp_pf_factor, fp_cache_line, fp_pf_loop, "_neuron_index_2", "_neuron_index_2", stride_h, 1, 0], 'weight':[1, "i_inner", -5, inner_unroll_factor, "i_outer", 1, 1, 1]})
          #conv_ens.prefetch(phase="forward", prefetch_dict_list={'value': [1, "_neuron_index_3", -2, outer_unroll_factor, "_neuron_index_3", 1, outer_unroll_factor, 0], 'inputs': [2, "i_inner", -3, fp_pf_factor, fp_cache_line, fp_pf_loop, "_neuron_index_2", "_neuron_index_2", stride_h, 1, 0], 'weight':[1, "i_inner", -5, inner_unroll_factor, "i_outer", 1, 1, 1]})
        else: # huge data can not fit into L1 cache
          # Wrong, needs fixing
          #print ("WARNING!!!!!  Disable prefetch as data does not fit L1 ")
          #conv_ens.prefetch(phase="forward", prefetch_dict_list={'value': [1, "_neuron_index_3", -3, outer_unroll_factor, "_neuron_index_2", 1, 1, 0], 'inputs': [2, "i_inner", -3, fp_pf_factor, fp_cache_line, fp_pf_loop, "_neuron_index_2", "_neuron_index_2", stride_h, 1, 0]})
          conv_ens.prefetch(phase="forward", prefetch_dict_list={'value': [1, "_neuron_index_3", -2, outer_unroll_factor, "_neuron_index_3", 1, outer_unroll_factor, 0], 'inputs': [3, "i_inner", -2, outer_unroll_factor, "k", 1, stride_w * outer_unroll_factor, 2, 0]})


        #backward
        conv_ens.unroll(phase="backward", loop_var="_neuron_index_3", factor=outer_unroll_factor)
        conv_ens.unroll(phase="backward", loop_var="_neuron_index_1_inner", factor=inner_unroll_factor)
        
        #conv_ens.unroll(phase="update_internal", loop_var="_neuron_index_3", factor=outer_unroll_factor) 
        #conv_ens.unroll(phase="update_internal", loop_var="_neuron_index_1_inner", factor=inner_unroll_factor)

        '''
        if data_needed_by_each_ifh < l1_size:
          if outer_unroll_factor == output_width:
            conv_ens.prefetch(phase="backward", prefetch_dict_list={'grad_inputs': [1, "k", -3, outer_unroll_factor, "_neuron_index_2", 1, 1, 0]})
          else:
            conv_ens.prefetch(phase="backward", prefetch_dict_list={'grad_inputs': [1, "k", -2, outer_unroll_factor, "_neuron_index_3", 1, outer_unroll_factor, 0]})
        else: # huge data can not fit into L1 cache
          # Wrong, needs fixing
          #print ("WARNING!!!!!  Disable prefetch as data does not fit L1 ")
          conv_ens.prefetch(phase="backward", prefetch_dict_list={'value': [1, "_neuron_index_3", -2, outer_unroll_factor, "_neuron_index_3", 1, outer_unroll_factor, 0], 'inputs': [3, "i_inner", -2, outer_unroll_factor, "k", 1, stride_w * outer_unroll_factor, 0]})
        '''
        wu_inner_unroll_factor = 8
        while 16 % wu_inner_unroll_factor != 0:
          wu_inner_unroll_factor -= 1
        if wu_inner_unroll_factor > 1:
          conv_ens.unroll(phase="update_internal", loop_var="_neuron_index_1_inner", factor=wu_inner_unroll_factor)
        wu_outer_unroll_factor = 32 - wu_inner_unroll_factor
        while output_width % wu_outer_unroll_factor != 0:
          wu_outer_unroll_factor -= 1
        if wu_outer_unroll_factor>1:
          conv_ens.unroll(phase="update_internal", loop_var="_neuron_index_3", factor=wu_outer_unroll_factor)
      else: #AVX 2 code
        outer_unroll_factor = 16
        while output_width % outer_unroll_factor != 0:
          outer_unroll_factor -= 1
        conv_ens.unroll(phase="forward", loop_var="_neuron_index_3", factor=outer_unroll_factor)
        #backward  
        conv_ens.unroll(phase="backward", loop_var="_neuron_index_3", factor=outer_unroll_factor)
        conv_ens.unroll(phase="backward", loop_var="_neuron_index_1_inner", factor=inner_unroll_factor)
        #update-internal
        conv_ens.unroll(phase="update_internal", loop_var="_neuron_index_3", factor=outer_unroll_factor)
        conv_ens.unroll(phase="update_internal", loop_var="_neuron_index_1_inner", factor=inner_unroll_factor)
         
    # End Optimizations

    return conv_ens
