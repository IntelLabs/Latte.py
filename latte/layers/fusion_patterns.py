import numpy as np
from ..neuron import WeightedNeuron, BiasNeuron
from ..ensemble import Ensemble, EnsembleGroup
import itertools
import latte.core
import math

def gcd(a, b):
  while (a != 0):
     c = a 
     a = b%a  
     b = c
  return b
def gcd3 (a, b, c):
  return gcd2(a, gcd(b, c))

def gcd4 (a, b, c, d):
  return gcd2(a, gcd(b, gcd(c, d)))

class FusionRecipe:
    def pattern_cbr(self, net, phase, conv_bias_ens, relu_ens):
      if isinstance(conv_bias_ens, EnsembleGroup): 
        bias_ens = conv_bias_ens.ensembles[-1]
        conv_ens = conv_bias_ens.ensembles[-2]
      else:
        conv_ens = conv_bias_ens

      IFM_BLOCK_THRESHOLD = 4
      HEIGHT_THRESHOLD = 14
      WIDTH_THRESHOLD = 14

      if phase != "forward":
        return

      #print (net.connections_map[bias_ens][0].source)
      #First try to fuse CBR
      #if bias_ens is not None and net.connections_map[bias_ens][0].source.shape[0] < IFM_BLOCK_THRESHOLD and conv_ens.shape[1]  > HEIGHT_THRESHOLD and conv_ens.shape[2] > WIDTH_THRESHOLD:
      if bias_ens is not None and conv_ens.shape[1]  > HEIGHT_THRESHOLD and conv_ens.shape[2] > WIDTH_THRESHOLD:

        conv_ens.reset_prefetch(phase="forward")
        #bring ifm inside
        conv_ens.swap_loops(phase="forward", loop_vars=("i_outer", "_neuron_index_2"))
        conv_ens.swap_loops(phase="forward", loop_vars=("i_outer", "_neuron_index_3"))
        
        #perform register blocking on h and w
        #determine h_unroll_factor, w_unroll_factor
        h_unroll_factor = 7
        while conv_ens.shape[1] % h_unroll_factor != 0:
          h_unroll_factor -= 1
        w_unroll_factor = int(28/h_unroll_factor)
        while conv_ens.shape[2] % w_unroll_factor != 0:
          w_unroll_factor -= 1

        conv_ens.unroll(phase="forward", loop_var="_neuron_index_2", factor=h_unroll_factor, unroll_type= 1)
        conv_ens.unroll(phase="forward", loop_var="_neuron_index_3", factor=w_unroll_factor, unroll_type=1)
        relu_ens.unroll(phase="forward", loop_var="_neuron_index_2", factor=h_unroll_factor, unroll_type= 1)
        relu_ens.unroll(phase="forward", loop_var="_neuron_index_3", factor=w_unroll_factor, unroll_type=1)
        bias_ens.unroll(phase="forward", loop_var="_neuron_index_2", factor=h_unroll_factor, unroll_type = 1)
        bias_ens.unroll(phase="forward", loop_var="_neuron_index_3", factor=w_unroll_factor, unroll_type =1)
      else: #try fusing B and R
        h_unroll_factor = 2
        w_unroll_factor = 2
        if bias_ens.shape[1] % h_unroll_factor == 0 and bias_ens.shape[2] % w_unroll_factor == 0 and relu_ens.shape[1] % h_unroll_factor == 0 and relu_ens.shape[2] % w_unroll_factor == 0:
          relu_ens.unroll(phase="forward", loop_var="_neuron_index_2", factor=h_unroll_factor, unroll_type= 1)
          relu_ens.unroll(phase="forward", loop_var="_neuron_index_3", factor=w_unroll_factor, unroll_type=1)
          bias_ens.unroll(phase="forward", loop_var="_neuron_index_2", factor=h_unroll_factor, unroll_type = 1)
          bias_ens.unroll(phase="forward", loop_var="_neuron_index_3", factor=w_unroll_factor, unroll_type =1)
