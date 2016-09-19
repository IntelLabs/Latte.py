import numpy as np
from ..neuron import Neuron
from ..ensemble import Ensemble, ConcatEnsemble
import itertools
import latte

class ConcatNeuron(Neuron):
    batch_fields     = Neuron.batch_fields# + ["inputs1", "grad_inputs1"]
    #zero_init_fields = Neuron.zero_init_fields + ["mask_j", "mask_k"]
    #["value", "grad", "inputs0", "grad_inputs0"]

    def __init__(self, n):
        super().__init__()
        
        
        self.inputs = []
        self.grad_inputs = []
        #batch_fields  = ["value", "grad", "inputs", "grad_inputs"]


        #for i in range(1,n):
        #    batch_fields += str("inputs" + str(i))
        #    batch_fields += str("grad_inputs" + str(i))
         
        #self.values = values    
        #self.mask_j = 0
        #self.mask_k = 0

    def forward(self):
        #FOR(INT 
        self.value = self.inputs[0,0,0]
        #self.value = self.inputs1[0,0,0]          

    def backward(self):
        self.grad_inputs[0,0,0] = self.grad
        #self.grad_inputs1[0,0,0] = self.grad 
    def update_internal(self):
        pass

def ConcatLayer(net, input_ensemble):
    
    assert len(input_ensemble) > 1
    total_channels, width,height  = input_ensemble[0].shape
    #values = []
    #value += 0.0
    max_channels = total_channels        
    #print(input_ensemble[0].shape)
    #print(input_ensemble[1].shape)
    # print(input_ensemble[2].shape)
    #print(input_ensemble[3].shape)


    for i in range(1, len(input_ensemble)):
        channels, width1,height1  = input_ensemble[i].shape
        print(input_ensemble[i].shape)
        assert input_ensemble[i].ndim == 3 and \
                width == width1 and \
                height == height1
        total_channels += channels#input_ensemble[i].shape[0]
        max_channels = max(channels, max_channels)
        #values += 0.0


    #for i in range(len(input_ensemble)):
    #    if input_ensemble[i].shape[0] < max_channels:
    #        input_ensemble[i].set_padding((0,max_channels-input_ensemble[i].shape[0]),(0,0), (0,0))

    #assert input_ensemble.ndim == 3, "PoolingLayer only supports 3-d input"

    #print(total_channels)

    shape = (total_channels,input_ensemble[0].shape[1], input_ensemble[0].shape[2])
    neurons = np.empty(shape, dtype='object')
    neurons[:] = ConcatNeuron(len(input_ensemble))

    concat_ens = ConcatEnsemble(neurons)
    net.add_ensemble(concat_ens)


    #input_shape = input_ensemble.shape
    #def mapping(c, y, x):
    #   return range(c, c+1), range(y, y+1), range(x, x+1)

    d=[]


    def make_mapping(start, end):
        def mapping(c,y,x):
            #start = offset
            #end =  offset + input_ensemble[i].shape[0]   
            #c_end = -1 
            #y_end  = y
            #x_end = x
            #c_begin=c
            #c_end=c
            #if start <=  c  and c < end:
            #    c_begin = c -start
            #    c_end = c_begin+1
            #   #c_begin = c_begin2
            #    #c_end = c_end2
            #    #c_end += c - start + 1
            #    #c_end = c_begin+ 1 
            #    #y_end = y+1
            #    #x_end = x+1
            #else:
            #    c_begin =  c 
            #    c_end = c                   
            #    #return range(c_begin, c_begin + 1), range(y, y+1), range(x, x+1)
            #    #else:
            return range(c, c+1), range(y,y+1), range(x,x+1)

        #def mapping2(c,y,x):
        #    #start = offset
        #    #end =  offset + input_ensemble[i].shape[0]   
        #    #c_end = -1 
        #    #y_end  = y
        #    #x_end = x
        #    #c_begin=c
        #    #c_end=c
        #    #if start <=  c  and c < end:
        #    #    c_begin = c -start
        #    #    c_end = c_begin+1
        #    #    #c_begin = c_begin2
        #    #    #c_end = c_end2
        #    #    #c_end += c - start + 1
        #    #    #c_end = c_begin+ 1 
        #    #    #y_end = y+1
        #    #     #x_end = x+1
        #    #else:
        #    #    c_begin = c
        #    #    c_end = c    
        #    #    #return range(c_begin, c_begin + 1), range(y, y+1), range(x, x+1)
        #    # else:
        #    #   c_end = c 
        #    return range(c, c+1), range(y,y+1), range(x,x+1)
        #if start == 0:
        #    return  mapping1
        #else:
        return mapping   

    for i in range(len(input_ensemble)):
        
        offset = 0   
        if i > 0:
            offset += input_ensemble[i-1].shape[0]
        start = offset
        end =  offset + input_ensemble[i].shape[0]
        d.append(make_mapping(start,end))
   
    for i in range(len(input_ensemble)):
        net.add_connections(input_ensemble[i], concat_ens, d[i])

    if "value" in input_ensemble[0].tiling_info:
        tiled_dims = input_ensemble[0].tiling_info["value"]
        for dim, factor in tiled_dims:
           concat_ens.tile('inputs', dim=dim, factor=factor)
        #pooling_ens.parallelize(phase="forward", loop_var="_neuron_index_1_outer")
        #pooling_ens.parallelize(phase="backward", loop_var="_neuron_index_1_outer")
        concat_ens.tile('value', dim=0, factor=latte.config.SIMDWIDTH)
        #pooling_ens.tile('sum_value', dim=0, factor=latte.config.SIMDWIDTH)
        #pooling_ens.tile('alpha', dim=0, factor=latte.config.SIMDWIDTH)
        #pooling_ens.tile('beta', dim=0, factor=latte.config.SIMDWIDTH)
        #pooling_ens.tile('n', dim=0, factor=latte.config.SIMDWIDTH)
        #pooling_ens.tile('k', dim=0, factor=latte.config.SIMDWIDTH)
 
        #pooling_ens.tile('mask_k', dim=0, factor=latte.config.SIMDWIDTH)
    #else:
    #    pooling_ens.parallelize(phase="forward", loop_var="_neuron_index_1")
    #    pooling_ens.parallelize(phase="backward", loop_var="_neuron_index_1")
    #if "grad" in input_ensemble.tiling_info:
    #    tiled_dims = input_ensemble.tiling_info["grad"]
                                                                                                                                                             
    if "grad" in input_ensemble[0].tiling_info:
        tiled_dims = input_ensemble[0].tiling_info["grad"]
        for dim, factor in tiled_dims:
            concat_ens.tile('grad_inputs', dim=dim, factor=factor)
        concat_ens.tile('grad', dim=0, factor=latte.config.SIMDWIDTH)
    return concat_ens
