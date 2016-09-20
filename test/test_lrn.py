#import ast
import pytest
import numpy as np
from latte import *
import latte.util as util
#from latte.ensemble import Ensemble, DataEnsemble, ActivationEnsemble, LossEnsemble, AccuracyEnsemble, EnsembleGroup
#import ctree
#from ctree.transformations import PyBasicConversions
#import sys

def reference_pooling_forward(_input,k,window, alpha, beta,scale):
    batch_size, in_channels, in_height, in_width = _input.shape
    output = np.zeros((batch_size, in_channels,in_height, in_width), dtype=np.float32)
    
    #print("in_channels is %d\n", in_channels)

    for n in range(batch_size):
         for y in range(in_height):
            for x in range(in_width):
                for o in range(in_channels):
                    sumval = 0.0
                    for m in range(min(window, in_channels -o)):   
                        sumval += _input[n,o+m,y,x]*_input[n,o+m,y,x]
                    sumval /= window
                    sumval *= alpha    
                    sumval += k
                    scale[n,o,y,x] = sumval    
                    output[n,o,y,x] = _input[n,o,y,x]/(sumval**beta)            
    return output

def reference_pooling_backward(top_grad,scale,_output,window,  _input,alpha,beta):
    batch_size, in_channels, in_height, in_width = _input.shape
    _, output_channels, output_height, output_width = top_grad.shape
    bot_grad = np.zeros((batch_size, output_channels+window-1,output_height, output_width), dtype=np.float32)

    #print("in_channels is %d\n", in_channels)
    #print("output_channels is %d\n", output_channels)

    for n in range(batch_size):
        for y in range(in_height): 
            for x in range(in_width):
                for o in range(in_channels):
                    sumval = 0.0
                    for m in range(min(window, in_channels - o)):
                        sumval += _output[n,o+m,y,x]
                    sumval /= window
                    sumval *= 2*alpha*beta*_input[n,o,y,x]
                    sumval /= scale[n,o,y,x]
                    bot_grad[n,o,y,x] += (1/(scale[n,o,y,x]**beta) - sumval)*top_grad[n,o,y,x]
                   
    return bot_grad

def check_equal(actual, expected, atol=1e-3):
    assert np.allclose(actual, expected, atol=atol)

def test_forward_backward():
    net = Net(8)
    net.force_backward = True
    channels, height, width = 16, 16, 16
    pad = 0
    

    data = MemoryDataLayer(net, (channels, height, width))
    #data_value = np.random.rand(8, channels, height, width)
    #data.set_value(data_value)
    #k = 1.0
    window = 5
    beta = 0.75 
    alpha =0.0001
    k = 1.0 

    pool1 = LRNLayer(net, data)
   
    net.compile()
 
    data_value = np.random.rand(8, channels, height, width)
    scale = np.zeros((8, channels,height,width),np.float32)
    data.set_value(data_value)
    #print(data_value)
    net.forward()
    #print(data_value)
    expected = reference_pooling_forward(data_value,k,window, alpha, beta,scale)

    actual  = pool1.get_value()
    #print(actual)

    #actual_mask_j  = pool1.get_mask_j()
    #actual_mask_k  = pool1.get_mask_k()
    check_equal(actual, expected)
    #check_equal(actual_mask_j, expected_mask[:, :, :, :, 0])
    #check_equal(actual_mask_k, expected_mask[:, :, :, :, 1])

    top_grad = pool1.get_grad()
    top_grad = np.random.rand(*top_grad.shape)
    pool1.set_grad(top_grad)

    net.backward()

    expected_bot_grad = \
        reference_pooling_backward(top_grad,scale,expected,window, data_value,alpha,beta)
    bot_grad = pool1.get_grad_inputs()
    #print(bot_grad.shape)
    check_equal(bot_grad, expected_bot_grad)

def main():
    test_forward_backward()

if __name__ == "__main__":
   main()

