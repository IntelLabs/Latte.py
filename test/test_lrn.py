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
#import ast
import pytest
import numpy as np
from latte import *
import latte.util as util
#from latte.ensemble import Ensemble, DataEnsemble, ActivationEnsemble, LossEnsemble, AccuracyEnsemble, EnsembleGroup
#import ctree
#from ctree.transformations import PyBasicConversions
#import sys
def reference_conv_forward(_input, weights, bias, pad=1, stride=1, dilation=1):
    stride_h, stride_w = stride, stride
    pad_h, pad_w = pad, pad
    batch_size, in_channels, in_height, in_width = _input.shape
    output_channels, _, kernel_h, kernel_w = weights.shape
    output_width = ((in_width - kernel_w * dilation + 2 * pad_w) // stride_w) + 1
    output_height = ((in_height - kernel_h * dilation + 2 * pad_h) // stride_h) + 1
    output = np.zeros((batch_size, output_channels, output_height, output_width), dtype=np.float32)
    for n in range(batch_size):
        for o in range(output_channels):
            for y in range(output_height):
                for x in range(output_width):
                    in_y = y*stride_h - pad
                    in_x = x*stride_w - pad
                    out_y = in_y + kernel_h * dilation
                    out_x = in_x + kernel_w * dilation
                    for c in range(in_channels):
                        for i, p in enumerate(range(in_y, out_y, dilation)):
                            if p >= 0 and p < in_height:
                                for j, q in enumerate(range(in_x, out_x, dilation)):
                                    if q >= 0 and q < in_width:
                                        output[n, o, y, x] += weights[o, c, i, j] * _input[n, c, p, q]
                    output[n, o, y, x] += bias[o][0]
    return output
 
 



def reference_pooling_forward(_input,k,window, alpha, beta,scale):
    batch_size, in_channels, in_height, in_width = _input.shape
    output = np.zeros((batch_size, in_channels,in_height, in_width), dtype=np.float32)
    
    #print("in_channels is %d\n", in_channels)
    offset = (window - 1)//2

    for n in range(batch_size):
         for y in range(in_height):
            for x in range(in_width):
                for o in range(in_channels):
                    sumval = 0.0
                    #for m in range(min(window, in_channels -o)):   
                    for m in range (max(0,o-offset), min(o+offset+1, in_channels)):
                        sumval += _input[n,m,y,x]*_input[n,m,y,x]
                    sumval /= window
                    sumval *= alpha    
                    sumval += k
                    scale[n,o,y,x] = sumval    
                    output[n,o,y,x] = _input[n,o,y,x]/(sumval**beta)            
    return output

def reference_pooling_backward(top_grad,scale,_output,window,  _input,alpha,beta):
    batch_size, in_channels, in_height, in_width = _input.shape
    _, output_channels, output_height, output_width = top_grad.shape
    bot_grad = np.zeros((batch_size, output_channels+8,output_height, output_width), dtype=np.float32)

    #print("in_channels is %d\n", in_channels)
    #print("output_channels is %d\n", output_channels)
    offset = (window - 1)//2
    for n in range(batch_size):
        for y in range(in_height): 
            for x in range(in_width):
                for o in range(in_channels):
                    sumval = 0.0
                    for m in range (min(0,o-offset), max(o+offset, in_channels)):
                        sumval += _output[n,m,y,x]
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
    #pad = 0
    

    data = MemoryDataLayer(net, (channels, height, width))
    conv1 = ConvLayer(net, data, num_filters=16, kernel=3, stride=1, pad=1)
  

    #data_value = np.random.rand(8, channels, height, width)
    #data.set_value(data_value)
    #k = 1.0
    window = 5
    beta = 0.75 
    alpha =0.0001
    k = 1.0
    conv1.set_padding((8,8),(0,0), (0,0))
    pool1 = LRNLayer(net, conv1)
   
    net.compile()
 
    data_value = np.random.rand(8, channels, height, width)
    scale = np.zeros((8, channels,height,width),np.float32)
    data.set_value(data_value)
    #print(data_value)
    #conv1.set_padding((8,8),(0,0), (0,0)) 

    net.forward()
    #print(data_value)
    #weights = conv1.get_weights()
    #bias = conv1.get_bias()

    weights = conv1.get_weights()
    bias    = conv1.get_bias()
    #bias    = np.random.rand(*bias.shape)
    #conv1.set_bias(bias)
    #conv1.set_padding((8,8),(0,0), (0,0)) 
    #conv1_expected = reference_conv_forward(data_value[0], weights, bias)

  

    expected_0 = reference_conv_forward(data_value, weights, bias, 1, 1)
    check_equal(expected_0, conv1.get_value())

    expected = reference_pooling_forward(expected_0,k,window, alpha, beta,scale)

    actual  = pool1.get_value()
    #print(actual)

    #actual_mask_j  = pool1.get_mask_j()
    #actual_mask_k  = pool1.get_mask_k()
    check_equal(actual, expected)
    #check_equal(actual_mask_j, expected_mask[:, :, :, :, 0])
    #check_equal(actual_mask_k, expected_mask[:, :, :, :, 1])

    #top_grad = pool1.get_grad()
    #top_grad = np.random.rand(*top_grad.shape)
    #pool1.set_grad(top_grad)

    #net.backward()

    #expected_bot_grad = \
    #    reference_pooling_backward(top_grad,scale,expected,window, data_value,alpha,beta)
   #bot_grad = pool1.get_grad_inputs()
    #print(bot_grad.shape)
    #check_equal(bot_grad, expected_bot_grad)

#def main():
#    test_forward_backward()

#if __name__ == "__main__":
#   main()

