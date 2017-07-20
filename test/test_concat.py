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
import ast
import pytest
import numpy as np
from latte import *
import latte.util as util
from latte.ensemble import Ensemble, DataEnsemble, ActivationEnsemble, LossEnsemble, AccuracyEnsemble, EnsembleGroup
import ctree
from ctree.transformations import PyBasicConversions
import sys


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




def reference_pooling_forward(_input):
    
    out_channels = 0
        
    for i in range(len(_input)):
        out_channels += _input[i].shape[1]
     
    #TODO need to add assertions for checking size dimensions
    batch_size,_, in_height,in_width = _input[1].shape
    output = np.zeros((batch_size, out_channels, in_height, in_width), dtype=np.float32)
    #for  i in range(len(_input))
    #count = 0
    for n in range(batch_size):
       count = 0  
       for i in range(len(_input)):
            if(i > 0):
                count += _input[i-1].shape[1]
            for o in range(_input[i].shape[1]):
                for y in range(in_height):
                    for x in range(in_width):
                        output[n, o+count, y, x] = _input[i][n, o, y,x]
    return output

def reference_pooling_backward(top_grad, _input, mask, stride=2, kernel=2, pad=0):
    
    batch_size,out_channels, in_height,in_width = top_grad.shape
    bot_grad = np.zeros((batch_size, out_channels, in_height, in_width), dtype=np.float32)

    
    for n in range(batch_size):
        for o in range(output_channels):
            for y in range(in_height):
                for x in range(in_width):
                    bot_grad[n,o,y,x] += top_grad[n, o, y, x]
    return bot_grad

def check_equal(actual, expected, atol=1e-6):
    assert np.allclose(actual, expected, atol=atol)

def test_forward_backward():
    net = Net(1)
    net.force_backward = True
    channels, height, width = 16, 16, 16
    data = MemoryDataLayer(net, (channels, height, width))
    pad = 1
    conv1 = ConvLayer(net, data, num_filters=16, kernel=3, stride=1, pad=pad)
    conv2 = ConvLayer(net, data, num_filters=32, kernel=3, stride=1, pad=pad)
    data_vector = []
    data_value = []
    data_vector.append(data)
    #data_vector.append(data2)
    #data_vector.append(data3)
    #data_vector.append(data4)


    concat1 = ConcatLayer(net, [conv1, conv2])

    net.compile()
 
        
    data_value.append(np.random.rand(1, channels, height, width))
    data_vector[0].set_value(data_value[0])

    weights = conv1.get_weights()
    bias    = conv1.get_bias()
    bias    = np.random.rand(*bias.shape)
    conv1.set_bias(bias)
 
    conv1_expected = reference_conv_forward(data_value[0], weights, bias)
 
    weights2 = conv2.get_weights()
    bias2    = conv2.get_bias()
    bias2    = np.random.rand(*bias2.shape)
    conv2.set_bias(bias2)
 
    conv2_expected = reference_conv_forward(data_value[0], weights2, bias2)

    

    net.forward()

    expected = reference_pooling_forward([conv1_expected, conv2_expected])

    actual  = concat1.get_value()
    print(actual.shape)
    #print(actual)
    #print(expected)    
    #actual_mask_j  = pool1.get_mask_j()
    #actual_mask_k  = pool1.get_mask_k()
    check_equal(actual, expected)
    #check_equal(actual_mask_j, expected_mask[:, :, :, :, 0])
    #check_equal(actual_mask_k, expected_mask[:, :, :, :, 1])

    #top_grad = concat1.get_grad()
    #print(*top_grad.shape)
    #top_grad = np.random.rand(*top_grad.shape)
    #concat1.set_grad(top_grad)

    #net.backward()

    #expected_bot_grad = \
    #    reference_pooling_backward(top_grad)

    #bot_grad = concat1.get_grad_inputs()
    #check_equal(bot_grad, top_grad)

#def main():
#    test_forward_backward()
 
#if __name__ == "__main__":
#    main()
 
