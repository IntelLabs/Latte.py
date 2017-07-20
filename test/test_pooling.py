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

def reference_pooling_forward(_input, kernel, pad, stride):
    stride_h, stride_w = stride, stride
    pad_h, pad_w = pad, pad
    kernel_h, kernel_w = kernel, kernel
    batch_size, in_channels, in_height, in_width = _input.shape
    output_width = ((in_width - kernel_w + 2 * pad_w) // stride_w) + 1
    output_height = ((in_height - kernel_h + 2 * pad_h) // stride_h) + 1
    output = np.zeros((batch_size, in_channels, output_height, output_width), dtype=np.float32)
    output_mask = np.zeros((batch_size, in_channels, output_height, output_width, 2), dtype=np.int32)
    for n in range(batch_size):
        for o in range(in_channels):
            for y in range(output_height):
                for x in range(output_width):
                    in_y = y*stride_h-pad_h
                    in_x = x*stride_w-pad_w
                    out_y = in_y+kernel_h
                    out_x = in_x+kernel_w
                    maxval = float('-inf')
                    idx = ()
                    for i, p in enumerate(range(in_y, out_y)):
                        p = min(max(p, 0), in_height-1)
                        for j, q in enumerate(range(in_x, out_x)):
                            q = min(max(q, 0), in_width-1)
                            curr = _input[n, o, p, q]
                            if curr > maxval:
                                idx = (i, j)
                                maxval = curr
                    output[n, o, y, x] = maxval
                    output_mask[n, o, y, x, :] = idx
    return output, output_mask

def reference_pooling_backward(top_grad, _input, mask, stride=2, kernel=2, pad=0):
    stride_h, stride_w = stride, stride
    pad_h, pad_w = pad, pad
    batch_size, in_channels, in_height, in_width = _input.shape
    _, output_channels, output_height, output_width = top_grad.shape
    bot_grad = np.zeros_like(_input)
    for n in range(batch_size):
        for o in range(output_channels):
            for y in range(output_height):
                for x in range(output_width):
                    i, j = mask[n, o, y, x]   
                    in_y = min(max((y*stride_h-pad_h)+i, 0), in_height-1)
                    in_x = min(max((x*stride_w-pad_w)+j, 0), in_width-1)
                    bot_grad[n,o,in_y,in_x] += top_grad[n, o, y, x]
    return bot_grad

def check_equal(actual, expected, atol=1e-6):
    assert np.allclose(actual, expected, atol=atol)

def check_forward_backward(batch_size=8, input_shape=(16,16,16), pad=0, kernel=2, stride=2):
    net = Net(batch_size)
    net.force_backward = True
    channels, height, width = input_shape
    data = MemoryDataLayer(net, (channels, height, width))
    pool1 = MaxPoolingLayer(net, data, kernel=kernel, stride=stride, pad=pad)

    net.compile()
 
    data_value = np.random.rand(batch_size, channels, height, width)
    data.set_value(data_value)
    
    net.forward()

    expected, expected_mask = reference_pooling_forward(data_value, kernel, pad, stride)

    actual  = pool1.get_value()
    actual_mask_j  = pool1.get_mask_j()
    actual_mask_k  = pool1.get_mask_k()

    check_equal(actual, expected)
    check_equal(actual_mask_j, expected_mask[:, :, :, :, 0])
    check_equal(actual_mask_k, expected_mask[:, :, :, :, 1])

    top_grad = pool1.get_grad()
    top_grad = np.random.rand(*top_grad.shape)
    pool1.set_grad(top_grad)

    net.backward()

    expected_bot_grad = \
        reference_pooling_backward(top_grad, data_value, expected_mask, stride=stride, kernel=kernel, pad=pad)

    bot_grad = pool1.get_grad_inputs()

    check_equal(bot_grad, expected_bot_grad)


def test_forward_backward():
    check_forward_backward()
    
def test_padding():
    check_forward_backward(batch_size=1, input_shape=(1,16,16), pad=1)
 
def test_stride_one():
    check_forward_backward(kernel=2,stride=1)

def test_kernel_pad():
    check_forward_backward(pad=1,kernel=3,stride=1)

def test_kernel_pad_large():
    check_forward_backward(batch_size=1, input_shape=(1,40,40),pad=1,kernel=3,stride=1)


