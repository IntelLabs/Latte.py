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
import pytest
import numpy as np
from latte import *
import latte.util as util
import sys
import math

def reference_interpolation_forward(_input, pad, resize_factor):
    pad_h, pad_w = pad, pad
    batch_size, in_channels, in_height, in_width = _input.shape
    output_width = math.floor(in_width * resize_factor) - pad
    output_height = math.floor(in_height * resize_factor) - pad
    output = np.zeros((batch_size, in_channels, output_height, output_width), dtype=np.float32)
    for n in range(batch_size):
        for o in range(in_channels):
            for y in range(output_height):
                delta_r = (y/resize_factor) - math.floor(y/resize_factor)
                for x in range(output_width):
                    delta_c = (x/resize_factor) - math.floor(x/resize_factor)
                    in_y = min(max(math.floor(y/resize_factor) - pad, 0), in_height - 1)
                    in_x = min(max(math.floor(x/resize_factor) - pad, 0), in_width - 1)
                    
                    in_y_plus_1 = in_y+1
                    in_x_plus_1 = in_x+1

                    if in_y_plus_1 > in_height - 1:
                        in_y_plus_1 = in_y                    
                    if in_x_plus_1 > in_width - 1:
                        in_x_plus_1 = in_x                

                    output[n, o, y, x] = \
                        _input[n, o, in_y, in_x]               * (1-delta_r) * (1-delta_c) + \
                        _input[n, o, in_y_plus_1, in_x]        * (delta_r)   * (1-delta_c) + \
                        _input[n, o, in_y, in_x_plus_1]        * (1-delta_r) * (delta_c)   + \
                        _input[n, o, in_y_plus_1, in_x_plus_1] * (delta_r)   * (delta_c)

    return output


def reference_interpolation_backward(top_grad, _input, pad, resize_factor):
    pad_h, pad_w = pad, pad
    batch_size, in_channels, in_height, in_width = _input.shape
    _, output_channels, output_height, output_width = top_grad.shape
    bot_grad = np.zeros_like(_input)
    for n in range(batch_size):
        for o in range(output_channels):
            for y in range(output_height):
                delta_r = (y/resize_factor) - math.floor(y/resize_factor)
                for x in range(output_width):
                    delta_c = (x/resize_factor) - math.floor(x/resize_factor)

                    in_y = min(max(math.floor(y/resize_factor) - pad, 0), in_height - 1)
                    in_x = min(max(math.floor(x/resize_factor) - pad, 0), in_width - 1)
                    
                    in_y_plus_1 = in_y+1
                    in_x_plus_1 = in_x+1

                    if in_y_plus_1 > in_height - 1:
                        in_y_plus_1 = in_y                    
                    if in_x_plus_1 > in_width - 1:
                        in_x_plus_1 = in_x                

                    bot_grad[n,o,in_y,in_x] += (1-delta_r)*(1-delta_c)*top_grad[n, o, y, x]
                    bot_grad[n,o,in_y_plus_1,in_x] += (delta_r)*(1-delta_c)*top_grad[n, o, y, x]
                    bot_grad[n,o,in_y,in_x_plus_1] += (1-delta_r)*(delta_c)*top_grad[n, o, y, x]
                    bot_grad[n,o,in_y_plus_1,in_x_plus_1] += (delta_r)*(delta_c)*top_grad[n, o, y, x]
    return bot_grad


def check_equal(actual, expected, atol=1e-6):
    assert np.allclose(actual, expected, atol=atol)

def check_forward_backward(batch_size=1, input_shape=(16,16,16), pad=0, resize_factor=1.0):
    net = Net(batch_size)
    net.force_backward = True
    channels, height, width = input_shape
    
    data_value = np.random.randint(0,255, (batch_size, *input_shape)).astype(np.float32)

    data = MemoryDataLayer(net, (channels, height, width))
    interp1 = InterpolationLayer(net, data, pad=pad, resize_factor=resize_factor)
    
    net.compile()

    data.set_value(data_value)

    net.forward()
   
    expected = reference_interpolation_forward(data_value, pad, resize_factor)
    actual  = interp1.get_value()
    check_equal(actual, expected)

    top_grad = interp1.get_grad()
    top_grad = np.random.rand(*top_grad.shape)
    interp1.set_grad(top_grad)

    net.backward()

    expected_bot_grad = reference_interpolation_backward(top_grad,
            data_value, pad, resize_factor)

    bot_grad = interp1.get_grad_inputs()
    check_equal(bot_grad, expected_bot_grad)


def test_shrink_half():
    check_forward_backward(resize_factor=0.5)

def test_shrink_quarter():
    check_forward_backward(resize_factor=0.25)

def test_shrink():
    check_forward_backward(resize_factor=0.125)

def test_shrink_pad():
    check_forward_backward(pad=-1,resize_factor=0.125)

def test_enlarge():
    check_forward_backward(resize_factor=2.0)

def test_enlarge_pad():
    check_forward_backward(pad=-1,resize_factor=2.0)

