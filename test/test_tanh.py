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

def check_equal(actual, expected, atol=1e-6):
    assert np.allclose(actual, expected, atol=atol)


def reference_conv_forward(_input, weights, bias, pad, stride, dilation=1):
    stride_h, stride_w = stride, stride
    pad_h, pad_w = pad, pad
    batch_size, in_channels, in_height, in_width = _input.shape
    output_channels, _, kernel_h, kernel_w = weights.shape

    kernel_h_eff = kernel_h + (kernel_h - 1) * (dilation - 1)
    kernel_w_eff = kernel_w + (kernel_w - 1) * (dilation - 1)

    output_width = ((in_width + 2 * pad_w - kernel_w_eff) // stride_w) + 1
    output_height = ((in_height + 2 * pad_h - kernel_h_eff) // stride_h) + 1

    output = np.zeros((batch_size, output_channels, output_height, output_width), dtype=np.float32)
    for n in range(batch_size):
        for o in range(output_channels):
            for y in range(output_height):
                for x in range(output_width):
                    in_y = y*stride_h - pad
                    in_x = x*stride_w - pad
                    out_y = in_y + kernel_h_eff
                    out_x = in_x + kernel_w_eff
                    for c in range(in_channels):
                        for i, p in enumerate(range(in_y, out_y, dilation)):
                            if p >= 0 and p < in_height:
                                for j, q in enumerate(range(in_x, out_x, dilation)):
                                    if q >= 0 and q < in_width:
                                        output[n, o, y, x] += weights[o, c, i, j] * _input[n, c, p, q]
                    output[n, o, y, x] += bias[o][0]
    return output


def test_forward_backward():
    net = Net(8)
    channels, height, width = 16, 16, 16
    data = MemoryDataLayer(net, (channels, height, width))
    conv1 = ConvLayer(net, data, num_filters=8, kernel=3, stride=1, pad=1, dilation=1)
    tanh1 = TanhLayer(net, conv1)

    net.compile()
    
    data_value = np.random.uniform(0, 0.01, (8, channels, height, width))
    data.set_value(data_value)

    weights = conv1.get_weights()
    bias    = conv1.get_bias()

    net.forward()

    expected_conv = reference_conv_forward(data_value, weights, bias, 1, 1, 1)
    expected = np.tanh(expected_conv) 


    actual  = tanh1.get_value()
    check_equal(conv1.get_value(), expected_conv, 1e-4)
    check_equal(actual, expected, 1e-4)
    
    top_grad = tanh1.get_grad()
    top_grad_value = np.random.rand(*top_grad.shape)
    tanh1.set_grad(top_grad_value)

    net.backward()
    bot_grad = conv1.get_grad()
    
    expected_bot_grad = (expected * (1.0 - expected)) * top_grad_value
    check_equal(bot_grad, expected_bot_grad)

def main():
    test_forward_backward()

if __name__ == "__main__":
   main()

