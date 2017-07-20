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
from .test_conv import reference_conv_forward, reference_conv_backward
from .test_pooling import reference_pooling_forward, reference_pooling_backward
import os

def check_equal(actual, expected, atol=1e-6):
    assert np.allclose(actual, expected, atol=atol)

def test_forward_backward():
    batch_size = 8

    net = Net(batch_size)
    net.nowait = False

    channels, height, width = 8, 8, 8
    pad = 1
    data = MemoryDataLayer(net, (channels, height, width))
    conv1 = ConvLayer(net, data, num_filters=32, kernel=3, stride=1, pad=pad)
    relu1 = ReLULayer(net, conv1)
    pool1 = MaxPoolingLayer(net, relu1, kernel=2, stride=2, pad=0)

    net.compile()

    data_value = np.random.rand(batch_size, channels, height, width)
    data.set_value(data_value)

    weights = conv1.get_weights()
    bias    = conv1.get_bias()
    bias    = np.random.rand(*bias.shape)
    conv1.set_bias(bias)

    net.forward()

    expected = reference_conv_forward(data_value, weights, bias, pad, 1)

    actual  = conv1.get_value()

    expected = (expected > 0.0) * expected

    check_equal(actual, expected, 1e-4)

    expected_pooling_output, expected_mask = reference_pooling_forward(expected, 2, 0, 2)

    actual  = pool1.get_value()
    check_equal(actual, expected_pooling_output)

    actual_mask_j  = pool1.get_mask_j()
    actual_mask_k  = pool1.get_mask_k()
    check_equal(actual_mask_j, expected_mask[:,:,:,:,0])
    check_equal(actual_mask_k, expected_mask[:,:,:,:,1])

    top_grad = pool1.get_grad()
    top_grad = np.random.rand(*top_grad.shape)
    pool1.set_grad(top_grad)

    net.backward()

    expected_bot_grad = \
        reference_pooling_backward(top_grad, expected, expected_mask, stride=2, kernel=2, pad=0)

    expected_bot_grad = (expected > 0.0) * expected_bot_grad

    bot_grad = conv1.get_grad()
    check_equal(bot_grad, expected_bot_grad)

    weights = conv1.get_weights()

    expected_bot_grad, expected_weights_grad, expected_bias_grad = \
        reference_conv_backward(expected_bot_grad, data_value,
                weights, pad, 1)

    # skip data layer grad
    # bot_grad = conv1.get_grad_inputs()
    # check_equal(bot_grad, expected_bot_grad, 1e-5)

    # weights_grad = np.sum(conv1.get_grad_weights(), axis=0)
    weights_grad = conv1.get_grad_weights()
    check_equal(weights_grad, expected_weights_grad, 1e-5)

    # bias_grad = np.sum(conv1.get_grad_bias(), axis=0)
    bias_grad = conv1.get_grad_bias()
    check_equal(bias_grad, expected_bias_grad)
