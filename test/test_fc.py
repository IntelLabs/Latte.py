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

def test_forward_backward():
    net = Net(8)
    data = MemoryDataLayer(net, (32, ))
    fc1 = FullyConnectedLayer(net, data, 32)
    #fc2 = FullyConnectedLayer(net, fc1, 32)
    net.force_backward=True
    net.compile()

    data_value = np.random.rand(8, 32)
    data.set_value(data_value)

    bias = fc1.get_bias()
    bias_value = np.random.rand(*bias.shape)
    fc1.set_bias(bias_value)
    net.forward()

    weights = fc1.get_weights()
    assert np.allclose(weights, weights)
    actual  = fc1.get_value()
    expected = np.dot(data_value, weights.transpose())

    for n in range(8):
        expected[n, :] += bias_value.reshape((32,))

    check_equal(actual, expected, 1e-4)

    top_grad = fc1.get_grad()
    top_grad = np.random.rand(*top_grad.shape)
    fc1.set_grad(top_grad)
    grad_weights = fc1.get_grad_weights() 
    fc1.set_grad_weights(np.zeros(grad_weights.shape))



    net.backward()
    weights = fc1.get_weights()

    bot_grad = data.get_grad()
    expected_bot_grad = np.dot(top_grad, weights)
    check_equal(bot_grad, expected_bot_grad, atol=1e-3)

    # weights_grad = np.sum(fc2.get_grad_weights(), axis=0)
    weights_grad = fc1.get_grad_weights()
    expected_weights_grad = np.dot(top_grad.transpose(), data_value)
    print(weights_grad)
    print(expected_weights_grad)
    check_equal(weights_grad, expected_weights_grad, atol=1e-4)

    bias_grad = fc1.get_grad_bias()
    expected_bias_grad = np.sum(top_grad, 0).reshape(32, 1)
    check_equal(bias_grad, expected_bias_grad, atol=1e-4)

def test_forward_backward_not_flat():
    batch_size = 32
    net = Net(batch_size)
    data = MemoryDataLayer(net, (16, 32, 32))
    conv1 = ConvLayer(net, data, num_filters=32, kernel=3, stride=1, pad=1)
    fc1 = FullyConnectedLayer(net, conv1, 128)
    fc2 = FullyConnectedLayer(net, fc1, 32)

    net.compile()

    data_value = np.random.rand(batch_size, 16, 32, 32)
    data.set_value(data_value)

    bias = fc1.get_bias()
    bias_value = np.random.rand(*bias.shape)
    fc1.set_bias(bias_value)
    net.forward()

    weights = fc1.get_weights()
    actual  = fc1.get_value()
    expected = np.dot(conv1.get_value().reshape(batch_size, 32 * 32 * 32), weights.reshape(128, 32* 32 * 32).transpose())

    for n in range(batch_size):
        expected[n, :] += bias_value.reshape((128,))

    check_equal(actual, expected, 1e-5)

    top_grad = fc2.get_grad()
    top_grad = np.random.rand(*top_grad.shape)
    fc2.set_grad(top_grad)

    net.backward()
    weights = fc2.get_weights()

    bot_grad = fc1.get_grad()
    expected_bot_grad = np.dot(top_grad, weights)
    check_equal(bot_grad, expected_bot_grad, atol=1e-4)

    # weights_grad = np.sum(fc2.get_grad_weights(), axis=0)
    weights_grad = fc2.get_grad_weights()
    expected_weights_grad = np.dot(top_grad.transpose(), actual)
    check_equal(weights_grad, expected_weights_grad, atol=1e-4)

    bias_grad = fc2.get_grad_bias()
    expected_bias_grad = np.sum(top_grad, 0).reshape(32, 1)
    check_equal(bias_grad, expected_bias_grad, atol=1e-4)



if __name__ == "__main__":
   test_forward_backward()
   test_forward_backward_not_flat() 

