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

def reference_softmax_forward(_input, label):
    output = np.zeros_like(_input)
    loss = 0.0
    for n in range(_input.shape[0]):
        x = _input[n]
        e_x = np.exp(x - np.max(x))
        output[n] = e_x / e_x.sum()
        loss -= np.log(max(output[n, int(label[n, 0])], np.finfo(np.float32).min))
    return output, loss / _input.shape[0]

def reference_softmax_backward(prob, label):
    bot_grad = np.zeros_like(prob)
    np.copyto(bot_grad, prob)
    for n in range(prob.shape[0]):
        bot_grad[n, int(label[n, 0])] -= 1
    bot_grad /= np.prod(bot_grad.shape)
    return bot_grad


# def test_forward_backward():
#     net = Net(8)
#     net.force_backward = True
#     data = MemoryDataLayer(net, (1000, ))
#     fc1, fc1bias = FullyConnectedLayer(net, data, 1000, )
#     label = MemoryDataLayer(net, (1, ))
#     softmax = SoftmaxLossLayer(net, fc1bias, label)
#     
#     data_value = np.random.rand(8, 1000)
#     data.set_value(data_value)
#     
#     label_value = np.floor(np.random.rand(8, 1) * 1000)
#     label.set_value(label_value)
# 
#     net.compile()
#     net.forward()
#     bottom = net.buffers[fc1bias.name + "value"].reshape(8, 1000)
# 
#     expected, loss = reference_softmax_forward(bottom, label_value)
#     assert np.allclose(softmax.prob, expected)
#     assert np.allclose(net.loss, loss)
#     net.backward()
# 
#     expected_grad = reference_softmax_backward(expected, label_value)
#     bot_grad = net.buffers[fc1bias.name + "grad"].reshape(8, 1000)
#     assert np.allclose(bot_grad, expected_grad)
