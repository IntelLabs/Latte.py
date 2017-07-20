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

def check_forward_backward(batch_size=8, input_shape=(16,16,16), pad=1, ratio=0.5):
    net = Net(batch_size)
    net.force_backward = True
    channels, height, width = input_shape
    data = MemoryDataLayer(net, (channels, height, width))
    drop1 = DropoutLayer(net, data, ratio)

    net.compile()
    
    data_value = np.random.rand(batch_size, channels, height, width)
    data.set_value(data_value)

    net.forward()    

    scale = 1./(1.-ratio)
    randvals = drop1.get_randval()
    randvals = randvals.reshape(randvals.shape)
    expected = (randvals > ratio) * (data_value * scale)

    actual  = drop1.get_value()
    check_equal(actual, expected)

    top_grad = drop1.get_grad()
    top_grad_value = np.random.rand(*top_grad.shape)
    drop1.set_grad(top_grad_value)

    net.backward()
    bot_grad = drop1.get_grad_inputs()

    expected_bot_grad = (randvals > ratio) * (top_grad_value * scale)
    check_equal(bot_grad, expected_bot_grad)

def test_ratio_0_5():
    check_forward_backward()

def test_ratio_0_3():
    check_forward_backward(ratio=0.3)

def test_ratio_0_7():
    check_forward_backward(ratio=0.7)
