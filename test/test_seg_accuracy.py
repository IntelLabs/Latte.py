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
import math

def check_equal(actual, expected, atol=1e-6):
    assert np.allclose(actual, expected, atol=atol)

def reference_seg_accuracy_forward(_input, label):
    accuracy = 0.0
    confusion_matrix = np.zeros((_input.shape[0], _input.shape[0]))
    for n in range(_input.shape[0]):
        actual = math.floor(label[n,0])
        pred = np.argmax(_input[n])
        confusion_matrix[actual][pred] += 1

    if np.sum(confusion_matrix) == 0:
        accuracy = 0.0
    else:
        accuracy = np.trace(confusion_matrix) / np.sum(confusion_matrix)
    return accuracy


# def test_forward_backward():
#     batch_size = 64
#     net = Net(batch_size)
#     net.force_backward = True
#     data = MemoryDataLayer(net, (1000, ))
#     fc1, fc1bias = FullyConnectedLayer(net, data, 8)
#     label = MemoryDataLayer(net, (1000, ))
#     acc = SegAccuracyLayer(net, fc1bias, label)
#     
#     data_value = np.random.rand(batch_size, 1000) * 10 - 5
#     data.set_value(data_value)
#     
#     label_value = np.floor(np.random.rand(batch_size, 1) * 8)
#     label.set_value(label_value)
# 
#     net.compile()
#     net.test()
#     bottom = net.buffers[fc1bias.name + "value"].reshape(batch_size, 8)
# 
#     accuracy = reference_seg_accuracy_forward(bottom, label_value)
#     assert np.allclose(net.accuracy, accuracy)
