import pytest
import numpy as np
from latte import *
import latte.util as util

def check_equal(actual, expected, atol=1e-6):
    assert np.allclose(actual, expected, atol=atol)

def reference_accuracy_forward(_input, label):
    accuracy = 0.0
    for n in range(_input.shape[0]):
        if np.argmax(_input[n]) == label[n, 0]:
            accuracy += 1
    return accuracy / label.size


def test_forward_backward():
    batch_size = 64
    net = Net(batch_size)
    net.force_backward = True
    data = MemoryDataLayer(net, (1000, ))
    fc1, fc1bias = FullyConnectedLayer(net, data, 8)
    label = MemoryDataLayer(net, (1, ))
    softmax = AccuracyLayer(net, fc1bias, label)
    
    data_value = np.random.rand(batch_size, 1000) * 10 - 5
    data.set_value(data_value)
    
    label_value = np.floor(np.random.rand(batch_size, 1) * 8)
    label.set_value(label_value)

    net.compile()
    net.test()
    bottom = net.buffers[fc1bias.name + "value"].reshape(batch_size, 8)

    accuracy = reference_accuracy_forward(bottom, label_value)
    assert np.allclose(net.accuracy, accuracy)
