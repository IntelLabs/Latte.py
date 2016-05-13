import pytest
import numpy as np
from latte import *
import latte.util as util

def check_equal(actual, expected, atol=1e-6):
    assert np.allclose(actual, expected, atol=atol)

def test_forward_backward():
    net = Net(8)
    channels, height, width = 16, 16, 16
    pad = 1
    data, data_value = MemoryDataLayer(net, (channels, height, width))
    relu1 = ReLULayer(net, data)
    
    data_value[:, :, :] = np.random.rand(8, channels, height, width)

    net.compile()
    net.forward()
    
    expected = (data_value > 0.0) * data_value

    actual  = net.buffers[relu1.name + "value"]
    check_equal(actual, expected)

    net.backward()
    
    top_grad = net.buffers[relu1.name + "grad"]
    expected_bot_grad = (data_value > 0.0) * top_grad
    check_equal(top_grad, expected_bot_grad)
