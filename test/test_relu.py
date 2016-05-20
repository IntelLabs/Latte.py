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
    data = MemoryDataLayer(net, (channels, height, width))
    relu1 = ReLULayer(net, data)
    
    data_value = np.random.rand(8, channels, height, width)
    data.set_value(data_value)

    net.compile()
    net.forward()
    
    expected = (data_value > 0.0) * data_value

    actual  = net.buffers[relu1.name + "value"]
    actual = util.convert_5d_4d(actual)
    check_equal(actual, expected)

    top_grad = net.buffers[relu1.name + "grad"]
    top_grad_value = np.random.rand(*top_grad.shape)
    np.copyto(top_grad, top_grad_value)
    top_grad_value = util.convert_5d_4d(top_grad_value)

    net.backward()
    bot_grad = net.buffers[relu1.name + "grad_inputs"]
    bot_grad = util.convert_5d_4d(bot_grad)
    
    expected_bot_grad = (data_value > 0.0) * top_grad_value
    check_equal(bot_grad, expected_bot_grad)
