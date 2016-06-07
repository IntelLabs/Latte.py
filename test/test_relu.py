import pytest
import numpy as np
from latte import *
import latte.util as util

def check_equal(actual, expected, atol=1e-6):
    assert np.allclose(actual, expected, atol=atol)

def test_forward_backward():
    net = Net(8)
    net.force_backward = True
    channels, height, width = 16, 16, 16
    data = MemoryDataLayer(net, (channels, height, width))
    conv1 = ConvLayer(net, data, num_filters=8, kernel=3, stride=1, pad=1, dilation=1)
    relu1 = ReLULayer(net, conv1)
    
    data_value = np.random.rand(8, channels, height, width)
    data.set_value(data_value)

    net.compile()
    net.forward()
    
    expected = (conv1.get_value() > 0.0) * conv1.get_value()

    actual  = relu1.get_value()
    check_equal(actual, expected)

    top_grad = relu1.get_grad()
    top_grad_value = np.random.rand(*top_grad.shape)
    relu1.set_grad(top_grad_value)

    net.backward()
    bot_grad = conv1.get_grad()
    
    expected_bot_grad = (conv1.get_value() > 0.0) * top_grad_value
    check_equal(bot_grad, expected_bot_grad)
