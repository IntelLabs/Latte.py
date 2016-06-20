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
    pad = 1
    ratio = 0.5
    data = MemoryDataLayer(net, (channels, height, width))
    drop1 = DropoutLayer(net, data, ratio)

    net.compile()
    
    data_value = np.random.rand(8, channels, height, width)
    data.set_value(data_value)

    net.forward()    

    randvals = drop1.get_randval()
    randvals = randvals.reshape(randvals.shape[:-1])
    expected = (randvals > ratio) * data_value
    print(expected.shape)

    actual  = drop1.get_value()
    check_equal(actual, expected)

    top_grad = drop1.get_grad()
    top_grad_value = np.random.rand(*top_grad.shape)
    drop1.set_grad(top_grad_value)

    net.backward()
    bot_grad = drop1.get_grad_inputs()

    expected_bot_grad = (randvals > ratio) * top_grad_value
    check_equal(bot_grad, expected_bot_grad)
