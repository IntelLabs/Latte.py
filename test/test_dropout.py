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
