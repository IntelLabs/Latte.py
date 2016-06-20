import pytest
import numpy as np
from latte import *
import latte.util as util

def check_equal(actual, expected, atol=1e-6):
    assert np.allclose(actual, expected, atol=atol)

def test_forward_backward():
    net = Net(8)
    data = MemoryDataLayer(net, (24, ))
    fc1 = FullyConnectedLayer(net, data, 24)
    fc2 = FullyConnectedLayer(net, fc1, 16)

    net.compile()

    data_value = np.random.rand(8, 24)
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
        expected[n, :] += bias_value.reshape((24,))

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
    expected_bias_grad = np.sum(top_grad, 0).reshape(16, 1)
    check_equal(bias_grad, expected_bias_grad, atol=1e-4)
