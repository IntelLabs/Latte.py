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
    fc2 = FullyConnectedLayer(net, fc1, 24)

    data_value = np.random.rand(8, 24)
    data.set_value(data_value)

    net.compile()

    bias = fc1.get_bias()
    bias_value = np.random.rand(*bias.shape)
    fc1.set_bias(bias_value)
    net.forward()

    weights = net.buffers[fc1.ensembles[0].name + "weights"]
    weights_converted = util.convert_4d_2d(weights)
    weights = fc1.get_weights()
    assert np.allclose(weights, weights_converted)
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
    check_equal(bot_grad, expected_bot_grad)

    weights_grad = np.sum(fc2.get_grad_weights(), axis=0)
    expected_weights_grad = np.dot(top_grad.transpose(), actual)
    check_equal(weights_grad, expected_weights_grad)

    bias_grad = np.sum(fc2.get_grad_bias(), axis=0)
    expected_bias_grad = np.sum(top_grad, 0).reshape(24, 1)
    check_equal(bias_grad, expected_bias_grad)
