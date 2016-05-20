import pytest
import numpy as np
from latte import *
import latte.util as util

def check_equal(actual, expected, atol=1e-6):
    assert np.allclose(actual, expected, atol=atol)

def test_forward_backward():
    net = Net(8)
    data = MemoryDataLayer(net, (24, ))
    fc1, fc1bias = FullyConnectedLayer(net, data, 24)
    fc2, fc2bias = FullyConnectedLayer(net, fc1bias, 24)

    data_value = np.random.rand(8, 24)
    data.set_value(data_value)

    net.compile()

    bias    = net.buffers[fc1bias.name + "bias"]
    np.copyto(bias, np.random.rand(*bias.shape))

    net.forward()

    weights = net.buffers[fc1.name + "weights"]
    weights_converted = util.convert_4d_2d(weights)
    actual  = net.buffers[fc1.name + "value"]
    actual = actual.reshape((actual.shape[0], actual.shape[1] * actual.shape[2]))
    expected = np.dot(data_value, weights_converted.transpose())

    bias = util.convert_3d_2d(bias)
    for n in range(8):
        expected[n, :] += bias.reshape((24,))

    check_equal(actual, expected, 1e-5)

    top_grad = net.buffers[fc2.name + "grad"]
    np.copyto(top_grad, np.random.rand(*top_grad.shape))
    top_grad = top_grad.reshape((top_grad.shape[0], top_grad.shape[1] * top_grad.shape[2]))

    net.backward()
    weights = net.buffers[fc2.name + "weights"]
    weights_converted = util.convert_4d_2d(weights)

    bot_grad = net.buffers[fc1.name + "grad"]
    expected_bot_grad = np.dot(top_grad, weights_converted)
    bot_grad = bot_grad.reshape((bot_grad.shape[0], bot_grad.shape[1] * bot_grad.shape[2]))
    check_equal(bot_grad, expected_bot_grad)

    weights_grad = np.sum(net.buffers[fc2.name + "grad_weights"], axis=0)
    weights_grad_converted = util.convert_4d_2d(weights_grad)
    expected_weights_grad = np.dot(top_grad.transpose(), actual)
    check_equal(weights_grad_converted, expected_weights_grad)

    bias_grad = np.sum(net.buffers[fc2bias.name + "grad_bias"], axis=0)
    bias_grad = util.convert_3d_2d(bias_grad)
    expected_bias_grad = np.sum(top_grad, 0).reshape(24, 1)
    check_equal(bias_grad, expected_bias_grad)
