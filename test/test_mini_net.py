import pytest
import numpy as np
from latte import *
import latte.util as util
from .test_conv import reference_conv_forward, reference_conv_backward
from .test_pooling import reference_pooling_forward, reference_pooling_backward
import os

def check_equal(actual, expected, atol=1e-6):
    assert np.allclose(actual, expected, atol=atol)

def test_forward_backward():
    batch_size = 8

    net = Net(batch_size)
    net.nowait = False
    net.force_backward = True

    channels, height, width = 8, 8, 8
    pad = 1
    data = MemoryDataLayer(net, (channels, height, width))
    conv1 = ConvLayer(net, data, num_filters=32, kernel=3, stride=1, pad=pad)
    relu1 = ReLULayer(net, conv1)
    pool1 = MaxPoolingLayer(net, relu1, kernel=2, stride=2, pad=0)

    data_value = np.random.rand(batch_size, channels, height, width)
    data.set_value(data_value)

    net.compile()

    weights = conv1.get_weights()
    bias    = conv1.get_bias()
    bias    = np.random.rand(*bias.shape)
    conv1.set_bias(bias)

    net.forward()

    expected = reference_conv_forward(data_value, weights, bias, pad, 1)

    actual  = conv1.get_value()

    expected = (expected > 0.0) * expected

    check_equal(actual, expected, 1e-5)

    expected_pooling_output, expected_mask = reference_pooling_forward(expected, 2, 0, 2)

    actual  = pool1.get_value()
    check_equal(actual, expected_pooling_output)

    actual_mask  = pool1.get_mask()
    check_equal(actual_mask, expected_mask)

    top_grad = pool1.get_grad()
    top_grad = np.random.rand(*top_grad.shape)
    pool1.set_grad(top_grad)

    net.backward()

    expected_bot_grad = \
        reference_pooling_backward(top_grad, expected, expected_mask, stride=2, kernel=2, pad=0)

    expected_bot_grad = (expected > 0.0) * expected_bot_grad

    bot_grad = conv1.get_grad()
    check_equal(bot_grad, expected_bot_grad)

    weights = conv1.get_weights()

    expected_bot_grad, expected_weights_grad, expected_bias_grad = \
        reference_conv_backward(expected_bot_grad, data_value,
                weights, pad, 1)

    # skip data layer grad
    # bot_grad = conv1.get_grad_inputs()
    # check_equal(bot_grad, expected_bot_grad, 1e-5)

    weights_grad = np.sum(conv1.get_grad_weights(), axis=0)
    check_equal(weights_grad, expected_weights_grad, 1e-5)

    bias_grad = np.sum(conv1.get_grad_bias(), axis=0)
    check_equal(bias_grad, expected_bias_grad)
