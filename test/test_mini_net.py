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
    batch_size = 16

    net = Net(batch_size)
    net.nowait = False
    net.force_backward = True

    channels, height, width = 8, 8, 8
    pad = 1
    data = MemoryDataLayer(net, (channels, height, width))
    conv1, conv1bias = ConvLayer(net, data, num_filters=32, kernel=3, stride=1, pad=pad)
    relu1 = ReLULayer(net, conv1bias)
    pool1 = MaxPoolingLayer(net, relu1, kernel=2, stride=2, pad=0)

    data_value = np.random.rand(batch_size, channels, height, width)
    data.set_value(data_value)

    net.compile()

    weights = net.buffers[conv1.name + "weights"]
    bias    = net.buffers[conv1bias.name + "bias"]
    np.copyto(bias, np.random.rand(*bias.shape))

    weights_converted = util.convert_6d_4d(weights)

    net.forward()

    bias = util.convert_3d_2d(bias)
    expected = reference_conv_forward(data_value, weights_converted, bias,
            pad, 1)

    actual  = net.buffers[conv1.name + "value"]
    actual_converted = util.convert_5d_4d(actual)

    expected = (expected > 0.0) * expected

    check_equal(actual_converted, expected, 1e-5)

    expected_pooling_output, expected_mask = reference_pooling_forward(expected, 2, 0, 2)

    actual  = net.buffers[pool1.name + "value"]
    actual_converted = util.convert_5d_4d(actual)
    check_equal(actual_converted, expected_pooling_output)

    actual_mask  = net.buffers[pool1.name + "mask"]
    actual_mask = util.convert_6d_5d(actual_mask)
    check_equal(actual_mask, expected_mask)

    top_grad = net.buffers[pool1.name + "grad"]
    np.copyto(top_grad, np.random.rand(*top_grad.shape))
    top_grad_converted = util.convert_5d_4d(top_grad)

    net.backward()

    expected_bot_grad = \
        reference_pooling_backward(top_grad_converted, expected, expected_mask, stride=2, kernel=2, pad=0)

    expected_bot_grad = (expected > 0.0) * expected_bot_grad

    bot_grad = net.buffers[conv1.name + "grad"]
    bot_grad = util.convert_5d_4d(bot_grad)
    check_equal(bot_grad, expected_bot_grad)

    weights = net.buffers[conv1.name + "weights"]
    weights_converted = util.convert_6d_4d(weights)

    expected_bot_grad, expected_weights_grad, expected_bias_grad = \
        reference_conv_backward(expected_bot_grad, data_value,
                weights_converted, pad, 1)

    bot_grad = net.buffers[conv1.name + "grad_inputs"]
    actual_converted = util.convert_5d_4d(bot_grad)[:, :, pad:-pad, pad:-pad]
    check_equal(actual_converted, expected_bot_grad, 1e-5)

    weights_grad = np.sum(net.buffers[conv1.name + "grad_weights"], axis=0)
    weights_converted = util.convert_6d_4d(weights_grad)
    check_equal(weights_converted, expected_weights_grad, 1e-5)

    bias_grad = np.sum(net.buffers[conv1bias.name + "grad_bias"], axis=0)
    bias_grad = util.convert_3d_2d(bias_grad)
    check_equal(bias_grad, expected_bias_grad)
