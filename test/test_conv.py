import pytest
import numpy as np
from latte import *
import latte.util as util

def reference_conv_forward(_input, weights, bias, pad, stride):
    stride_h, stride_w = stride, stride
    pad_h, pad_w = pad, pad
    batch_size, in_channels, in_height, in_width = _input.shape
    output_channels, _, kernel_h, kernel_w = weights.shape
    output_width = ((in_width - kernel_w + 2 * pad_w) // stride_w) + 1
    output_height = ((in_height - kernel_h + 2 * pad_h) // stride_h) + 1
    output = np.zeros((batch_size, output_channels, output_height, output_width), dtype=np.float32)
    for n in range(batch_size):
        for o in range(output_channels):
            for y in range(output_height):
                for x in range(output_width):
                    in_y = y*stride_h - pad
                    in_x = x*stride_w - pad
                    out_y = in_y + kernel_h
                    out_x = in_x + kernel_w
                    for c in range(in_channels):
                        for i, p in enumerate(range(in_y, out_y)):
                            if p >= 0 and p < in_height:
                                for j, q in enumerate(range(in_x, out_x)):
                                    if q >= 0 and q < in_width:
                                        output[n, o, y, x] += weights[o, c, i, j] * _input[n, c, p, q]
                    output[n, o, y, x] += bias[o][0]
    return output

def reference_conv_backward(top_grad, _input, weights, pad, stride):
    stride_h, stride_w = stride, stride
    pad_h, pad_w = pad, pad
    batch_size, in_channels, in_height, in_width = _input.shape
    output_channels, _, kernel_h, kernel_w = weights.shape
    _, output_channels, output_height, output_width = top_grad.shape
    bot_grad = np.zeros_like(_input)
    bias_grad = np.zeros((output_channels, 1), dtype=np.float32)
    weights_grad = np.zeros_like(weights)
    for n in range(batch_size):
        for o in range(output_channels):
            for y in range(output_height):
                for x in range(output_width):
                    bias_grad[o] += top_grad[n, o, y, x]
                    in_y = y*stride_h - pad
                    in_x = x*stride_w - pad
                    out_y = in_y + kernel_h
                    out_x = in_x + kernel_w
                    for c in range(in_channels):
                        for i, p in enumerate(range(in_y, out_y)):
                            if p >= 0 and p < in_height:
                                for j, q in enumerate(range(in_x, out_x)):
                                    if q >= 0 and q < in_width:
                                        weights_grad[o, c, i , j] += top_grad[n, o, y, x] * _input[n, c, p, q]
                                        bot_grad[n, c, p, q] += weights[o, c, i, j] * top_grad[n, o, y, x]
    return bot_grad, weights_grad, bias_grad

def check_equal(actual, expected, atol=1e-6):
    assert np.allclose(actual, expected, atol=atol)

def test_forward_backward():
    net = Net(8)
    channels, height, width = 16, 16, 16
    pad = 1
    data = MemoryDataLayer(net, (channels, height, width))
    conv1, conv1bias = ConvLayer(net, data, num_filters=16, kernel=3, stride=1, pad=pad)
    conv2, conv2bias = ConvLayer(net, conv1bias, num_filters=16, kernel=3, stride=1, pad=pad)

    _input = np.random.rand(8, channels, height, width)
    data.set_value(_input)

    net.compile()

    weights = net.buffers[conv1.name + "weights"]
    bias    = net.buffers[conv1bias.name + "bias"]
    # np.copyto(bias, np.random.rand(*bias.shape))
    weights_converted = util.convert_6d_4d(weights)

    net.forward()

    bias = util.convert_3d_2d(bias)
    expected = reference_conv_forward(_input, weights_converted, bias,
            pad, 1)

    actual  = net.buffers[conv1.name + "value"]
    actual_converted = util.convert_5d_4d(actual)[:, :, pad:-pad, pad:-pad]
    check_equal(actual_converted, expected, 1e-5)

    top_grad = net.buffers[conv2.name + "grad"]
    np.copyto(top_grad, np.random.rand(*top_grad.shape))
    top_grad_converted = util.convert_5d_4d(top_grad)

    weights = net.buffers[conv2.name + "weights"]
    weights_converted = util.convert_6d_4d(weights)
    net.backward()

    expected_bot_grad, expected_weights_grad, expected_bias_grad = \
        reference_conv_backward(top_grad_converted, expected,
                weights_converted, pad, 1)

    bot_grad = net.buffers[conv1.name + "grad"]
    bot_grad = util.convert_5d_4d(bot_grad)[:, :, pad:-pad, pad:-pad]
    check_equal(bot_grad, expected_bot_grad)

    weights_grad = np.sum(net.buffers[conv2.name + "grad_weights"], axis=0)
    # weights_grad = net.buffers[conv2.name + "grad_weights"][0]
    weights_converted = util.convert_6d_4d(weights_grad)
    check_equal(weights_converted, expected_weights_grad, 1e-5)

    bias_grad = np.sum(net.buffers[conv2bias.name + "grad_bias"], axis=0)
    bias_grad = util.convert_3d_2d(bias_grad)
    # bias_grad = net.buffers[conv2bias.name + "grad_bias"][0]
    check_equal(bias_grad, expected_bias_grad)
