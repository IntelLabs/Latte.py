import pytest
import numpy as np
from latte import *
import latte.util as util

def reference_conv_forward(_input, weights, bias, pad, stride, dilation=1):
    stride_h, stride_w = stride, stride
    pad_h, pad_w = pad, pad
    batch_size, in_channels, in_height, in_width = _input.shape
    output_channels, _, kernel_h, kernel_w = weights.shape
    output_width = ((in_width - kernel_w * dilation + 2 * pad_w) // stride_w) + 1
    output_height = ((in_height - kernel_h * dilation + 2 * pad_h) // stride_h) + 1
    output = np.zeros((batch_size, output_channels, output_height, output_width), dtype=np.float32)
    for n in range(batch_size):
        for o in range(output_channels):
            for y in range(output_height):
                for x in range(output_width):
                    in_y = y*stride_h - pad
                    in_x = x*stride_w - pad
                    out_y = in_y + kernel_h * dilation
                    out_x = in_x + kernel_w * dilation
                    for c in range(in_channels):
                        for i, p in enumerate(range(in_y, out_y, dilation)):
                            if p >= 0 and p < in_height:
                                for j, q in enumerate(range(in_x, out_x, dilation)):
                                    if q >= 0 and q < in_width:
                                        output[n, o, y, x] += weights[o, c, i, j] * _input[n, c, p, q]
                    output[n, o, y, x] += bias[o][0]
    return output

def reference_conv_backward(top_grad, _input, weights, pad, stride, dilation=1):
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
                    out_y = in_y + kernel_h * dilation
                    out_x = in_x + kernel_w * dilation
                    for c in range(in_channels):
                        for i, p in enumerate(range(in_y, out_y, dilation)):
                            if p >= 0 and p < in_height:
                                for j, q in enumerate(range(in_x, out_x, dilation)):
                                    if q >= 0 and q < in_width:
                                        weights_grad[o, c, i , j] += top_grad[n, o, y, x] * _input[n, c, p, q]
                                        bot_grad[n, c, p, q] += weights[o, c, i, j] * top_grad[n, o, y, x]
    return bot_grad, weights_grad, bias_grad

def check_equal(actual, expected, atol=1e-6, rtol=1e-5):
    assert np.allclose(actual, expected, atol=atol, rtol=rtol)

def test_forward_backward():
    for dilation in range(1, 2):
        net = Net(3)
        channels, height, width = 16, 14, 14
        pad = 1
        data = MemoryDataLayer(net, (channels, height, width))
        conv1 = ConvLayer(net, data, num_filters=16, kernel=3, stride=1, pad=pad, dilation=dilation)
        conv2 = ConvLayer(net, conv1, num_filters=16, kernel=3, stride=1, pad=pad, dilation=dilation)

        _input = np.random.rand(3, channels, height, width)
        net.compile()
        data.set_value(_input)


        weights = conv1.get_weights()
        bias    = conv1.get_bias()
        bias    = np.random.rand(*bias.shape)
        conv1.set_bias(bias)

        conv1_expected = reference_conv_forward(_input, weights, bias,
                pad, 1, dilation)

        weights = conv2.get_weights()
        bias    = conv2.get_bias()
        bias    = np.random.rand(*bias.shape)
        conv2.set_bias(bias)

        expected = reference_conv_forward(conv1_expected, weights, bias,
                pad, 1, dilation)
        net.forward()

        check_equal(data.get_value(), _input)
        check_equal(conv1.get_value(), conv1_expected)
        actual  = conv2.get_value()
        check_equal(actual, expected)

        top_grad = conv2.get_grad()
        top_grad = np.random.rand(*top_grad.shape)
        conv2.set_grad(top_grad)

        weights = conv2.get_weights()
        net.backward()

        expected_bot_grad, expected_weights_grad, expected_bias_grad = \
            reference_conv_backward(top_grad, conv1_expected,
                    weights, pad, 1, dilation)

        bot_grad = conv1.get_grad()
        check_equal(bot_grad, expected_bot_grad)

        # weights_grad = np.sum(conv2.get_grad_weights(), axis=0)
        weights_grad = conv2.get_grad_weights()
        check_equal(weights_grad, expected_weights_grad, atol=1e-4)

        # bias_grad = np.sum(conv2.get_grad_bias(), axis=0)
        bias_grad = conv2.get_grad_bias()
        check_equal(bias_grad, expected_bias_grad)
