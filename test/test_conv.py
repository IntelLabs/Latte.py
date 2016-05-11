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
                            p = min(max(p, 0), in_height - 1)
                            for j, q in enumerate(range(in_x, out_x)):
                                q = min(max(q, 0), in_width - 1)
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
                            p = min(max(p, 0), in_height - 1)
                            for j, q in enumerate(range(in_x, out_x)):
                                q = min(max(q, 0), in_width - 1)
                                weights_grad[o, c, i , j] += top_grad[n, o, y, x] * _input[n, c, p, q]
                                bot_grad[n, c, p, q] += weights[o, c, i, j] * top_grad[n, o, y, x]
    return bot_grad, weights_grad, bias_grad

def check_equal(actual, expected, atol=1e-6):
    assert np.allclose(actual, expected, atol=atol)

def test_forward_backward():
    net = Net(8)
    channels, height, width = 16, 16, 16
    pad = 1
    data, data_value = MemoryDataLayer(net, (channels, height, width))
    conv1, conv1bias = ConvLayer(net, data, num_filters=16, kernel=3, stride=1, pad=pad)
    conv2, conv2bias = ConvLayer(net, conv1, num_filters=16, kernel=3, stride=1, pad=pad)

    data_value[:, :, :] = np.random.rand(8, channels, height, width)

    net.compile()

    weights = net.buffers[conv1.name + "weights"]
    bias    = net.buffers[conv1bias.name + "bias"]
    np.copyto(bias, np.random.rand(*bias.shape))
    weights_converted = util.convert_6d_4d(weights)

    net.forward()

    expected = reference_conv_forward(data_value, weights_converted, bias,
            pad, 1)

    actual  = net.buffers[conv1.name + "value"]
    actual_converted = util.convert_5d_4d(actual)
    for row1, row2 in zip(actual_converted, expected):
        for x, y in zip(row1, row2):
            count = 0
            for i, j in zip(x, y):
                if not np.allclose(i, j, atol=1e-4):
                    print("==========================")
                    print("count = ", count)
                    print(i)
                    print("--------------------------")
                    print(j)
                    print("==========================")
                    exit(1)
                count += 1
    check_equal(actual_converted, expected, 1e-4)

    top_grad = net.buffers[conv2.name + "grad"]
    np.copyto(top_grad, np.random.rand(*top_grad.shape))
    top_grad_converted = util.convert_5d_4d(top_grad)

    weights = net.buffers[conv2.name + "weights"]
    weights_converted = util.convert_6d_4d(weights)
    net.backward()

    expected_bot_grad, expected_weights_grad, expected_bias_grad = \
        reference_conv_backward(top_grad_converted, actual_converted,
                weights_converted, pad, 1)

    bot_grad = net.buffers[conv1.name + "grad"]
    actual_converted = util.convert_5d_4d(bot_grad)
    check_equal(actual_converted, expected_bot_grad)

    weights_grad = net.buffers[conv2.name + "grad_weights"]
    weights_converted = util.convert_6d_4d(weights_grad)
    check_equal(weights_converted, expected_weights_grad, 1e-3)

    bias_grad = net.buffers[conv2bias.name + "grad_bias"]
    check_equal(bias_grad, expected_bias_grad)
