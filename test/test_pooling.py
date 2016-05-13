import pytest
import numpy as np
from latte import *
import latte.util as util

def reference_pooling_forward(_input, pad, stride):
    maxval = -sys.maxsize - 1
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
                                curr = _input[n, c, p, q]
                                if curr > maxval:
                                    maxval = curr
                                    output[n, o, y, x] = curr
    return output

def reference_conv_backward(top_grad, _input, pad, stride):
    stride_h, stride_w = stride, stride
    pad_h, pad_w = pad, pad
    batch_size, in_channels, in_height, in_width = _input.shape
    output_channels, _, kernel_h, kernel_w = weights.shape
    _, output_channels, output_height, output_width = top_grad.shape
    bot_grad = np.zeros_like(_input)
    for n in range(batch_size):
        for o in range(output_channels):
            for y in range(output_height):
                for x in range(output_width):
                    bias_grad[o] += top_grad[n, o, y, x]
                    in_y = max(y*stride_h - pad, 0)
                    in_x = max(x*stride_w - pad, 0)
                    out_y = min(in_y + kernel_h, in_height)
                    out_x = min(in_x + kernel_w, in_width)
                    for c in range(in_channels):
                        for i, p in enumerate(range(in_y, out_y)):
                            for j, q in enumerate(range(in_x, out_x)):
                                bot_grad[n, c, p, q] += top_grad[n, o, y, x]
    return bot_grad

def check_equal(actual, expected, atol=1e-6):
    assert np.allclose(actual, expected, atol=atol)

def test_forward_backward():
    net = Net(8)
    channels, height, width = 16, 16, 16
    pad = 1
    data, data_value = MemoryDataLayer(net, (channels, height, width))
    pool1 = MaxPoolingLayer(net, data, num_filters=16, kernel=3, stride=1, pad=pad)
    
    data_value[:, :, :] = np.random.rand(8, channels, height, width)

    net.compile()

    net.forward()

    expected = reference_conv_forward(data_value, pad, 1)

    actual  = net.buffers[pool1.name + "value"]
    actual_converted = util.convert_5d_4d(actual)
    check_equal(actual_converted, expected, 1e-5)

    top_grad = net.buffers[pool1.name + "grad"]
    np.copyto(top_grad, np.random.rand(*top_grad.shape))
    top_grad_converted = util.convert_5d_4d(top_grad)

    net.backward()

    expected_bot_grad = \
        reference_conv_backward(top_grad_converted, actual_converted, pad, 1)

    bot_grad = net.buffers[pool1.name + "grad"]
    actual_converted = util.convert_5d_4d(bot_grad)
    check_equal(actual_converted, expected_bot_grad)
