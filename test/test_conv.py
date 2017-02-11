import pytest
import numpy as np
from latte import *
import latte.util as util

def reference_conv_forward(_input, weights, bias, pad, stride, dilation=1):
    stride_h, stride_w = stride, stride
    pad_h, pad_w = pad, pad
    batch_size, in_channels, in_height, in_width = _input.shape
    output_channels, _, kernel_h, kernel_w = weights.shape

    kernel_h_eff = kernel_h + (kernel_h - 1) * (dilation - 1)
    kernel_w_eff = kernel_w + (kernel_w - 1) * (dilation - 1)

    output_width = ((in_width + 2 * pad_w - kernel_w_eff) // stride_w) + 1
    output_height = ((in_height + 2 * pad_h - kernel_h_eff) // stride_h) + 1

    output = np.zeros((batch_size, output_channels, output_height, output_width), dtype=np.float32)
    for n in range(batch_size):
        for o in range(output_channels):
            for y in range(output_height):
                for x in range(output_width):
                    in_y = y*stride_h - pad 
                    in_x = x*stride_w - pad
                    out_y = in_y + kernel_h_eff
                    out_x = in_x + kernel_w_eff
                    for c in range(in_channels):
                        for i, p in enumerate(range(in_y, out_y, dilation)):
                            if p >= 0 and p < in_height:
                                for j, q in enumerate(range(in_x, out_x, dilation)):
                                    if q >= 0 and q < in_width:
                                        output[n, o, y, x] += weights[o, c, i, j] * _input[n, c, p, q]
                    output[n, o, y, x] += bias[o][0]
    return output


# Caffe-based implementation of forward, using im2col and matrix-multiply (dot product)
def reference_caffe_conv_forward(_input, weights, bias, pad, stride, dilation=1):
    stride_h, stride_w = stride, stride
    pad_h, pad_w = pad, pad
    batch_size, in_channels, in_height, in_width = _input.shape
    output_channels, _, kernel_h, kernel_w = weights.shape
    kernel_h_eff = kernel_h + (kernel_h-1) * (dilation - 1)
    kernel_w_eff = kernel_w + (kernel_w-1) * (dilation - 1)

    output_width = ((in_width + 2 * pad_w - kernel_w_eff) // stride_w) + 1
    output_height = ((in_height + 2 * pad_h - kernel_h_eff) // stride_h) + 1

    channels_col = in_channels * kernel_h * kernel_w
    col_buffer = np.zeros((batch_size, channels_col, output_height, output_width), dtype=np.float32)

    # perform im2col
    for n in range(batch_size):
        for c in range(channels_col):
            w_offset = (c % kernel_w) * dilation
            h_offset = ((c // kernel_w) % kernel_h) * dilation
            c_im = c // kernel_w // kernel_h
            for h in range(output_height):
                h_im = h * stride_h + h_offset - pad_h
                for w in range(output_width):
                    w_im = w * stride_w + w_offset - pad_w
                    if h_im >=0 and h_im < in_height and w_im >=0 and w_im < in_width:
                        col_buffer.flat[((n * channels_col + c) * output_height + h) * output_width + w] = _input.flat[((n * in_channels + c_im) * in_height + h_im) * in_width + w_im]
                    else:
                        col_buffer.flat[((n*channels_col+c)*output_height+h)*output_width+w] = 0


    col_buffer_reshaped = col_buffer.reshape(channels_col, output_height * output_width)
    # weights array is padded so only use unpadded array to perform dot product. Otherwise, dimensions will not match.
    weights_reshaped = weights[:,0:in_channels,:,:].reshape(output_channels, in_channels * kernel_h * kernel_w)

    # perform dot product
    output = np.dot(weights_reshaped, col_buffer_reshaped)
    output = output.reshape(batch_size, output_channels, in_height, in_width)

    for n in range(batch_size):
        for o in range(output_channels):
            for y in range(output_height):
                for x in range(output_width):
                    output[n,o,y,x] += bias[o][0]

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

    kernel_h_eff = kernel_h + (kernel_h - 1) * (dilation - 1)
    kernel_w_eff = kernel_w + (kernel_w - 1) * (dilation - 1)

    for n in range(batch_size):
        for o in range(output_channels):
            for y in range(output_height):
                for x in range(output_width):
                    bias_grad[o] += top_grad[n, o, y, x]
                    in_y = y*stride_h - pad
                    in_x = x*stride_w - pad
                    out_y = in_y + kernel_h_eff
                    out_x = in_x + kernel_w_eff
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

def check_forward_backward(batch_size=3, input_shape=(16, 14, 14), ofm1=16, ofm2=32, pad=0, kernel=3, stride=1, dilation=1):
    net = Net(batch_size)
    channels, height, width = input_shape
    data = MemoryDataLayer(net, (channels, height, width))
    conv1 = ConvLayer(net, data, num_filters=ofm1, kernel=kernel, stride=stride, pad=pad, dilation=dilation)
    conv2 = ConvLayer(net, conv1, num_filters=ofm2, kernel=kernel, stride=stride, pad=pad, dilation=dilation)

    _input = np.random.rand(batch_size, channels, height, width)
    net.compile()
    data.set_value(_input)

    weights = conv1.get_weights()
    weights = np.random.rand(*weights.shape)
    conv1.set_weights(weights)
    bias    = conv1.get_bias()
    bias    = np.random.rand(*bias.shape)
    conv1.set_bias(bias)

    conv1_expected = reference_conv_forward(_input, weights, bias,
            pad, stride, dilation)

    weights = conv2.get_weights()
    weights = np.random.rand(*weights.shape)
    conv2.set_weights(weights)
    bias    = conv2.get_bias()
    bias    = np.random.rand(*bias.shape)
    conv2.set_bias(bias)

    expected = reference_conv_forward(conv1_expected, weights, bias,
            pad, stride, dilation)
    
    net.forward()

    check_equal(data.get_value(), _input)
    check_equal(conv1.get_value(), conv1_expected)
    actual  = conv2.get_value()
    check_equal(actual, expected, 1e-4)

    top_grad = conv2.get_grad()
    top_grad = np.random.rand(*top_grad.shape)
    conv2.set_grad(top_grad)

    weights = conv2.get_weights()
    net.backward()

    expected_bot_grad, expected_weights_grad, expected_bias_grad = \
        reference_conv_backward(top_grad, conv1_expected,
                weights, pad, stride, dilation)

    bot_grad = conv1.get_grad()
    check_equal(bot_grad, expected_bot_grad, atol=1e-4)

    # weights_grad = np.sum(conv2.get_grad_weights(), axis=0)
    weights_grad = conv2.get_grad_weights()
    check_equal(weights_grad, expected_weights_grad, atol=1e-4)

    # bias_grad = np.sum(conv2.get_grad_bias(), axis=0)
    bias_grad = conv2.get_grad_bias()
    check_equal(bias_grad, expected_bias_grad)

def check_caffe_forward(batch_size=3, input_shape=(16, 14, 14), ofm1=16, ofm2=32, pad=0, kernel=3, stride=1, dilation=1):
    net = Net(batch_size)
    channels, height, width = input_shape
    data = MemoryDataLayer(net, (channels, height, width))
    conv1 = ConvLayer(net, data, num_filters=ofm1, kernel=kernel, stride=stride, pad=pad, dilation=dilation)
    conv2 = ConvLayer(net, conv1, num_filters=ofm2, kernel=kernel, stride=stride, pad=pad, dilation=dilation)

    _input = np.random.rand(batch_size, channels, height, width)
    net.compile()
    data.set_value(_input)


    weights = np.random.rand(ofm1, channels, kernel, kernel)
    conv1.set_weights(weights)
    bias    = conv1.get_bias()
    bias    = np.random.rand(*bias.shape)
    conv1.set_bias(bias)

    conv1_expected = reference_caffe_conv_forward(_input, conv1.get_weights(), bias,
            pad, stride, dilation)

    weights = np.random.rand(ofm2, channels, kernel, kernel)
    conv2.set_weights(weights)
    bias    = conv2.get_bias()
    bias    = np.random.rand(*bias.shape)
    conv2.set_bias(bias)

    expected = reference_caffe_conv_forward(conv1_expected, conv2.get_weights(), bias,
            pad, stride, dilation)

    net.forward()

    check_equal(data.get_value(), _input)
    check_equal(conv1.get_value(), conv1_expected)
    actual  = conv2.get_value()
    check_equal(actual, expected, 1e-4)


def test_padding():
    check_forward_backward(input_shape=(3, 14, 14), pad=1)
def test_medium():
    check_forward_backward(input_shape=(16, 14, 14), pad=1)

def test_dilation():
    check_forward_backward(input_shape=(16, 14, 14), pad=1, dilation=2)

def test_pad_ofm():
    check_forward_backward(input_shape=(16, 14, 14), ofm1=19, dilation=2)

def test_pad_dilation():
    check_forward_backward(input_shape=(3,16,16), ofm1=16, ofm2=16, pad=2, dilation=2)

def test_pad_kernel_dilation():
    check_forward_backward(input_shape=(3,16,16), ofm1=16, ofm2=16, kernel=4, pad=6, dilation=4)

def test_caffe_forward():
    check_caffe_forward(batch_size=1, input_shape=(3,16,16), pad=1)
