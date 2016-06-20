import pytest
import numpy as np
from latte import *
import latte.util as util
import sys
import math

def reference_interpolation_forward(_input, pad, resize_factor):
    pad_h, pad_w = pad, pad
    batch_size, in_channels, in_height, in_width = _input.shape
    output_width = math.floor(in_width * resize_factor)
    output_height = math.floor(in_height * resize_factor)
    output = np.zeros((batch_size, in_channels, output_height, output_width), dtype=np.float32)
    for n in range(batch_size):
        for o in range(in_channels):
            for y in range(output_height):
                delta_r = (y*resize_factor) - math.floor(y*resize_factor)
                for x in range(output_width):
                    delta_c = (x*resize_factor) - math.floor(x*resize_factor)
                    
                    in_y = min(max(math.floor(y/resize_factor) - pad, 0), in_height - 1)
                    in_x = min(max(math.floor(x/resize_factor) - pad, 0), in_width - 1)
                    
                    in_y_plus_1 = in_y+1
                    in_x_plus_1 = in_x+1

                    if in_y_plus_1 > in_height - 1:
                        in_y_plus_1 = in_y                    
                    if in_x_plus_1 > in_width - 1:
                        in_x_plus_1 = in_x                

                    output[n, o, y, x] = \
                        _input[n, o, in_y, in_x]               * (1-delta_r) * (1-delta_c) + \
                        _input[n, o, in_y_plus_1, in_x]        * (delta_r)   * (1-delta_c) + \
                        _input[n, o, in_y, in_x_plus_1]        * (1-delta_r) * (delta_c)   + \
                        _input[n, o, in_y_plus_1, in_x_plus_1] * (delta_r)   * (delta_c)

    return output


def reference_interpolation_backward(top_grad, _input, pad, resize_factor):
    pad_h, pad_w = pad, pad
    batch_size, in_channels, in_height, in_width = _input.shape
    _, output_channels, output_height, output_width = top_grad.shape
    bot_grad = np.zeros_like(_input)
    for n in range(batch_size):
        for o in range(output_channels):
            for y in range(output_height):
                delta_r = (y*resize_factor) - math.floor(y*resize_factor)
                for x in range(output_width):
                    delta_c = (x*resize_factor) - math.floor(x*resize_factor)

                    in_y = min(max(math.floor(y/resize_factor) - pad, 0), in_height - 1)
                    in_x = min(max(math.floor(x/resize_factor) - pad, 0), in_width - 1)
                    
                    in_y_plus_1 = in_y+1
                    in_x_plus_1 = in_x+1

                    if in_y_plus_1 > in_height - 1:
                        in_y_plus_1 = in_y                    
                    if in_x_plus_1 > in_width - 1:
                        in_x_plus_1 = in_x                

                    bot_grad[n,o,in_y,in_x] += (1-delta_r)*(1-delta_c)*top_grad[n, o, y, x]
                    bot_grad[n,o,in_y_plus_1,in_x] += (delta_r)*(1-delta_c)*top_grad[n, o, y, x]
                    bot_grad[n,o,in_y,in_x_plus_1] += (1-delta_r)*(delta_c)*top_grad[n, o, y, x]
                    bot_grad[n,o,in_y_plus_1,in_x_plus_1] += (delta_r)*(delta_c)*top_grad[n, o, y, x]
    return bot_grad


def check_equal(actual, expected, atol=1e-6):
    assert np.allclose(actual, expected, atol=atol)

def test_forward_backward_double():
    net = Net(8)
    net.force_backward = True
    channels, height, width = 16, 16, 16
    pad = 0
    resize_factor = 2.0
    data = MemoryDataLayer(net, (channels, height, width))
    interp1 = InterpolationLayer(net, data, pad=pad, resize_factor=resize_factor)
    
    net.compile()

    data_value = np.random.rand(8, channels, height, width)
    data.set_value(data_value)


    net.forward()
   
    expected = reference_interpolation_forward(data_value, pad, resize_factor)
    actual  = interp1.get_value()
    check_equal(actual, expected)

    top_grad = interp1.get_grad()
    top_grad = np.random.rand(*top_grad.shape)
    interp1.set_grad(top_grad)

    net.backward()

    expected_bot_grad = reference_interpolation_backward(top_grad,
            data_value, pad, resize_factor)

    bot_grad = interp1.get_grad_inputs()
    check_equal(bot_grad, expected_bot_grad)

def test_forward_backward_enlarge():
    net = Net(8)
    net.force_backward = True
    channels, height, width = 16, 16, 16
    pad = 0
    resize_factor = 8.0
    data = MemoryDataLayer(net, (channels, height, width))
    interp1 = InterpolationLayer(net, data, pad=pad, resize_factor=resize_factor)

    net.compile()
    
    data_value = np.random.rand(8, channels, height, width)
    data.set_value(data_value)

    net.forward()
   
    expected = reference_interpolation_forward(data_value, pad, resize_factor)
    actual  = interp1.get_value()
    check_equal(actual, expected)

    top_grad = interp1.get_grad()
    top_grad = np.random.rand(*top_grad.shape)
    interp1.set_grad(top_grad)

    net.backward()

    expected_bot_grad = reference_interpolation_backward(top_grad,
            data_value, pad, resize_factor)

    bot_grad = interp1.get_grad_inputs()
    check_equal(bot_grad, expected_bot_grad, atol=1e-4)


