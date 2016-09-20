#import ast
import pytest
import numpy as np
from latte import *
import latte.util as util
#from latte.ensemble import Ensemble, DataEnsemble, ActivationEnsemble, LossEnsemble, AccuracyEnsemble, EnsembleGroup
#import ctree
#from ctree.transformations import PyBasicConversions
#import sys

def reference_pooling_forward(_input, kernel, pad, stride):
    stride_h, stride_w = stride, stride
    pad_h, pad_w = pad, pad
    kernel_h, kernel_w = kernel, kernel
    batch_size, in_channels, in_height, in_width = _input.shape
    output_width = ((in_width - kernel_w + 2 * pad_w) // stride_w) + 1
    output_height = ((in_height - kernel_h + 2 * pad_h) // stride_h) + 1
    output = np.zeros((batch_size, in_channels, output_height, output_width), dtype=np.float32)
    output_mask = np.zeros((batch_size, in_channels, output_height, output_width, 2), dtype=np.int32)
    for n in range(batch_size):
        for o in range(in_channels):
            for y in range(output_height):
                for x in range(output_width):
                    in_y = y*stride_h - pad
                    in_x = x*stride_w - pad
                    out_y = in_y + kernel_h
                    out_x = in_x + kernel_w
                    sumval = 0.0
                    #idx = ()
                    for i, p in enumerate(range(in_y, out_y)):
                        p = min(max(p, 0), in_height - 1)
                        for j, q in enumerate(range(in_x, out_x)):
                            q = min(max(q, 0), in_width - 1)
                            curr = _input[n, o, p, q]
                            #rint(curr)
                            sumval += curr    
                            #if curr > maxval:
                            #    idx = (i, j)
                            #    maxval = curr
                    output[n, o, y, x] = sumval/(kernel_h*kernel_w)
                    #output_mask[n, o, y, x, :] = idx
    return output

def reference_pooling_backward(top_grad, _input, stride=2, kernel=2, pad=0):
    stride_h, stride_w = stride, stride
    pad_h, pad_w = pad, pad
    kernel_h, kernel_w = kernel,kernel
    batch_size, in_channels, in_height, in_width = _input.shape
    _, output_channels, output_height, output_width = top_grad.shape
    bot_grad = np.zeros_like(_input)
    for n in range(batch_size):
        for o in range(output_channels):
            for y in range(output_height):
                for x in range(output_width):
                    in_y = y*stride_h - pad
                    in_x = x*stride_w - pad
                    out_y = in_y + kernel_h
                    out_x = in_x + kernel_w
                    #sumval = 0.0
                    for i, p in enumerate(range(in_y, out_y)):
                        p = min(max(p, 0), in_height - 1)
                        for j, q in enumerate(range(in_x, out_x)):
                            q = min(max(q, 0), in_width - 1)
                            #curr = _input[n, o, p, q]
                            #sumval += curr
                            #if curr > maxval:
                            #    idx = (i, j)
                            #    maxval = curr
                            bot_grad[n,o,p,q] += top_grad[n, o, y, x]/(kernel_h*kernel_w)
    return bot_grad

def check_equal(actual, expected, atol=1e-6):
    assert np.allclose(actual, expected, atol=atol)

def test_forward_backward():
    net = Net(8)
    net.force_backward = True
    channels, height, width = 16, 16, 16
    pad = 0
    

    data = MemoryDataLayer(net, (channels, height, width))
    #data_value = np.random.rand(8, channels, height, width)
    #data.set_value(data_value)


    pool1 = MeanPoolingLayer(net, data, kernel=2, stride=2, pad=pad)

    net.compile()
 
    data_value = np.random.rand(8, channels, height, width)
    data.set_value(data_value)
    #print(data_value)
    net.forward()
    #print(data_value)
    expected = reference_pooling_forward(data_value, 2, pad, 2)

    actual  = pool1.get_value()
    #print(actual)

    #actual_mask_j  = pool1.get_mask_j()
    #actual_mask_k  = pool1.get_mask_k()
    check_equal(actual, expected)
    #check_equal(actual_mask_j, expected_mask[:, :, :, :, 0])
    #check_equal(actual_mask_k, expected_mask[:, :, :, :, 1])

    top_grad = pool1.get_grad()
    top_grad = np.random.rand(*top_grad.shape)
    pool1.set_grad(top_grad)

    net.backward()

    expected_bot_grad = \
        reference_pooling_backward(top_grad, data_value, stride=2, kernel=2, pad=0)

    bot_grad = pool1.get_grad_inputs()
    check_equal(bot_grad, expected_bot_grad)

#def main():
#    test_forward_backward()

#if __name__ == "__main__":
#   main()

