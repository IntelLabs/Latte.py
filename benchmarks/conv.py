import unittest
import numpy as np
from latte import *
import time

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
                    in_y = max(y*stride_h - pad, 0)
                    in_x = max(x*stride_w - pad, 0)
                    out_y = min(in_y + kernel_h, output_height)
                    out_x = min(in_x + kernel_w, output_width)
                    for c in range(in_channels):
                        for i, p in enumerate(range(in_y, out_y)):
                            for j, q in enumerate(range(in_x, out_x)):
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
                    in_y = max(y*stride_h - pad, 0)
                    in_x = max(x*stride_w - pad, 0)
                    out_y = min(in_y + kernel_h, output_height)
                    out_x = min(in_x + kernel_w, output_width)
                    for c in range(in_channels):
                        for i, p in enumerate(range(in_y, out_y)):
                            for j, q in enumerate(range(in_x, out_x)):
                                weights_grad[o, c, i , j] += top_grad[n, o, y, x] * _input[n, c, p, q]
                                bot_grad[n, c, p, q] += weights[o, c, i, j] * top_grad[n, o, y, x]
    return bot_grad, weights_grad, bias_grad


def check_equal(actual, expected):
    np.testing.assert_array_almost_equal(actual, expected)

def main():
    batch_size = 32
    net = Net(batch_size)
    channels, height, width = 256, 14, 14
    pad = 0
    ofm = 512
    data, data_value = MemoryDataLayer(net, (channels, height, width))
    conv1 = ConvLayer(net, data, num_filters=ofm, kernel=3, stride=1, pad=pad)

    data_value[:, :, :] = np.random.rand(batch_size, channels, height, width)

    net.compile()

    weights = net.buffers[conv1.name + "weights"]
    bias    = net.buffers[conv1.name + "bias"]
    weights_converted = np.copy(weights)
    shape = weights.shape
    for ofm in range(shape[0] // 8):
        for ifm in range(shape[1] // 8):
            for y in range(shape[2]):
                for x in range(shape[3]):
                    for v2 in range(8):
                        for v in range(8):
                            weights.flat[(((ofm * (shape[1] // 8) + ifm) * shape[2] + y) * shape[3] + x) * 8 + v] = \
                                    weights_converted[ofm * 8 + v, ifm * 8 + v2, y, x]

    assert(len(net.forward_tasks) == 2)
    assert(len(net.backward_tasks) == 1)

    # warmup
    for _ in range(3):
        net.forward_tasks[1]()
        net.backward_tasks[0]()

    forward_t_total = 0.0
    backward_t_total = 0.0
    num_trials = 2
    for _ in range(num_trials):
        t = time.time()
        net.forward_tasks[1]()
        forward_t_total += time.time() - t 
        t = time.time()
        net.backward_tasks[0]()
        backward_t_total += time.time() - t 


    _, ofm, oh, ow = net.buffers[conv1.name + "value"].shape
    gflops = (batch_size * channels * ofm * oh * ow * (2 * 3 * 3)) * num_trials * 1e-9
    print("GFLOPS/s : fp = {}, bp = {}".format(gflops / forward_t_total, (gflops * 2) / backward_t_total))

    # expected = reference_conv_forward(data_value, weights_converted, bias, pad, 1)

    # expected_converted = np.zeros_like(expected)
    # shape = expected.shape
    # for n in range(shape[0]):
    #     for ifm in range(shape[1] // 8):
    #         for y in range(shape[2]):
    #             for x in range(shape[3]):
    #                 for v in range(8):
    #                     expected_converted.flat[(((n * (shape[1] // 8) + ifm) * shape[2] + y) * shape[3] + x) * 8 + v] = expected[n, ifm * 8 + v, y, x]
    # actual  = net.buffers[conv1.name + "value"]
    # self._check_equal(actual, expected_converted)

    # top_grad = net.buffers[conv2.name + "grad"]
    # np.copyto(top_grad, np.random.rand(*top_grad.shape))

    # net.backward()
    # weights = net.buffers[conv2.name + "weights"]

    # expected_bot_grad, expected_weights_grad, expected_bias_grad = \
    #         reference_conv_backward(top_grad, actual, weights, 1, 1)

    # bot_grad = net.buffers[conv1.name + "grad"]
    # self._check_equal(bot_grad, expected_bot_grad)

    # weights_grad = net.buffers[conv2.name + "grad_weights"]
    # self._check_equal(weights_grad, expected_weights_grad)

    # bias_grad = net.buffers[conv2.name + "grad_bias"]
    # self._check_equal(bias_grad, expected_bias_grad)

if __name__ == '__main__':
    main()