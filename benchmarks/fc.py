import unittest
import numpy as np
from latte import *
import time
from scipy import linalg

def main():
    batch_size = 64
    net = Net(batch_size)
    net.force_backward = True
    channels, height, width = 512, 7, 7
    data = MemoryDataLayer(net, (channels, height, width))
    fc1 = FullyConnectedLayer(net, data, 1024)

    net.compile()

    data.set_value(np.random.rand(batch_size, channels, height, width))

    assert(len(net.forward_tasks) == 2)
    assert(len(net.backward_tasks) == 1)

    # warmup
    for _ in range(3):
        net.forward_tasks[1]()
        net.backward_tasks[0]()

    forward_t_total = 0.0
    backward_t_total = 0.0
    num_trials = 10
    for _ in range(num_trials):
        t = time.time()
        net.forward_tasks[1]()
        forward_t_total += time.time() - t 

        t = time.time()
        net.backward_tasks[0]()
        backward_t_total += time.time() - t 


    M = batch_size
    N = 1024
    K = channels * height * width
    A = np.random.rand(M, K).astype(np.float32)
    B = np.random.rand(K, N).astype(np.float32)
    C = np.zeros((M, N), np.float32)
    np.dot(A, B, out=C)

    t = time.time()
    np.dot(A, B, out=C)
    blas_forward_time = time.time() - t

    t = time.time()
    np.dot(A, B, out=C)
    np.dot(A, B, out=C)
    blas_backward_time = time.time() - t

    forward_flops = 2 * M * N * K
    backward_flops = 2 * M * N * K * 2
    print("FP      : {0:.3f} ms, {1:.3f} GFLOPS/s".format(forward_t_total / num_trials * 1000, 
                                                          (forward_flops * num_trials * 1e-9) / forward_t_total))
    print("BLAS FP : {0:.3f} ms, {1:.3f} GFLOPS/s".format(blas_forward_time * 1000, (forward_flops * 1e-9) / blas_forward_time))
    print("BP      : {0:.3f} ms, {1:.3f} GFLOPS/s".format(backward_t_total / num_trials * 1000, 
                                                          (backward_flops * num_trials * 1e-9) / backward_t_total))
    print("BLAS BP : {0:.3f} ms, {1:.3f} GFLOPS/s".format(blas_backward_time * 1000, (backward_flops * 1e-9) / blas_backward_time))

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
