# FP    : 212.61279582977295 ms, 1454.7795816750504 GFLOPS/s
# BP+WU : 359.9848985671997 ms, 1718.2454095988542 GFLOPS/s

import unittest
import numpy as np
from latte import *
import latte.util as util
import time
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", nargs=7, type=int, default=[256, 256, 64, 64, 3, 1, 1])
    args = parser.parse_args()

    batch_size = 64
    net = Net(batch_size)
    net.force_backward = True
    channels, height, width, kernel, stride, pad = args.d[1:]
    ofm = args.d[0]
    print("Benchmark Config")
    print("    batch_size = {}".format(batch_size))
    print("    channels   = {}".format(channels))
    print("    height     = {}".format(height))
    print("    width      = {}".format(width))
    print("    ofm        = {}".format(ofm))
    data = MemoryDataLayer(net, (channels, height, width))
    conv1 = ConvLayer(net, data, num_filters=ofm, kernel=kernel, stride=stride, pad=pad)

    net.compile()

    # data.set_value(np.random.rand(batch_size, channels, height, width))

    assert(len(net.forward_tasks) == 2)
    assert(len(net.backward_tasks) == 1)

    run_backward = True

    # warmup
    print("Warming up...")
    for _ in range(3):
        net.forward_tasks[1]()
        if run_backward:
            net.backward_tasks[0]()

    forward_t_total = 0.0
    backward_t_total = 0.0
    num_trials = 5
    print("Running {} trials".format(num_trials))
    for _ in range(num_trials):
        t = time.time()
        net.forward_tasks[1]()
        forward_t_total += time.time() - t 
        if run_backward:
            t = time.time()
            net.backward_tasks[0]()
            backward_t_total += time.time() - t 
    print("Done")

    if pad == 1:
        ofm, oh, ow = ofm, height, width
    else:
        raise NotImplementedError()
    flops = (batch_size * channels * ofm * oh * ow * (2 * 3 * 3))
    forward_flops = flops + batch_size * ofm * oh * ow
    backward_flops = 2 * flops + batch_size * ofm * oh * ow
    freq = util.get_cpu_freq()
    print("cpu_freq = {} GHz".format(freq * 1e-9))
    print("===== FP =====")
    print("    {} ms".format(forward_t_total / num_trials * 1000))
    print("    {} gflops/s".format(
        (forward_flops * num_trials * 1e-9) / forward_t_total))
    print("    {} flops/cycle".format(
        (forward_flops / ((forward_t_total / num_trials) * freq))
    ))
    print("==============")
    if run_backward:
        print("==== BP+WU ===")
        print("    {} ms".format(backward_t_total / num_trials * 1000))
        print("    {} gflops/s".format(
            (backward_flops * num_trials * 1e-9) / backward_t_total))
        print("    {} flops/cycle".format(
            (backward_flops / ((backward_t_total / num_trials) * freq))
        ))
        print("==============")

if __name__ == '__main__':
    main()
