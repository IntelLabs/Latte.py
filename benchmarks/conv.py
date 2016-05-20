import unittest
import numpy as np
from latte import *
import time
from tqdm import tqdm 
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", nargs=4, type=int, default=[256, 256, 64, 64])
    args = parser.parse_args()

    batch_size = 32
    net = Net(batch_size)
    print(args.d)
    channels, height, width = args.d[1:]
    pad = 1
    ofm = args.d[0]
    print("Benchmark Config")
    print("    batch_size = {}".format(batch_size))
    print("    channels   = {}".format(channels))
    print("    height     = {}".format(height))
    print("    width      = {}".format(width))
    print("    ofm        = {}".format(ofm))
    data = MemoryDataLayer(net, (channels, height, width))
    conv1, conv1bias = ConvLayer(net, data, num_filters=ofm, kernel=3, stride=1, pad=pad)

    data.set_value(np.random.rand(batch_size, channels, height, width))

    print("Compiling...")
    net.compile()

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
    num_trials = 10
    print("Running trials")
    for _ in tqdm(range(num_trials), ncols=100):
        t = time.time()
        net.forward_tasks[1]()
        forward_t_total += time.time() - t 
        if run_backward:
            t = time.time()
            net.backward_tasks[0]()
            backward_t_total += time.time() - t 

    _, ofm_outer, oh, ow, ofm_inner = net.buffers[conv1.name + "value"].shape
    ofm = ofm_outer * ofm_inner
    flops = (batch_size * channels * ofm * oh * ow * (2 * 3 * 3))
    forward_flops = flops + batch_size * ofm * oh * ow
    backward_flops = 2 * flops + batch_size * ofm * oh * ow
    print("FP    : {} ms, {} GFLOPS/s".format(forward_t_total / num_trials * 1000, 
                                             (forward_flops * num_trials * 1e-9) / forward_t_total))
    if run_backward:
        print("BP+WU : {} ms, {} GFLOPS/s".format(backward_t_total / num_trials * 1000, 
                                                 (backward_flops * num_trials * 1e-9) / backward_t_total))

if __name__ == '__main__':
    main()
