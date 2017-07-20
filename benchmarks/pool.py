'''
Copyright (c) 2015, Intel Corporation

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright notice,
      this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of Intel Corporation nor the names of its contributors
      may be used to endorse or promote products derived from this software
      without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
'''
import unittest
import numpy as np
from latte import *
import time
from tqdm import tqdm 
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", nargs=3, type=int, default=[256, 64, 64])
    args = parser.parse_args()

    batch_size = 32
    net = Net(batch_size)
    channels, height, width = args.d
    print("Benchmark Config")
    print("    batch_size = {}".format(batch_size))
    print("    channels = {}".format(channels))
    print("    height = {}".format(height))
    print("    width = {}".format(width))
    data = MemoryDataLayer(net, (channels, height, width))
    out = MaxPoolingLayer(net, data, kernel=2, stride=2, pad=0)

    #data.set_value(np.random.rand(batch_size, channels, height, width))

    print("Compiling...")
    net.compile()
    data.set_value(np.random.rand(batch_size, channels, height, width))

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

    flops = channels * height * width * batch_size
    forward_flops = flops * 2 * 2
    backward_flops = flops
    print("FP    : {} ms, {} GFLOPS/s".format(forward_t_total / num_trials * 1000, 
                                             (forward_flops * num_trials * 1e-9) / forward_t_total))
    if run_backward:
        print("BP+WU : {} ms, {} GFLOPS/s".format(backward_t_total / num_trials * 1000, 
                                                 (backward_flops * num_trials * 1e-9) / backward_t_total))

if __name__ == '__main__':
    main()
