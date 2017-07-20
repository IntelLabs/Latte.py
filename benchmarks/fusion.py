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
    batch_size = 128
    args = parser.parse_args()

    #fusion_recipe = FusionRecipe() 

    net = Net(batch_size)
    net.force_backward = True
    net.fuse_pattern_cbr()

    data = MemoryDataLayer(net, (3, 224, 224))
    conv1_7x7_s2 = ConvLayer(net, data, num_filters=64, kernel=7, stride=2, pad=3)
    conv1_relu_7x7 = ReLULayer(net, conv1_7x7_s2)
    pool1_3x3_s2 = MaxPoolingLayer(net, conv1_relu_7x7, kernel=3, stride=2, pad=0)
    #fusion_recipe.pattern_cbr(net, "forward", conv1_7x7_s2, conv1_relu_7x7)

    net.compile()

    # data.set_value(np.random.rand(batch_size, channels, height, width))

    assert(len(net.forward_tasks) == 2)
    assert(len(net.backward_tasks) == 1)

    run_backward = True
    #run_backward = channels%8 == 0 ? True : False

    # warmup
    print("Warming up...")
    for _ in range(3):
        net.forward_tasks[1]()
        if run_backward:
            net.backward_tasks[0]()

    forward_t_total = 0.0
    backward_t_total = 0.0
    #num_trials = 1000
    num_trials = 100
    print("Running {} trials".format(num_trials))
    #print(net.forward_tasks[1])
    for _ in range(num_trials):
        t = time.time()
        net.forward_tasks[1]()
        forward_t_total += time.time() - t 
        if run_backward:
            t = time.time()
            #net.backward_tasks[0]()
            backward_t_total += time.time() - t 
    print("Done")
    print("forward time=", forward_t_total, " backward time=", backward_t_total)

    '''
    oh = int(np.ceil((height-kernel+1+2*pad)/stride))
    ow = int(np.ceil((width-kernel+1+2*pad)/stride))
    #if stride == 1:
    #  ofm, oh, ow = ofm, height, width
    #else:
    #    raise NotImplementedError()
    print("batch=", batch_size, " channels=", channels, " ofm=", ofm, " oh=", oh, " ow=", ow, " kernel=", kernel, " pad=", pad, " stride=", stride)
    flops = (batch_size * channels * ofm * oh * ow * (2 * kernel * kernel))
    forward_flops = flops
    #includes both backward and weight update
    backward_flops = 2 * flops
    freq = util.get_cpu_freq()
    print("cpu_freq = {} GHz".format(freq * 1e-9))
    print("===== FP =====")
    print(" FP   {} ms".format(forward_t_total / num_trials * 1000))
    print(" FP   {} gflops/s".format(
        (forward_flops * num_trials * 1e-9) / forward_t_total))
    #print("    {} flops/cycle".format(
    #    (forward_flops / ((forward_t_total / num_trials) * freq))
    #))
    print("==============")
    if run_backward:
        print("==== BP+WU ===")
        print(" BP+WU   {} ms".format(backward_t_total / num_trials * 1000))
        print(" BP+WU   {} gflops/s".format(
            (backward_flops * num_trials * 1e-9) / backward_t_total))
        #print("    {} flops/cycle".format(
        #    (backward_flops / ((backward_t_total / num_trials) * freq))
        #))
        print("==============")
    print("PERFDUMP: ", batch_size, " ", channels, " ", height, " ", width, " ", ofm, " ", kernel, " ", stride, " ", pad, " {}".format((forward_flops * num_trials * 1e-9) / forward_t_total), " {}".format((backward_flops * num_trials * 1e-9) / backward_t_total)) 
    '''

if __name__ == '__main__':
    main()
