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
import numpy as np
from latte import *
import time
from latte.solvers import sgd_update
import latte
#latte.core.forward_unroll_factor = 8
#latte.core.backward_unroll_factor = 4

def main():
    batch_size = 64
    net = Net(batch_size)
    net.force_backward = False

    print("Benchmark Config")
    print("    batch_size = {}".format(batch_size))
    net.force_backward = False
    data     = MemoryDataLayer(net, (3, 224, 224))

    conv1   = ConvLayer(       net, data, num_filters=64, kernel=3, stride=1, pad=1)
    relu1   = ReLULayer(       net, conv1)
    pool1   = MaxPoolingLayer( net, relu1, kernel=2, stride=2, pad=0)

    conv2   = ConvLayer(       net, pool1, num_filters=128, kernel=3, stride=1, pad=1)
    relu2   = ReLULayer(       net, conv2)
    pool2   = MaxPoolingLayer( net, relu2, kernel=2, stride=2, pad=0)

    conv3_1 = ConvLayer(       net, pool2,   num_filters=256, kernel=3, stride=1, pad=1)
    relu3_1 = ReLULayer(       net, conv3_1)
    conv3_2 = ConvLayer(       net, relu3_1, num_filters=256, kernel=3, stride=1, pad=1)
    relu3_2 = ReLULayer(       net, conv3_2)
    pool3   = MaxPoolingLayer( net, relu3_2, kernel=2, stride=2, pad=0)

    conv4_1 = ConvLayer(       net, pool3,   num_filters=512, kernel=3, stride=1, pad=1)
    relu4_1 = ReLULayer(       net, conv4_1)
    conv4_2 = ConvLayer(       net, relu4_1, num_filters=512, kernel=3, stride=1, pad=1)
    relu4_2 = ReLULayer(       net, conv4_2)
    pool4   = MaxPoolingLayer( net, relu4_2, kernel=2, stride=2, pad=0)

    conv5_1 = ConvLayer(       net, pool4,   num_filters=512, kernel=3, stride=1, pad=1)
    relu5_1 = ReLULayer(       net, conv5_1)
    conv5_2 = ConvLayer(       net, relu5_1, num_filters=512, kernel=3, stride=1, pad=1)
    relu5_2 = ReLULayer(       net, conv5_2)
    pool5   = MaxPoolingLayer( net, relu5_2, kernel=2, stride=2, pad=0)

    fc6     = FullyConnectedLayer(net, pool5, 4096)
    fc7     = FullyConnectedLayer(net, fc6, 4096)
    fc8     = FullyConnectedLayer(net, fc7, 1008)


    print("Compiling...")
    net.compile()

    data.set_value(np.random.rand(batch_size, 3, 224, 224))

    # warmup
    print("Warming up...")
    for _ in range(10):
        net.forward()

    num_trials = 100
    forward_t_total = 0.0
    print("Running trials")
    for _ in range(num_trials):
        t = time.time()
        net.forward()
        forward_t_total += time.time() - t 

    print("FP                  : {0:.3f} ms".format(forward_t_total / num_trials * 1000))
    print("Testing Throughput  : {0:.3f} img/s".format((batch_size * num_trials) / (forward_t_total)))

if __name__ == '__main__':
    main()
