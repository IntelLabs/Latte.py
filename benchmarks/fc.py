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
from scipy import linalg

def main():
    batch_size = 64
    net = Net(batch_size)
    net.force_backward = True
    channels, height, width = 512, 7, 7
    data = MemoryDataLayer(net, (channels, height, width))
    fc1 = FullyConnectedLayerNoBias(net, data, 1024)

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
    print("==== BP+WU ===")
    print("    {} ms".format(backward_t_total / num_trials * 1000))
    print("    {} gflops/s".format(
        (backward_flops * num_trials * 1e-9) / backward_t_total))
    print("    {} flops/cycle".format(
        (backward_flops / ((backward_t_total / num_trials) * freq))
    ))
    print("==============")
    print("BLAS FP : {0:.3f} ms, {1:.3f} GFLOPS/s".format(blas_forward_time * 1000, (forward_flops * 1e-9) / blas_forward_time))
    print("BLAS BP : {0:.3f} ms, {1:.3f} GFLOPS/s".format(blas_backward_time * 1000, (backward_flops * 1e-9) / blas_backward_time))

if __name__ == '__main__':
    main()
