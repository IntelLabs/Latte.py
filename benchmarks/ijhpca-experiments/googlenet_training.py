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
#import caffe
from latte.math import compute_softmax_loss
import os
import time

batch_size = 128
net = Net(batch_size)
channels=3
height=224
width=224
net.force_backward = False
data = MemoryDataLayer(net, (3, 224, 224))
conv1_7x7_s2 = ConvLayer(net, data, num_filters=64, kernel=7, stride=2, pad=3)
conv1_relu_7x7 = ReLULayer(net, conv1_7x7_s2)
pool1_3x3_s2 = MaxPoolingLayer(net, conv1_relu_7x7, kernel=3, stride=2, pad=0)

#pool1_norm1 = LRNLayer(net, pool1_3x3_s2, n=5, beta=0.75, alpha=0.0001, k=1.0)
#conv2_3x3_reduce = ConvLayer(net, pool1_norm1, num_filters=64, kernel=1, stride=1, pad=0)
conv2_3x3_reduce = ConvLayer(net, pool1_3x3_s2, num_filters=64, kernel=1, stride=1, pad=0)

conv2_relu_3x3_reduce = ReLULayer(net, conv2_3x3_reduce)

conv2_3x3 = ConvLayer(net, conv2_relu_3x3_reduce, num_filters=192, kernel=3, stride=1, pad=1)
conv2_relu_3x3 = ReLULayer(net, conv2_3x3)
#conv2_norm2 = LRNLayer(net, conv2_relu_3x3, n=5, beta=0.75, alpha=0.0001, k=1.0)
#pool2_3x3_s2 = MaxPoolingLayer(net, conv2_norm2, kernel=3, stride=2, pad=0)
pool2_3x3_s2 = MaxPoolingLayer(net, conv2_relu_3x3, kernel=3, stride=2, pad=0)
 


inception_3a_1x1 = ConvLayer(net, pool2_3x3_s2, num_filters=64, kernel=1, stride=1, pad=0)
inception_3a_relu_1x1 = ReLULayer(net, inception_3a_1x1)

inception_3a_3x3_reduce = ConvLayer(net, pool2_3x3_s2, num_filters=96, kernel=1, stride=1, pad=0)
inception_3a_relu_3x3_reduce = ReLULayer(net, inception_3a_3x3_reduce)

inception_3a_3x3 = ConvLayer(net, inception_3a_relu_3x3_reduce, num_filters=128, kernel=3, stride=1, pad=1)
inception_3a_relu_3x3 = ReLULayer(net, inception_3a_3x3)

inception_3a_5x5_reduce = ConvLayer(net, pool2_3x3_s2, num_filters=16, kernel=1, stride=1, pad=0)
inception_3a_relu_5x5_reduce = ReLULayer(net, inception_3a_5x5_reduce)
inception_3a_5x5 = ConvLayer(net, inception_3a_relu_5x5_reduce, num_filters=32, kernel=5, stride=1, pad=2)
inception_3a_relu_5x5 = ReLULayer(net, inception_3a_5x5)
inception_3a_pool = MaxPoolingLayer(net, pool2_3x3_s2, kernel=3, stride=1, pad=1)

inception_3a_pool_proj = ConvLayer(net, inception_3a_pool, num_filters=32, kernel=1, stride=1, pad=0)
inception_3a_relu_pool_proj = ReLULayer(net, inception_3a_pool_proj)
inception_3a_output = ConcatLayer(net, [inception_3a_relu_1x1, inception_3a_relu_3x3, inception_3a_relu_5x5, inception_3a_relu_pool_proj])
inception_3b_1x1 = ConvLayer(net, inception_3a_output, num_filters=128, kernel=1, stride=1, pad=0)
inception_3b_relu_1x1 = ReLULayer(net, inception_3b_1x1)
inception_3b_3x3_reduce = ConvLayer(net, inception_3a_output, num_filters=128, kernel=1, stride=1, pad=0)
inception_3b_relu_3x3_reduce = ReLULayer(net, inception_3b_3x3_reduce)
inception_3b_3x3 = ConvLayer(net, inception_3b_relu_3x3_reduce, num_filters=192, kernel=3, stride=1, pad=1)
inception_3b_relu_3x3 = ReLULayer(net, inception_3b_3x3)
inception_3b_5x5_reduce = ConvLayer(net, inception_3a_output, num_filters=32, kernel=1, stride=1, pad=0)
inception_3b_relu_5x5_reduce = ReLULayer(net, inception_3b_5x5_reduce)
inception_3b_5x5 = ConvLayer(net, inception_3b_relu_5x5_reduce, num_filters=96, kernel=5, stride=1, pad=2)
inception_3b_relu_5x5 = ReLULayer(net, inception_3b_5x5)
inception_3b_pool = MaxPoolingLayer(net, inception_3a_output, kernel=3, stride=1, pad=1)
inception_3b_pool_proj = ConvLayer(net, inception_3b_pool, num_filters=64, kernel=1, stride=1, pad=0)
inception_3b_relu_pool_proj = ReLULayer(net, inception_3b_pool_proj)
inception_3b_output = ConcatLayer(net, [inception_3b_relu_1x1, inception_3b_relu_3x3, inception_3b_relu_5x5, inception_3b_relu_pool_proj])
pool3_3x3_s2 = MaxPoolingLayer(net, inception_3b_output, kernel=3, stride=2, pad=0)
inception_4a_1x1 = ConvLayer(net, pool3_3x3_s2, num_filters=192, kernel=1, stride=1, pad=0)
inception_4a_relu_1x1 = ReLULayer(net, inception_4a_1x1)
inception_4a_3x3_reduce = ConvLayer(net, pool3_3x3_s2, num_filters=96, kernel=1, stride=1, pad=0)
inception_4a_relu_3x3_reduce = ReLULayer(net, inception_4a_3x3_reduce)
inception_4a_3x3 = ConvLayer(net, inception_4a_relu_3x3_reduce, num_filters=208, kernel=3, stride=1, pad=1)
inception_4a_relu_3x3 = ReLULayer(net, inception_4a_3x3)
inception_4a_5x5_reduce = ConvLayer(net, pool3_3x3_s2, num_filters=16, kernel=1, stride=1, pad=0)
inception_4a_relu_5x5_reduce = ReLULayer(net, inception_4a_5x5_reduce)
inception_4a_5x5 = ConvLayer(net, inception_4a_relu_5x5_reduce, num_filters=48, kernel=5, stride=1, pad=2)
inception_4a_relu_5x5 = ReLULayer(net, inception_4a_5x5)
inception_4a_pool = MaxPoolingLayer(net, pool3_3x3_s2, kernel=3, stride=1, pad=1)
inception_4a_pool_proj = ConvLayer(net, inception_4a_pool, num_filters=64, kernel=1, stride=1, pad=0)
inception_4a_relu_pool_proj = ReLULayer(net, inception_4a_pool_proj)
inception_4a_output = ConcatLayer(net, [inception_4a_relu_1x1, inception_4a_relu_3x3, inception_4a_relu_5x5, inception_4a_relu_pool_proj])
inception_4b_1x1 = ConvLayer(net, inception_4a_output, num_filters=160, kernel=1, stride=1, pad=0)
inception_4b_relu_1x1 = ReLULayer(net, inception_4b_1x1)
inception_4b_3x3_reduce = ConvLayer(net, inception_4a_output, num_filters=112, kernel=1, stride=1, pad=0)
inception_4b_relu_3x3_reduce = ReLULayer(net, inception_4b_3x3_reduce)
inception_4b_3x3 = ConvLayer(net, inception_4b_relu_3x3_reduce, num_filters=224, kernel=3, stride=1, pad=1)
inception_4b_relu_3x3 = ReLULayer(net, inception_4b_3x3)
inception_4b_5x5_reduce = ConvLayer(net, inception_4a_output, num_filters=24, kernel=1, stride=1, pad=0)
inception_4b_relu_5x5_reduce = ReLULayer(net, inception_4b_5x5_reduce)
inception_4b_5x5 = ConvLayer(net, inception_4b_relu_5x5_reduce, num_filters=64, kernel=5, stride=1, pad=2)
inception_4b_relu_5x5 = ReLULayer(net, inception_4b_5x5)
inception_4b_pool = MaxPoolingLayer(net, inception_4a_output, kernel=3, stride=1, pad=1)
inception_4b_pool_proj = ConvLayer(net, inception_4b_pool, num_filters=64, kernel=1, stride=1, pad=0)
inception_4b_relu_pool_proj = ReLULayer(net, inception_4b_pool_proj)
inception_4b_output = ConcatLayer(net, [inception_4b_relu_1x1, inception_4b_relu_3x3, inception_4b_relu_5x5, inception_4b_relu_pool_proj])
inception_4c_1x1 = ConvLayer(net, inception_4b_output, num_filters=128, kernel=1, stride=1, pad=0)
inception_4c_relu_1x1 = ReLULayer(net, inception_4c_1x1)
inception_4c_3x3_reduce = ConvLayer(net, inception_4b_output, num_filters=128, kernel=1, stride=1, pad=0)
inception_4c_relu_3x3_reduce = ReLULayer(net, inception_4c_3x3_reduce)
inception_4c_3x3 = ConvLayer(net, inception_4c_relu_3x3_reduce, num_filters=256, kernel=3, stride=1, pad=1)
inception_4c_relu_3x3 = ReLULayer(net, inception_4c_3x3)
inception_4c_5x5_reduce = ConvLayer(net, inception_4b_output, num_filters=24, kernel=1, stride=1, pad=0)
inception_4c_relu_5x5_reduce = ReLULayer(net, inception_4c_5x5_reduce)
inception_4c_5x5 = ConvLayer(net, inception_4c_relu_5x5_reduce, num_filters=64, kernel=5, stride=1, pad=2)
inception_4c_relu_5x5 = ReLULayer(net, inception_4c_5x5)
inception_4c_pool = MaxPoolingLayer(net, inception_4b_output, kernel=3, stride=1, pad=1)
inception_4c_pool_proj = ConvLayer(net, inception_4c_pool, num_filters=64, kernel=1, stride=1, pad=0)
inception_4c_relu_pool_proj = ReLULayer(net, inception_4c_pool_proj)
inception_4c_output = ConcatLayer(net, [inception_4c_relu_1x1, inception_4c_relu_3x3, inception_4c_relu_5x5, inception_4c_relu_pool_proj])
inception_4d_1x1 = ConvLayer(net, inception_4c_output, num_filters=112, kernel=1, stride=1, pad=0)
inception_4d_relu_1x1 = ReLULayer(net, inception_4d_1x1)
inception_4d_3x3_reduce = ConvLayer(net, inception_4c_output, num_filters=144, kernel=1, stride=1, pad=0)
inception_4d_relu_3x3_reduce = ReLULayer(net, inception_4d_3x3_reduce)
inception_4d_3x3 = ConvLayer(net, inception_4d_relu_3x3_reduce, num_filters=288, kernel=3, stride=1, pad=1)
inception_4d_relu_3x3 = ReLULayer(net, inception_4d_3x3)
inception_4d_5x5_reduce = ConvLayer(net, inception_4c_output, num_filters=32, kernel=1, stride=1, pad=0)
inception_4d_relu_5x5_reduce = ReLULayer(net, inception_4d_5x5_reduce)
inception_4d_5x5 = ConvLayer(net, inception_4d_relu_5x5_reduce, num_filters=64, kernel=5, stride=1, pad=2)
inception_4d_relu_5x5 = ReLULayer(net, inception_4d_5x5)
inception_4d_pool = MaxPoolingLayer(net, inception_4c_output, kernel=3, stride=1, pad=1)
inception_4d_pool_proj = ConvLayer(net, inception_4d_pool, num_filters=64, kernel=1, stride=1, pad=0)
inception_4d_relu_pool_proj = ReLULayer(net, inception_4d_pool_proj)
inception_4d_output = ConcatLayer(net, [inception_4d_relu_1x1, inception_4d_relu_3x3, inception_4d_relu_5x5, inception_4d_relu_pool_proj])
inception_4e_1x1 = ConvLayer(net, inception_4d_output, num_filters=256, kernel=1, stride=1, pad=0)
inception_4e_relu_1x1 = ReLULayer(net, inception_4e_1x1)
inception_4e_3x3_reduce = ConvLayer(net, inception_4d_output, num_filters=160, kernel=1, stride=1, pad=0)
inception_4e_relu_3x3_reduce = ReLULayer(net, inception_4e_3x3_reduce)
inception_4e_3x3 = ConvLayer(net, inception_4e_relu_3x3_reduce, num_filters=320, kernel=3, stride=1, pad=1)
inception_4e_relu_3x3 = ReLULayer(net, inception_4e_3x3)
inception_4e_5x5_reduce = ConvLayer(net, inception_4d_output, num_filters=32, kernel=1, stride=1, pad=0)
inception_4e_relu_5x5_reduce = ReLULayer(net, inception_4e_5x5_reduce)
inception_4e_5x5 = ConvLayer(net, inception_4e_relu_5x5_reduce, num_filters=128, kernel=5, stride=1, pad=2)
inception_4e_relu_5x5 = ReLULayer(net, inception_4e_5x5)
inception_4e_pool = MaxPoolingLayer(net, inception_4d_output, kernel=3, stride=1, pad=1)
inception_4e_pool_proj = ConvLayer(net, inception_4e_pool, num_filters=128, kernel=1, stride=1, pad=0)
inception_4e_relu_pool_proj = ReLULayer(net, inception_4e_pool_proj)
inception_4e_output = ConcatLayer(net, [inception_4e_relu_1x1, inception_4e_relu_3x3, inception_4e_relu_5x5, inception_4e_relu_pool_proj])
pool4_3x3_s2 = MaxPoolingLayer(net, inception_4e_output, kernel=3, stride=2, pad=0)
inception_5a_1x1 = ConvLayer(net, pool4_3x3_s2, num_filters=256, kernel=1, stride=1, pad=0)
inception_5a_relu_1x1 = ReLULayer(net, inception_5a_1x1)
inception_5a_3x3_reduce = ConvLayer(net, pool4_3x3_s2, num_filters=160, kernel=1, stride=1, pad=0)
inception_5a_relu_3x3_reduce = ReLULayer(net, inception_5a_3x3_reduce)
inception_5a_3x3 = ConvLayer(net, inception_5a_relu_3x3_reduce, num_filters=320, kernel=3, stride=1, pad=1)
inception_5a_relu_3x3 = ReLULayer(net, inception_5a_3x3)
inception_5a_5x5_reduce = ConvLayer(net, pool4_3x3_s2, num_filters=32, kernel=1, stride=1, pad=0)
inception_5a_relu_5x5_reduce = ReLULayer(net, inception_5a_5x5_reduce)
inception_5a_5x5 = ConvLayer(net, inception_5a_relu_5x5_reduce, num_filters=128, kernel=5, stride=1, pad=2)
inception_5a_relu_5x5 = ReLULayer(net, inception_5a_5x5)
inception_5a_pool = MaxPoolingLayer(net, pool4_3x3_s2, kernel=3, stride=1, pad=1)
inception_5a_pool_proj = ConvLayer(net, inception_5a_pool, num_filters=128, kernel=1, stride=1, pad=0)
inception_5a_relu_pool_proj = ReLULayer(net, inception_5a_pool_proj)
inception_5a_output = ConcatLayer(net, [inception_5a_relu_1x1, inception_5a_relu_3x3, inception_5a_relu_5x5, inception_5a_relu_pool_proj])
inception_5b_1x1 = ConvLayer(net, inception_5a_output, num_filters=384, kernel=1, stride=1, pad=0)
inception_5b_relu_1x1 = ReLULayer(net, inception_5b_1x1)
inception_5b_3x3_reduce = ConvLayer(net, inception_5a_output, num_filters=192, kernel=1, stride=1, pad=0)
inception_5b_relu_3x3_reduce = ReLULayer(net, inception_5b_3x3_reduce)
inception_5b_3x3 = ConvLayer(net, inception_5b_relu_3x3_reduce, num_filters=384, kernel=3, stride=1, pad=1)
inception_5b_relu_3x3 = ReLULayer(net, inception_5b_3x3)
inception_5b_5x5_reduce = ConvLayer(net, inception_5a_output, num_filters=48, kernel=1, stride=1, pad=0)
inception_5b_relu_5x5_reduce = ReLULayer(net, inception_5b_5x5_reduce)
inception_5b_5x5 = ConvLayer(net, inception_5b_relu_5x5_reduce, num_filters=128, kernel=5, stride=1, pad=2)
inception_5b_relu_5x5 = ReLULayer(net, inception_5b_5x5)
inception_5b_pool = MaxPoolingLayer(net, inception_5a_output, kernel=3, stride=1, pad=1)
inception_5b_pool_proj = ConvLayer(net, inception_5b_pool, num_filters=128, kernel=1, stride=1, pad=0)
inception_5b_relu_pool_proj = ReLULayer(net, inception_5b_pool_proj)
inception_5b_output = ConcatLayer(net, [inception_5b_relu_1x1, inception_5b_relu_3x3, inception_5b_relu_5x5, inception_5b_relu_pool_proj])
pool5_7x7_s1 = MeanPoolingLayer(net, inception_5b_output, kernel=7, stride=1, pad=0)
#pool5_drop_7x7_s1 = DropoutLayer(net, pool5_7x7_s1, ratio=0.4)
loss3_classifier =  FullyConnectedLayer(net, pool5_7x7_s1, 1008)
#prob = SoftmaxLossLayer(net, loss3_classifier)


net.compile() 


data_value = np.random.rand(batch_size, channels, height, width)

data.set_value(data_value)
#print("Finished Copying\n")
#net.forward()
#net.forward()
#net.forward()
print("Warming up...")

for _ in range(10):
    net.forward()
 



total_backward_time=0.0
num_trials = 100
for epoch in range(num_trials):
 
 
        #for i in range(1):
        #forward_time = 0.0
        #backward_time = 0.0
 
 
        t = time.time()
        net.forward()
        net.backward()
        total_backward_time += time.time() - t
print("FP+BP+WU                  : {0:.3f} ms".format(total_backward_time / num_trials * 1000))
print("Training Throughput  : {0:.3f} img/s".format((batch_size * num_trials) / (total_backward_time)))



