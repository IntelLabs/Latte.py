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
#import caffe
from latte import *
from latte.math import compute_softmax_loss, softmax_loss_backprop
import time
from latte.solvers import sgd_update
import os




batch_size = 128
net = Net(batch_size)
channels=3
height=224
width=224

#caffe_root ='/nfs_home/avenkat/caffe/'
#caffe.set_mode_cpu()
 
data = MemoryDataLayer(net, (3, 224, 224))
conv1_7x7_s2 = ConvLayer(net, data, num_filters=64, kernel=7, stride=2, pad=3)
conv1_relu_7x7 = ReLULayer(net, conv1_7x7_s2)
pool1_3x3_s2 = MaxPoolingLayer(net, conv1_7x7_s2, kernel=3, stride=2, pad=0)

pool1_norm1 = LRNLayer(net, pool1_3x3_s2, n=5, beta=0.75, alpha=0.0001, k=1.0)
conv2_3x3_reduce = ConvLayer(net, pool1_norm1, num_filters=64, kernel=1, stride=1, pad=0)
#conv2_3x3_reduce = ConvLayer(net, pool1_3x3_s2, num_filters=64, kernel=1, stride=1, pad=0)

conv2_relu_3x3_reduce = ReLULayer(net, conv2_3x3_reduce)
conv2_3x3 = ConvLayer(net, conv2_relu_3x3_reduce, num_filters=192, kernel=3, stride=1, pad=1)
conv2_relu_3x3 = ReLULayer(net, conv2_3x3)

conv2_norm2 = LRNLayer(net, conv2_relu_3x3, n=5, beta=0.75, alpha=0.0001, k=1.0)
pool2_3x3_s2 = MaxPoolingLayer(net, conv2_norm2, kernel=3, stride=2, pad=0)
#pool2_3x3_s2 = MaxPoolingLayer(net, conv2_relu_3x3, kernel=3, stride=2, pad=0)

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

net.compile() 

num_train = 1
 
 
#train_batches = [i for i in range(0, num_train, batch_size)]
total_forward_time = 0.0
total_backward_time = 0.0
epoch_size = 2
timing_info = True
 
#train_images = np.random.randint(0, 255, (num_train, 3, 306, 306)).astype(np.float32)
#train_labels = np.random.randint(0, 255, (num_train, 1, 306, 306)).astype(np.float32)

label = np.zeros((batch_size,1), dtype=int)

 
print("Training ...")
#forward_time = 0.0
#backward_time = 0.0
'''
labels_file = caffe_root + 'data/ilsvrc12/synset_words.txt'
if not os.path.exists(labels_file):
    os.system("...data/ilsvrc12/get_ilsvrc_aux.sh")
labels = np.loadtxt(labels_file, str, delimiter='\t')
solver = caffe.SGDSolver('solver.prototxt')
net2 = solver.net
'''

for epoch in range(epoch_size):
 
 
    for i in range(1):
        forward_time = 0.0
        backward_time = 0.0
 
        data.set_value(np.random.rand(batch_size, 3, 224, 224))
 
        #train_data = train_images[n:n+batch_size]
        #train_label = train_labels[n:n+batch_size]
        #data.set_value(train_data)
        #label.set_value(train_label)
        #data.set_value(net2.blobs["data"].data)

        #net2.forward()

        #print(net2.blobs["loss1/classifier"].diff)

        #net2.backward()
            
        #if i == 0:
             #copy_weights(net2,convs)
             #label = net2.blobs["label"].data
             #label = label.reshape(batch_size,1)
             #data.set_value(net2.blobs["data"].data)

        #print(label[0][0])

        t = time.time()
        net.forward()
        forward_time += time.time() - t

        #forward_time += time.time() - t
        if timing_info:
            print("Iteration {} -- ".format(epoch))
            print("FP                   : {0:.3f} ms".format(forward_time * 1000))
            print("BP+WU                : {0:.3f} ms".format(backward_time * 1000))
            print("Total Time           : {0:.3f} ms".format((forward_time+backward_time)*1000))
            print("")
 
        total_forward_time += forward_time
        total_backward_time += backward_time
 
print("Total FP                   : {0:.3f} ms".format(total_forward_time * 1000))
print("Total Inference Throughput : {0:.3f} images/second".format((num_train * epoch_size*batch_size)/(total_forward_time)))

