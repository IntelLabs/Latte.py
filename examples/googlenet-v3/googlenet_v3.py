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
 
data = MemoryDataLayer(net, (3, 224, 224))


conv_conv2d = ConvLayer(net, data, num_filters=32, kernel=3, stride=2, pad=0)
conv_relu = ReLULayer(net, conv_conv2d_bn)
conv_1_1_conv2d = ConvLayer(net, conv_conv2d_relu, num_filters=32, kernel=3, stride=1, pad=0)
conv_1_1_relu = ReLULayer(net, conv_1_1_conv2d_bn)
conv_2_2_conv2d = ConvLayer(net, conv_1_1_conv2d_relu, num_filters=64, kernel=3, stride=1, pad=1)
conv_2_2_relu = ReLULayer(net, conv_2_2_conv2d_bn)
pool = MaxPoolingLayer(net, conv_2_2_conv2d_relu, kernel=3, stride=2, pad=0)
conv_3_3_conv2d = FullyConnectedLayer(net, pool, 80)
conv_3_3_relu = ReLULayer(net, conv_3_3_conv2d_bn)
conv_4_4_conv2d = ConvLayer(net, conv_3_3_conv2d_relu, num_filters=192, kernel=3, stride=1, pad=0)
conv_4_4_relu = ReLULayer(net, conv_4_4_conv2d_bn)
pool1 = MaxPoolingLayer(net, conv_4_4_conv2d_relu, kernel=3, stride=2, pad=0)
mixed_conv_conv2d = FullyConnectedLayer(net, pool1, 64)
mixed_conv_relu = ReLULayer(net, mixed_conv_conv2d_bn)
mixed_tower_conv_conv2d = FullyConnectedLayer(net, pool1, 48)
mixed_tower_conv_relu = ReLULayer(net, mixed_tower_conv_conv2d_bn)
mixed_tower_conv_1_conv2d = ConvLayer(net, mixed_tower_conv_conv2d_relu, num_filters=64, kernel=5, stride=1, pad=2)
mixed_tower_conv_1_relu = ReLULayer(net, mixed_tower_conv_1_conv2d_bn)
mixed_tower_1_conv_conv2d = FullyConnectedLayer(net, pool1, 64)
mixed_tower_1_conv_relu = ReLULayer(net, mixed_tower_1_conv_conv2d_bn)
mixed_tower_1_conv_1_conv2d = ConvLayer(net, mixed_tower_1_conv_conv2d_relu, num_filters=96, kernel=3, stride=1, pad=1)
mixed_tower_1_conv_1_relu = ReLULayer(net, mixed_tower_1_conv_1_conv2d_bn)
mixed_tower_1_conv_2_conv2d = ConvLayer(net, mixed_tower_1_conv_1_conv2d_relu, num_filters=96, kernel=3, stride=1, pad=1)
mixed_tower_1_conv_2_relu = ReLULayer(net, mixed_tower_1_conv_2_conv2d_bn)
AVE_pool_mixed_pool = MeanPoolingLayer(net, pool1, kernel=3, stride=1, pad=1)
mixed_tower_2_conv_conv2d = FullyConnectedLayer(net, AVE_pool_mixed_pool, 32)
mixed_tower_2_conv_relu = ReLULayer(net, mixed_tower_2_conv_conv2d_bn)
ch_concat_mixed_chconcat = ConcatLayer(net, [mixed_conv_conv2d_relu, mixed_tower_conv_1_conv2d_relu, mixed_tower_1_conv_2_conv2d_relu, mixed_tower_2_conv_conv2d_relu])
mixed_1_conv_conv2d = FullyConnectedLayer(net, ch_concat_mixed_chconcat, 64)
mixed_1_conv_relu = ReLULayer(net, mixed_1_conv_conv2d_bn)
mixed_1_tower_conv_conv2d = FullyConnectedLayer(net, ch_concat_mixed_chconcat, 48)
mixed_1_tower_conv_relu = ReLULayer(net, mixed_1_tower_conv_conv2d_bn)
mixed_1_tower_conv_1_conv2d = ConvLayer(net, mixed_1_tower_conv_conv2d_relu, num_filters=64, kernel=5, stride=1, pad=2)
mixed_1_tower_conv_1_relu = ReLULayer(net, mixed_1_tower_conv_1_conv2d_bn)
mixed_1_tower_1_conv_conv2d = FullyConnectedLayer(net, ch_concat_mixed_chconcat, 64)
mixed_1_tower_1_conv_relu = ReLULayer(net, mixed_1_tower_1_conv_conv2d_bn)
mixed_1_tower_1_conv_1_conv2d = ConvLayer(net, mixed_1_tower_1_conv_conv2d_relu, num_filters=96, kernel=3, stride=1, pad=1)
mixed_1_tower_1_conv_1_relu = ReLULayer(net, mixed_1_tower_1_conv_1_conv2d_bn)
mixed_1_tower_1_conv_2_conv2d = ConvLayer(net, mixed_1_tower_1_conv_1_conv2d_relu, num_filters=96, kernel=3, stride=1, pad=1)
mixed_1_tower_1_conv_2_relu = ReLULayer(net, mixed_1_tower_1_conv_2_conv2d_bn)
AVE_pool_mixed_1_pool = MeanPoolingLayer(net, ch_concat_mixed_chconcat, kernel=3, stride=1, pad=1)
mixed_1_tower_2_conv_conv2d = FullyConnectedLayer(net, AVE_pool_mixed_1_pool, 64)
mixed_1_tower_2_conv_relu = ReLULayer(net, mixed_1_tower_2_conv_conv2d_bn)
ch_concat_mixed_1_chconcat = ConcatLayer(net, [mixed_1_conv_conv2d_relu, mixed_1_tower_conv_1_conv2d_relu, mixed_1_tower_1_conv_2_conv2d_relu, mixed_1_tower_2_conv_conv2d_relu])
mixed_2_conv_conv2d = FullyConnectedLayer(net, ch_concat_mixed_1_chconcat, 64)
mixed_2_conv_relu = ReLULayer(net, mixed_2_conv_conv2d_bn)
mixed_2_tower_conv_conv2d = FullyConnectedLayer(net, ch_concat_mixed_1_chconcat, 48)
mixed_2_tower_conv_relu = ReLULayer(net, mixed_2_tower_conv_conv2d_bn)
mixed_2_tower_conv_1_conv2d = ConvLayer(net, mixed_2_tower_conv_conv2d_relu, num_filters=64, kernel=5, stride=1, pad=2)
mixed_2_tower_conv_1_relu = ReLULayer(net, mixed_2_tower_conv_1_conv2d_bn)
mixed_2_tower_1_conv_conv2d = FullyConnectedLayer(net, ch_concat_mixed_1_chconcat, 64)
mixed_2_tower_1_conv_relu = ReLULayer(net, mixed_2_tower_1_conv_conv2d_bn)
mixed_2_tower_1_conv_1_conv2d = ConvLayer(net, mixed_2_tower_1_conv_conv2d_relu, num_filters=96, kernel=3, stride=1, pad=1)
mixed_2_tower_1_conv_1_relu = ReLULayer(net, mixed_2_tower_1_conv_1_conv2d_bn)
mixed_2_tower_1_conv_2_conv2d = ConvLayer(net, mixed_2_tower_1_conv_1_conv2d_relu, num_filters=96, kernel=3, stride=1, pad=1)
mixed_2_tower_1_conv_2_relu = ReLULayer(net, mixed_2_tower_1_conv_2_conv2d_bn)
AVE_pool_mixed_2_pool = MeanPoolingLayer(net, ch_concat_mixed_1_chconcat, kernel=3, stride=1, pad=1)
mixed_2_tower_2_conv_conv2d = FullyConnectedLayer(net, AVE_pool_mixed_2_pool, 64)
mixed_2_tower_2_conv_relu = ReLULayer(net, mixed_2_tower_2_conv_conv2d_bn)
ch_concat_mixed_2_chconcat = ConcatLayer(net, [mixed_2_conv_conv2d_relu, mixed_2_tower_conv_1_conv2d_relu, mixed_2_tower_1_conv_2_conv2d_relu, mixed_2_tower_2_conv_conv2d_relu])
mixed_3_conv_conv2d = ConvLayer(net, ch_concat_mixed_2_chconcat, num_filters=384, kernel=3, stride=2, pad=0)
mixed_3_conv_relu = ReLULayer(net, mixed_3_conv_conv2d_bn)
mixed_3_tower_conv_conv2d = FullyConnectedLayer(net, ch_concat_mixed_2_chconcat, 64)
mixed_3_tower_conv_relu = ReLULayer(net, mixed_3_tower_conv_conv2d_bn)
mixed_3_tower_conv_1_conv2d = ConvLayer(net, mixed_3_tower_conv_conv2d_relu, num_filters=96, kernel=3, stride=1, pad=1)
mixed_3_tower_conv_1_relu = ReLULayer(net, mixed_3_tower_conv_1_conv2d_bn)
mixed_3_tower_conv_2_conv2d = ConvLayer(net, mixed_3_tower_conv_1_conv2d_relu, num_filters=96, kernel=3, stride=2, pad=0)
mixed_3_tower_conv_2_relu = ReLULayer(net, mixed_3_tower_conv_2_conv2d_bn)
max_pool_mixed_3_pool = MaxPoolingLayer(net, ch_concat_mixed_2_chconcat, kernel=3, stride=2, pad=0)
ch_concat_mixed_3_chconcat = ConcatLayer(net, [max_pool_mixed_3_pool, mixed_3_conv_conv2d_relu, mixed_3_tower_conv_2_conv2d_relu])
mixed_4_conv_conv2d = FullyConnectedLayer(net, ch_concat_mixed_3_chconcat, 192)
mixed_4_conv_relu = ReLULayer(net, mixed_4_conv_conv2d_bn)
mixed_4_tower_conv_conv2d = FullyConnectedLayer(net, ch_concat_mixed_3_chconcat, 128)
mixed_4_tower_conv_relu = ReLULayer(net, mixed_4_tower_conv_conv2d_bn)
mixed_4_tower_conv_1_conv2d = ConvLayer(net, mixed_4_tower_conv_conv2d_relu, num_filters=128, stride=1, pad=0)
mixed_4_tower_conv_1_relu = ReLULayer(net, mixed_4_tower_conv_1_conv2d_bn)
mixed_4_tower_conv_2_conv2d = ConvLayer(net, mixed_4_tower_conv_1_conv2d_relu, num_filters=192, stride=1, pad=0)
mixed_4_tower_conv_2_relu = ReLULayer(net, mixed_4_tower_conv_2_conv2d_bn)
mixed_4_tower_1_conv_conv2d = FullyConnectedLayer(net, ch_concat_mixed_3_chconcat, 128)
mixed_4_tower_1_conv_relu = ReLULayer(net, mixed_4_tower_1_conv_conv2d_bn)
mixed_4_tower_1_conv_1_conv2d = ConvLayer(net, mixed_4_tower_1_conv_conv2d_relu, num_filters=128, stride=1, pad=0)
mixed_4_tower_1_conv_1_relu = ReLULayer(net, mixed_4_tower_1_conv_1_conv2d_bn)
mixed_4_tower_1_conv_2_conv2d = ConvLayer(net, mixed_4_tower_1_conv_1_conv2d_relu, num_filters=128, stride=1, pad=0)
mixed_4_tower_1_conv_2_relu = ReLULayer(net, mixed_4_tower_1_conv_2_conv2d_bn)
mixed_4_tower_1_conv_3_conv2d = ConvLayer(net, mixed_4_tower_1_conv_2_conv2d_relu, num_filters=128, stride=1, pad=0)
mixed_4_tower_1_conv_3_relu = ReLULayer(net, mixed_4_tower_1_conv_3_conv2d_bn)
mixed_4_tower_1_conv_4_conv2d = ConvLayer(net, mixed_4_tower_1_conv_3_conv2d_relu, num_filters=192, stride=1, pad=0)
mixed_4_tower_1_conv_4_relu = ReLULayer(net, mixed_4_tower_1_conv_4_conv2d_bn)
AVE_pool_mixed_4_pool = MeanPoolingLayer(net, ch_concat_mixed_3_chconcat, kernel=3, stride=1, pad=1)
mixed_4_tower_2_conv_conv2d = FullyConnectedLayer(net, AVE_pool_mixed_4_pool, 192)
mixed_4_tower_2_conv_relu = ReLULayer(net, mixed_4_tower_2_conv_conv2d_bn)
ch_concat_mixed_4_chconcat = ConcatLayer(net, [mixed_4_conv_conv2d_relu, mixed_4_tower_conv_2_conv2d_relu, mixed_4_tower_1_conv_4_conv2d_relu, mixed_4_tower_2_conv_conv2d_relu])
mixed_5_conv_conv2d = FullyConnectedLayer(net, ch_concat_mixed_4_chconcat, 192)
mixed_5_conv_relu = ReLULayer(net, mixed_5_conv_conv2d_bn)
mixed_5_tower_conv_conv2d = FullyConnectedLayer(net, ch_concat_mixed_4_chconcat, 160)
mixed_5_tower_conv_relu = ReLULayer(net, mixed_5_tower_conv_conv2d_bn)
mixed_5_tower_conv_1_conv2d = ConvLayer(net, mixed_5_tower_conv_conv2d_relu, num_filters=160, stride=1, pad=0)
mixed_5_tower_conv_1_relu = ReLULayer(net, mixed_5_tower_conv_1_conv2d_bn)
mixed_5_tower_conv_2_conv2d = ConvLayer(net, mixed_5_tower_conv_1_conv2d_relu, num_filters=192, stride=1, pad=0)
mixed_5_tower_conv_2_relu = ReLULayer(net, mixed_5_tower_conv_2_conv2d_bn)
mixed_5_tower_1_conv_conv2d = FullyConnectedLayer(net, ch_concat_mixed_4_chconcat, 160)
mixed_5_tower_1_conv_relu = ReLULayer(net, mixed_5_tower_1_conv_conv2d_bn)
mixed_5_tower_1_conv_1_conv2d = ConvLayer(net, mixed_5_tower_1_conv_conv2d_relu, num_filters=160, stride=1, pad=0)
mixed_5_tower_1_conv_1_relu = ReLULayer(net, mixed_5_tower_1_conv_1_conv2d_bn)
mixed_5_tower_1_conv_2_conv2d = ConvLayer(net, mixed_5_tower_1_conv_1_conv2d_relu, num_filters=160, stride=1, pad=0)
mixed_5_tower_1_conv_2_relu = ReLULayer(net, mixed_5_tower_1_conv_2_conv2d_bn)
mixed_5_tower_1_conv_3_conv2d = ConvLayer(net, mixed_5_tower_1_conv_2_conv2d_relu, num_filters=160, stride=1, pad=0)
mixed_5_tower_1_conv_3_relu = ReLULayer(net, mixed_5_tower_1_conv_3_conv2d_bn)
mixed_5_tower_1_conv_4_conv2d = ConvLayer(net, mixed_5_tower_1_conv_3_conv2d_relu, num_filters=192, stride=1, pad=0)
mixed_5_tower_1_conv_4_relu = ReLULayer(net, mixed_5_tower_1_conv_4_conv2d_bn)
AVE_pool_mixed_5_pool = MeanPoolingLayer(net, ch_concat_mixed_4_chconcat, kernel=3, stride=1, pad=1)
mixed_5_tower_2_conv_conv2d = FullyConnectedLayer(net, AVE_pool_mixed_5_pool, 192)
mixed_5_tower_2_conv_relu = ReLULayer(net, mixed_5_tower_2_conv_conv2d_bn)
ch_concat_mixed_5_chconcat = ConcatLayer(net, [mixed_5_conv_conv2d_relu, mixed_5_tower_conv_2_conv2d_relu, mixed_5_tower_1_conv_4_conv2d_relu, mixed_5_tower_2_conv_conv2d_relu])
mixed_6_conv_conv2d = FullyConnectedLayer(net, ch_concat_mixed_5_chconcat, 192)
mixed_6_conv_relu = ReLULayer(net, mixed_6_conv_conv2d_bn)
mixed_6_tower_conv_conv2d = FullyConnectedLayer(net, ch_concat_mixed_5_chconcat, 160)
mixed_6_tower_conv_relu = ReLULayer(net, mixed_6_tower_conv_conv2d_bn)
mixed_6_tower_conv_1_conv2d = ConvLayer(net, mixed_6_tower_conv_conv2d_relu, num_filters=160, stride=1, pad=0)
mixed_6_tower_conv_1_relu = ReLULayer(net, mixed_6_tower_conv_1_conv2d_bn)
mixed_6_tower_conv_2_conv2d = ConvLayer(net, mixed_6_tower_conv_1_conv2d_relu, num_filters=192, stride=1, pad=0)
mixed_6_tower_conv_2_relu = ReLULayer(net, mixed_6_tower_conv_2_conv2d_bn)
mixed_6_tower_1_conv_conv2d = FullyConnectedLayer(net, ch_concat_mixed_5_chconcat, 160)
mixed_6_tower_1_conv_relu = ReLULayer(net, mixed_6_tower_1_conv_conv2d_bn)
mixed_6_tower_1_conv_1_conv2d = ConvLayer(net, mixed_6_tower_1_conv_conv2d_relu, num_filters=160, stride=1, pad=0)
mixed_6_tower_1_conv_1_relu = ReLULayer(net, mixed_6_tower_1_conv_1_conv2d_bn)
mixed_6_tower_1_conv_2_conv2d = ConvLayer(net, mixed_6_tower_1_conv_1_conv2d_relu, num_filters=160, stride=1, pad=0)
mixed_6_tower_1_conv_2_relu = ReLULayer(net, mixed_6_tower_1_conv_2_conv2d_bn)
mixed_6_tower_1_conv_3_conv2d = ConvLayer(net, mixed_6_tower_1_conv_2_conv2d_relu, num_filters=160, stride=1, pad=0)
mixed_6_tower_1_conv_3_relu = ReLULayer(net, mixed_6_tower_1_conv_3_conv2d_bn)
mixed_6_tower_1_conv_4_conv2d = ConvLayer(net, mixed_6_tower_1_conv_3_conv2d_relu, num_filters=192, stride=1, pad=0)
mixed_6_tower_1_conv_4_relu = ReLULayer(net, mixed_6_tower_1_conv_4_conv2d_bn)
AVE_pool_mixed_6_pool = MeanPoolingLayer(net, ch_concat_mixed_5_chconcat, kernel=3, stride=1, pad=1)
mixed_6_tower_2_conv_conv2d = FullyConnectedLayer(net, AVE_pool_mixed_6_pool, 192)
mixed_6_tower_2_conv_relu = ReLULayer(net, mixed_6_tower_2_conv_conv2d_bn)
ch_concat_mixed_6_chconcat = ConcatLayer(net, [mixed_6_conv_conv2d_relu, mixed_6_tower_conv_2_conv2d_relu, mixed_6_tower_1_conv_4_conv2d_relu, mixed_6_tower_2_conv_conv2d_relu])
mixed_7_conv_conv2d = FullyConnectedLayer(net, ch_concat_mixed_6_chconcat, 192)
mixed_7_conv_relu = ReLULayer(net, mixed_7_conv_conv2d_bn)
mixed_7_tower_conv_conv2d = FullyConnectedLayer(net, ch_concat_mixed_6_chconcat, 192)
mixed_7_tower_conv_relu = ReLULayer(net, mixed_7_tower_conv_conv2d_bn)
mixed_7_tower_conv_1_conv2d = ConvLayer(net, mixed_7_tower_conv_conv2d_relu, num_filters=192, stride=1, pad=0)
mixed_7_tower_conv_1_relu = ReLULayer(net, mixed_7_tower_conv_1_conv2d_bn)
mixed_7_tower_conv_2_conv2d = ConvLayer(net, mixed_7_tower_conv_1_conv2d_relu, num_filters=192, stride=1, pad=0)
mixed_7_tower_conv_2_relu = ReLULayer(net, mixed_7_tower_conv_2_conv2d_bn)
mixed_7_tower_1_conv_conv2d = FullyConnectedLayer(net, ch_concat_mixed_6_chconcat, 192)
mixed_7_tower_1_conv_relu = ReLULayer(net, mixed_7_tower_1_conv_conv2d_bn)
mixed_7_tower_1_conv_1_conv2d = ConvLayer(net, mixed_7_tower_1_conv_conv2d_relu, num_filters=192, stride=1, pad=0)
mixed_7_tower_1_conv_1_relu = ReLULayer(net, mixed_7_tower_1_conv_1_conv2d_bn)
mixed_7_tower_1_conv_2_conv2d = ConvLayer(net, mixed_7_tower_1_conv_1_conv2d_relu, num_filters=192, stride=1, pad=0)
mixed_7_tower_1_conv_2_relu = ReLULayer(net, mixed_7_tower_1_conv_2_conv2d_bn)
mixed_7_tower_1_conv_3_conv2d = ConvLayer(net, mixed_7_tower_1_conv_2_conv2d_relu, num_filters=192, stride=1, pad=0)
mixed_7_tower_1_conv_3_relu = ReLULayer(net, mixed_7_tower_1_conv_3_conv2d_bn)
mixed_7_tower_1_conv_4_conv2d = ConvLayer(net, mixed_7_tower_1_conv_3_conv2d_relu, num_filters=192, stride=1, pad=0)
mixed_7_tower_1_conv_4_relu = ReLULayer(net, mixed_7_tower_1_conv_4_conv2d_bn)
AVE_pool_mixed_7_pool = MeanPoolingLayer(net, ch_concat_mixed_6_chconcat, kernel=3, stride=1, pad=1)
mixed_7_tower_2_conv_conv2d = FullyConnectedLayer(net, AVE_pool_mixed_7_pool, 192)
mixed_7_tower_2_conv_relu = ReLULayer(net, mixed_7_tower_2_conv_conv2d_bn)
ch_concat_mixed_7_chconcat = ConcatLayer(net, [mixed_7_conv_conv2d_relu, mixed_7_tower_conv_2_conv2d_relu, mixed_7_tower_1_conv_4_conv2d_relu, mixed_7_tower_2_conv_conv2d_relu])
mixed_8_tower_conv_conv2d = FullyConnectedLayer(net, ch_concat_mixed_7_chconcat, 192)
mixed_8_tower_conv_relu = ReLULayer(net, mixed_8_tower_conv_conv2d_bn)
mixed_8_tower_conv_1_conv2d = ConvLayer(net, mixed_8_tower_conv_conv2d_relu, num_filters=320, kernel=3, stride=2, pad=0)
mixed_8_tower_conv_1_relu = ReLULayer(net, mixed_8_tower_conv_1_conv2d_bn)
mixed_8_tower_1_conv_conv2d = FullyConnectedLayer(net, ch_concat_mixed_7_chconcat, 192)
mixed_8_tower_1_conv_relu = ReLULayer(net, mixed_8_tower_1_conv_conv2d_bn)
mixed_8_tower_1_conv_1_conv2d = ConvLayer(net, mixed_8_tower_1_conv_conv2d_relu, num_filters=192, stride=1, pad=0)
mixed_8_tower_1_conv_1_relu = ReLULayer(net, mixed_8_tower_1_conv_1_conv2d_bn)
mixed_8_tower_1_conv_2_conv2d = ConvLayer(net, mixed_8_tower_1_conv_1_conv2d_relu, num_filters=192, stride=1, pad=0)
mixed_8_tower_1_conv_2_relu = ReLULayer(net, mixed_8_tower_1_conv_2_conv2d_bn)
mixed_8_tower_1_conv_3_conv2d = ConvLayer(net, mixed_8_tower_1_conv_2_conv2d_relu, num_filters=192, kernel=3, stride=2, pad=0)
mixed_8_tower_1_conv_3_relu = ReLULayer(net, mixed_8_tower_1_conv_3_conv2d_bn)
MAX_pool_mixed_8_pool = MaxPoolingLayer(net, ch_concat_mixed_7_chconcat, kernel=3, stride=2, pad=0)
ch_concat_mixed_8_chconcat = ConcatLayer(net, [mixed_8_tower_conv_1_conv2d_relu, mixed_8_tower_1_conv_3_conv2d_relu, MAX_pool_mixed_8_pool])
mixed_9_conv_conv2d = ConvLayer(net, ch_concat_mixed_8_chconcat, num_filters=320, stride=1, pad=0)
mixed_9_conv_relu = ReLULayer(net, mixed_9_conv_conv2d_bn)
mixed_9_tower_conv_conv2d = FullyConnectedLayer(net, ch_concat_mixed_8_chconcat, 384)
mixed_9_tower_conv_relu = ReLULayer(net, mixed_9_tower_conv_conv2d_bn)
mixed_9_tower_mixed_conv_conv2d = ConvLayer(net, mixed_9_tower_conv_conv2d_relu, num_filters=384, stride=1, pad=0)
mixed_9_tower_mixed_conv_relu = ReLULayer(net, mixed_9_tower_mixed_conv_conv2d_bn)
mixed_9_tower_mixed_conv_1_conv2d = ConvLayer(net, mixed_9_tower_conv_conv2d_relu, num_filters=384, stride=1, pad=0)
mixed_9_tower_mixed_conv_1_relu = ReLULayer(net, mixed_9_tower_mixed_conv_1_conv2d_bn)
mixed_9_tower_1_conv_conv2d = FullyConnectedLayer(net, ch_concat_mixed_8_chconcat, 448)
mixed_9_tower_1_conv_relu = ReLULayer(net, mixed_9_tower_1_conv_conv2d_bn)
mixed_9_tower_1_conv_1_conv2d = ConvLayer(net, mixed_9_tower_1_conv_conv2d_relu, num_filters=384, kernel=3, stride=1, pad=1)
mixed_9_tower_1_conv_1_relu = ReLULayer(net, mixed_9_tower_1_conv_1_conv2d_bn)
mixed_9_tower_1_mixed_conv_conv2d = ConvLayer(net, mixed_9_tower_1_conv_1_conv2d_relu, num_filters=384, stride=1, pad=0)
mixed_9_tower_1_mixed_conv_relu = ReLULayer(net, mixed_9_tower_1_mixed_conv_conv2d_bn)
mixed_9_tower_1_mixed_conv_1_conv2d = ConvLayer(net, mixed_9_tower_1_conv_1_conv2d_relu, num_filters=384, stride=1, pad=0)
mixed_9_tower_1_mixed_conv_1_relu = ReLULayer(net, mixed_9_tower_1_mixed_conv_1_conv2d_bn)
AVE_pool_mixed_9_pool = MeanPoolingLayer(net, ch_concat_mixed_8_chconcat, kernel=3, stride=1, pad=1)
mixed_9_tower_2_conv_conv2d = FullyConnectedLayer(net, AVE_pool_mixed_9_pool, 192)
mixed_9_tower_2_conv_relu = ReLULayer(net, mixed_9_tower_2_conv_conv2d_bn)
ch_concat_mixed_9_chconcat = ConcatLayer(net, [mixed_9_conv_conv2d_relu, mixed_9_tower_mixed_conv_conv2d_relu, mixed_9_tower_mixed_conv_1_conv2d_relu, mixed_9_tower_1_mixed_conv_conv2d_relu, mixed_9_tower_1_mixed_conv_1_conv2d_relu, mixed_9_tower_2_conv_conv2d_relu])
mixed_10_conv_conv2d = ConvLayer(net, ch_concat_mixed_9_chconcat, num_filters=320, stride=1, pad=0)
mixed_10_conv_relu = ReLULayer(net, mixed_10_conv_conv2d_bn)
mixed_10_tower_conv_conv2d = FullyConnectedLayer(net, ch_concat_mixed_9_chconcat, 384)
mixed_10_tower_conv_relu = ReLULayer(net, mixed_10_tower_conv_conv2d_bn)
mixed_10_tower_mixed_conv_conv2d = ConvLayer(net, mixed_10_tower_conv_conv2d_relu, num_filters=384, stride=1, pad=0)
mixed_10_tower_mixed_conv_relu = ReLULayer(net, mixed_10_tower_mixed_conv_conv2d_bn)
mixed_10_tower_mixed_conv_1_conv2d = ConvLayer(net, mixed_10_tower_conv_conv2d_relu, num_filters=384, stride=1, pad=0)
mixed_10_tower_mixed_conv_1_relu = ReLULayer(net, mixed_10_tower_mixed_conv_1_conv2d_bn)
mixed_10_tower_1_conv_conv2d = FullyConnectedLayer(net, ch_concat_mixed_9_chconcat, 448)
mixed_10_tower_1_conv_relu = ReLULayer(net, mixed_10_tower_1_conv_conv2d_bn)
mixed_10_tower_1_conv_1_conv2d = ConvLayer(net, mixed_10_tower_1_conv_conv2d_relu, num_filters=384, kernel=3, stride=1, pad=1)
mixed_10_tower_1_conv_1_relu = ReLULayer(net, mixed_10_tower_1_conv_1_conv2d_bn)
mixed_10_tower_1_mixed_conv_conv2d = ConvLayer(net, mixed_10_tower_1_conv_1_conv2d_relu, num_filters=384, stride=1, pad=0)
mixed_10_tower_1_mixed_conv_relu = ReLULayer(net, mixed_10_tower_1_mixed_conv_conv2d_bn)
mixed_10_tower_1_mixed_conv_1_conv2d = ConvLayer(net, mixed_10_tower_1_conv_1_conv2d_relu, num_filters=384, stride=1, pad=0)
mixed_10_tower_1_mixed_conv_1_relu = ReLULayer(net, mixed_10_tower_1_mixed_conv_1_conv2d_bn)
MAX_pool_mixed_10_pool = MaxPoolingLayer(net, ch_concat_mixed_9_chconcat, kernel=3, stride=1, pad=1)
mixed_10_tower_2_conv_conv2d = FullyConnectedLayer(net, MAX_pool_mixed_10_pool, 192)
mixed_10_tower_2_conv_relu = ReLULayer(net, mixed_10_tower_2_conv_conv2d_bn)
ch_concat_mixed_10_chconcat = ConcatLayer(net, [mixed_10_conv_conv2d_relu, mixed_10_tower_mixed_conv_conv2d_relu, mixed_10_tower_mixed_conv_1_conv2d_relu, mixed_10_tower_1_mixed_conv_conv2d_relu, mixed_10_tower_1_mixed_conv_1_conv2d_relu, mixed_10_tower_2_conv_conv2d_relu])
global_pool = MeanPoolingLayer(net, ch_concat_mixed_10_chconcat, kernel=8, stride=1, pad=0)
#drop = DropoutLayer(net, global_pool, ratio=0.8)
#fc1 =  FullyConnectedLayer(net, flatten, 1000)
#loss = SoftmaxLossLayer(net, fc1, label)



net.compile()
 
 
data_value = np.random.rand(batch_size, channels, height, width)
 
data.set_value(data_value)
#print("Finished Copying\n")
net.forward()
net.forward()
net.forward()
#print("Finished forward computation\n") 
 
total_forward_time=0.0
timing_info = False
for epoch in range(10):
 
 
    for i in range(1):
        forward_time = 0.0
        backward_time = 0.0
 
 
        t = time.time()
        net.forward()
        forward_time += time.time() - t
 
        if timing_info:
            print("Iteration {} -- ".format(epoch))
            print("FP                   : {0:.3f} ms".format(forward_time * 1000))
 
        total_forward_time += forward_time
 
print("Total FP                   : {0:.3f} ms".format(total_forward_time * 1000))
print("Total Inference Throughput : {0:.3f} images/second".format((10*batch_size)/(total_forward_time)))


