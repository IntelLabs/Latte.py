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
import caffe

CONFIG = 'CityScapes/config/infinet/train.prototxt'
MODEL = 'CityScapes/model/infinet/train_iter_10000.caffemodel'

net = caffe.Net(CONFIG, MODEL)

for k,v in net.blobs.items():
    print(k)
    print(v.data.shape)


net.forward()

print('DATA: {}'.format(net.blobs['data'].data.shape))
print(net.blobs['data'].data)
print('CONV1_1 WEIGHTS: {}'.format(net.params['conv1_1'][0].data.shape))
print(net.params['conv1_1'][0].data)
print('CONV1_1 BIAS: {}'.format(net.params['conv1_1'][1].data.shape))
print(net.params['conv1_1'][1].data)
print('CONV1_1: {}'.format(net.blobs['conv1_1'].data.shape))
print(net.blobs['conv1_1'].data)


