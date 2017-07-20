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
import caffe_pb2
from google.protobuf.text_format import Merge

net = caffe_pb2.NetParameter()
Merge((open("CityScapes/config/infinet/train.prototxt", 'r').read()), net)
s = ''
for layer in range(len(net.layers)):
    if net.layers[layer].type == 4: #convolution    
       s += str(net.layers[layer].name) + ' = ConvLayer(net, ' + str(net.layers[layer].bottom[0]) + \
            ', num_filters=' + str(net.layers[layer].convolution_param.num_output)
       if net.layers[layer].convolution_param.kernel_size != None:
           s += ', kernel=' + str(net.layers[layer].convolution_param.kernel_size)
       if net.layers[layer].convolution_param.stride != None:
           s+= ', stride=' + str(net.layers[layer].convolution_param.stride)
       else:
           s+= ', stride=1'
       if net.layers[layer].convolution_param.pad != None:
           s+= ', pad=' + str(net.layers[layer].convolution_param.pad)
       s += ')\n'
       
    if net.layers[layer].type == 18: #relu
        s += str(net.layers[layer].name) + ' = ReLULayer(net, ' + str(net.layers[layer].bottom[0]) + ')\n'

    if net.layers[layer].type == 17: #pooling
        s += str(net.layers[layer].name) + ' = MaxPoolingLayer(net, ' + str(net.layers[layer].bottom[0])
        if net.layers[layer].pooling_param.kernel_size != None:
           s += ', kernel=' + str(net.layers[layer].pooling_param.kernel_size)
        if net.layers[layer].pooling_param.stride != None:
           s+= ', stride=' + str(net.layers[layer].pooling_param.stride)
        else:
           s+= ', stride=1'
        if net.layers[layer].pooling_param.pad != None:
           s+= ', pad=' + str(net.layers[layer].pooling_param.pad)
        s += ')\n'

    if net.layers[layer].type == 6: #dropout
        s += str(net.layers[layer].name) + ' = DropoutLayer(net, ' + str(net.layers[layer].bottom[0]) + \
            ', ratio=' + str(net.layers[layer].dropout_param.dropout_ratio) + ')\n'        

    if net.layers[layer].type == 41: #interp
        s += str(net.layers[layer].name) + ' = InterpolationLayer(net, ' + str(net.layers[layer].bottom[0]) + \
            ', pad=' + str(-int(net.layers[layer].interp_param.pad_beg)) + \
            ', resize_factor=' + str(net.layers[layer].interp_param.shrink_factor) + ')\n'        

print(s)
