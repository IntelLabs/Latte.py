import caffe_pb2

#import sys, os
#sys.path.append('/home/avenkat/protobuf/python/lib')

from google.protobuf.text_format import Merge

net = caffe_pb2.NetParameter()
Merge((open("train_val.prototxt", 'r').read()), net)
s = ''
for layer in range(len(net.layer)):
    if net.layer[layer].type == "Convolution": #convolution    
        s += str(net.layer[layer].name) + ' = ConvLayer(net, ' + str(net.layer[layer].bottom[0]) + \
            ', num_filters=' + str(net.layer[layer].convolution_param.num_output)
        if len(net.layer[layer].convolution_param.kernel_size) != 0 :
            s += ', kernel=' + str(net.layer[layer].convolution_param.kernel_size[0])
        if len(net.layer[layer].convolution_param.stride) != 0:
            s+= ', stride=' + str(net.layer[layer].convolution_param.stride[0])
        else:
            s+= ', stride=1'
        if len(net.layer[layer].convolution_param.pad) != 0:
            s+= ', pad=' + str(net.layer[layer].convolution_param.pad[0])
        s += ')\n'

    if net.layer[layer].type == "InnerProduct": #InnerProduct/FullyConnected   
        s += str(net.layer[layer].name) + ' =  FullyConnectedLayer(net, ' + str(net.layer[layer].bottom[0]) + \
            ', ' + str(net.layer[layer].inner_product_param.num_output) + ')\n'
           
    if net.layer[layer].type == "ReLU": #relu
        s += str(net.layer[layer].name) + ' = ReLULayer(net, ' + str(net.layer[layer].bottom[0]) + ')\n'

    if net.layer[layer].type == "SoftmaxWithLoss": #softmax
        s += str(net.layer[layer].name) + ' = SoftmaxLossLayer(net, ' + str(net.layer[layer].bottom[0]) +', ' + str(net.layer[layer].bottom[1]) + ')\n'

    if net.layer[layer].type == "LRN": #Local Response Normalization
        s += str(net.layer[layer].name) + ' = LRNLayer(net, ' + str(net.layer[layer].bottom[0]) +', n=' + \
            str(net.layer[layer].lrn_param.local_size) + ', beta=' + str(net.layer[layer].lrn_param.beta) + ', alpha=' + \
            str(net.layer[layer].lrn_param.alpha) 
        if net.layer[layer].lrn_param.k != None:
            s+= ', k=' + str(net.layer[layer].lrn_param.k)
        else:
            s+= ', k=1'
        s += ')\n'
 
    if net.layer[layer].type == "Pooling" and net.layer[layer].pooling_param.pool == caffe_pb2.PoolingParameter.MAX: # max pool
        s += str(net.layer[layer].name) + ' = MaxPoolingLayer(net, ' + str(net.layer[layer].bottom[0])
        if net.layer[layer].pooling_param.kernel_size != None:
            s += ', kernel=' + str(net.layer[layer].pooling_param.kernel_size)
        if net.layer[layer].pooling_param.stride != None:
            s+= ', stride=' + str(net.layer[layer].pooling_param.stride)
        else:
            s+= ', stride=1'
        if net.layer[layer].pooling_param.pad != None:
            s+= ', pad=' + str(net.layer[layer].pooling_param.pad)
        s += ')\n'

    if net.layer[layer].type == "Pooling" and net.layer[layer].pooling_param.pool == caffe_pb2.PoolingParameter.AVE: # average pool
        s += str(net.layer[layer].name) + ' = MeanPoolingLayer(net, ' + str(net.layer[layer].bottom[0])
        if net.layer[layer].pooling_param.kernel_size != None:
            s += ', kernel=' + str(net.layer[layer].pooling_param.kernel_size)
        if net.layer[layer].pooling_param.stride != None:
            s+= ', stride=' + str(net.layer[layer].pooling_param.stride)
        else:
            s+= ', stride=1'
        if net.layer[layer].pooling_param.pad != None:
            s+= ', pad=' + str(net.layer[layer].pooling_param.pad)
        s += ')\n'
    

print(s)