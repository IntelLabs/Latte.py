import caffe_pb2

#import sys, os
#sys.path.append('/home/avenkat/protobuf/python/lib')

from google.protobuf.text_format import Merge

net = caffe_pb2.NetParameter()
Merge((open("train_val.prototxt", 'r').read()), net)
s = ''
for layer in range(len(net.layer)):
    if net.layer[layer].type == "Convolution": #convolution    
        if len(net.layer[layer].convolution_param.kernel_size) != 0 and net.layer[layer].convolution_param.kernel_size[0] == 1:
            s += str(net.layer[layer].name) + ' = FullyConnectedLayer(net, ' + str(net.layer[layer].bottom[0]) +\
             ', ' + str(net.layer[layer].convolution_param.num_output)
        else:    
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
            else:
                s+= ', pad=0'

        s += ')\n'

    elif net.layer[layer].type == "InnerProduct": #InnerProduct/FullyConnected   
        s += str(net.layer[layer].name) + ' =  FullyConnectedLayer(net, ' + str(net.layer[layer].bottom[0]) + \
            ', ' + str(net.layer[layer].inner_product_param.num_output) + ')\n'
           
    elif net.layer[layer].type == "ReLU": #relu
        s += str(net.layer[layer].name) + ' = ReLULayer(net, ' + str(net.layer[layer].bottom[0]) + ')\n'

    elif net.layer[layer].type == "SoftmaxWithLoss": #softmax
        s += str(net.layer[layer].name) + ' = SoftmaxLossLayer(net, ' + str(net.layer[layer].bottom[0]) +', ' + str(net.layer[layer].bottom[1]) + ')\n'

    elif net.layer[layer].type == "Softmax": #softmax 
        s += str(net.layer[layer].name) + ' = SoftmaxLossLayer(net, ' + str(net.layer[layer].bottom[0]) +')\n'

    elif net.layer[layer].type == "LRN": #Local Response Normalization
        s += str(net.layer[layer].name) + ' = LRNLayer(net, ' + str(net.layer[layer].bottom[0]) +', n=' + \
            str(net.layer[layer].lrn_param.local_size) + ', beta=' + str(net.layer[layer].lrn_param.beta) + ', alpha=' + \
            str(net.layer[layer].lrn_param.alpha) 
        if net.layer[layer].lrn_param.k != None:
            s+= ', k=' + str(net.layer[layer].lrn_param.k)
        else:
            s+= ', k=1'
        s += ')\n'
 
    elif net.layer[layer].type == "Pooling" and net.layer[layer].pooling_param.pool == caffe_pb2.PoolingParameter.MAX: # max pool
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

    elif net.layer[layer].type == "Pooling" and net.layer[layer].pooling_param.pool == caffe_pb2.PoolingParameter.AVE: # average pool
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
    elif net.layer[layer].type == "Concat":  
        s2 = '['
        for i in range(len(net.layer[layer].bottom)):
            s2 += str(net.layer[layer].bottom[i])
           
            if i < len(net.layer[layer].bottom) - 1:
                s2+=', '
            else:
               s2+=']'     
        s += str(net.layer[layer].name) + ' = ConcatLayer(net, ' + s2 
        s += ')\n'
    
    elif net.layer[layer].type == "Dropout": #dropout
        s += str(net.layer[layer].name) + ' = DropoutLayer(net, ' + str(net.layer[layer].bottom[0]) + \
            ', ratio=' + str(net.layer[layer].dropout_param.dropout_ratio) + ')\n'

    elif net.layer[layer].type == "Input":
        s+= str(net.layer[layer].name) + ' = MemoryDataLayer(net'
        #for i in range(len(net.layer[layer].input_param.shape)):
        #print (net.layer[layer].input_param.shape) 
        assert len(net.layer[layer].input_param.shape) == 1
        for i in range(len(net.layer[layer].input_param.shape[0].dim)):
            s+= ', ' + str(net.layer[layer].input_param.shape[0].dim[i])   
        s+=')\n'

    elif net.layer[layer].type != "Accuracy" and net.layer[layer].type != "Data":
        raise ValueError('layer type:{} unidentified'.format(net.layer[layer].type))        

print(s)
