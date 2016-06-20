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
