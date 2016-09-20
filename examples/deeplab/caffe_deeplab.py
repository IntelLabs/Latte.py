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


