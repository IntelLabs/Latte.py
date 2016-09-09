import numpy as np
import caffe

CONFIG = 'CityScapes/config/infinet/train.prototxt'
MODEL = 'CityScapes/model/infinet/train_iter_10000.caffemodel'

net = caffe.Net(CONFIG, MODEL)

[(k,v.data.shape) for k,v in net.blobs.items()]
