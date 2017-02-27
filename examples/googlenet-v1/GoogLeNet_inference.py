import numpy as np
from latte import *
import caffe
from latte.math import compute_softmax_loss
import os
 
caffe_root ='/home/avenkat/caffe/'
 
caffe.set_mode_cpu()
 
model_def = './deploy.prototxt'
model_weights = caffe_root + 'models/bvlc_googlenet/bvlc_googlenet.caffemodel'
 
net2 = caffe.Net(model_def,      # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)   
#caffe_output = net2.forward()

convs = ["conv1/7x7_s2","conv2/3x3_reduce","conv2/3x3","inception_3a/1x1",\
"inception_3a/3x3_reduce","inception_3a/3x3","inception_3a/5x5_reduce","inception_3a/5x5",\
"inception_3a/pool_proj","inception_3b/1x1","inception_3b/3x3_reduce","inception_3b/3x3", \
"inception_3b/5x5_reduce","inception_3b/5x5","inception_3b/pool_proj","inception_4a/1x1", \
"inception_4a/3x3_reduce","inception_4a/3x3","inception_4a/5x5_reduce","inception_4a/5x5",\
"inception_4a/pool_proj","inception_4b/1x1","inception_4b/3x3_reduce","inception_4b/3x3",\
"inception_4b/5x5_reduce","inception_4b/5x5","inception_4b/pool_proj","inception_4c/1x1",\
"inception_4c/3x3_reduce","inception_4c/3x3","inception_4c/5x5_reduce","inception_4c/5x5",\
"inception_4c/pool_proj","inception_4d/1x1","inception_4d/3x3_reduce","inception_4d/3x3",\
"inception_4d/5x5_reduce","inception_4d/5x5","inception_4d/pool_proj","inception_4e/1x1",\
"inception_4e/3x3_reduce","inception_4e/3x3","inception_4e/5x5_reduce","inception_4e/5x5",\
"inception_4e/pool_proj","inception_5a/1x1","inception_5a/3x3_reduce","inception_5a/3x3",\
"inception_5a/5x5_reduce","inception_5a/5x5","inception_5a/pool_proj","inception_5b/1x1",\
"inception_5b/3x3_reduce","inception_5b/3x3","inception_5b/5x5_reduce","inception_5b/5x5",\
"inception_5b/pool_proj","loss3/classifier"]

#ioss3_classifier.set_bias(bias)
##print("HIGHLIGHT\n")
##print(net2.blobs["inception_5b/output"].data.shape)
###print(net2.blobs["inception_3a/1x1"].data.shape)

conv_params = {pr: (net2.params[pr][0].data, net2.params[pr][1].data) for pr in convs}

#for key, value in conv_params.items():
#    ##print(key, value)
#    ##print("\n")

#for i in net2.params:
#    if net2.params[i][0].type == 'Convolution':
#        ##print( net2.params[i].type)

###print(conv_params)
labels_file = caffe_root + 'data/ilsvrc12/synset_words.txt'
if not os.path.exists(labels_file):
    os.system("../data/ilsvrc12/get_ilsvrc_aux.sh")
#label = np.loadtxt(labels_file,str, comments='\t', delimiter='\t', converters=None, skiprows=0, usecols=None, unpack=False, ndmin=2) 
labels = np.loadtxt(labels_file, str, delimiter='\t')
###print(label.shape)
###print(label[0,0])
###print( 'output label:', labels[output_prob.argmax()])

batch_size = 10
net = Net(batch_size)
channels=3
height=224
width=224

# load the mean ImageNet image (as distributed with Caffe) for subtraction
mu = np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy')
mu = mu.mean(1).mean(1)  # average over pixels to obtain the mean (BGR) pixel values
test2 = list(zip('BGR',mu))
# zipped_list = test2[:]
# zipped_list_2 = list(test2)
 
##print ('mean-subtracted values:', test2)
 
# create transformer for the input called 'data'
transformer = caffe.io.Transformer({'data': net2.blobs['data'].data.shape})
 
transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
transformer.set_mean('data', mu)            # subtract the dataset-mean value in each channel
transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR
 
# set the size of the input (we can skip this if we're happy
#  with the default; we can also change it later, e.g., for different batch sizes)
#net.blobs['data'].reshape(50,        # batch size
#                          3,         # 3-channel (BGR) images
#                          227, 227)  # image size is 227x227
 
image = caffe.io.load_image(caffe_root + 'examples/images/cat.jpg')
 
transformed_image = transformer.preprocess('data', image)
##print(transformed_image.shape)

batched_image = np.zeros((batch_size,channels, height, width), dtype = np.float32)

for i in range (batch_size):
    np.copyto(batched_image[i],transformed_image)
   # ##print(batched_image[i])
   # ##print(transformed_image)    

##print (batched_image.shape)

# copy the image data into the memory allocated for the net
net2.blobs['data'].data[...] = transformed_image
 
net2.forward()
net2.backward()

##print("caffe weights\n") 
##print(conv_params["loss3/classifier"][0].size)

caffe_blobs = [(k, v.data.shape) for k, v in net2.blobs.items()]
##print(caffe_blobs)
data = MemoryDataLayer(net, (3, 224, 224))
conv1_7x7_s2 = ConvLayer(net, data, num_filters=64, kernel=7, stride=2, pad=3)
conv1_relu_7x7 = ReLULayer(net, conv1_7x7_s2)
pool1_3x3_s2 = MaxPoolingLayer(net, conv1_relu_7x7, kernel=3, stride=2, pad=0)

pool1_norm1 = LRNLayer(net, pool1_3x3_s2, n=5, beta=0.75, alpha=0.0001, k=1.0)
# Anand Changing to zero pad
conv2_3x3_reduce = ConvLayer(net, pool1_norm1, num_filters=64, kernel=1, stride=1, pad=0)
conv2_relu_3x3_reduce = ReLULayer(net, conv2_3x3_reduce)

conv2_3x3 = ConvLayer(net, conv2_relu_3x3_reduce, num_filters=192, kernel=3, stride=1, pad=1)
conv2_relu_3x3 = ReLULayer(net, conv2_3x3)
conv2_norm2 = LRNLayer(net, conv2_relu_3x3, n=5, beta=0.75, alpha=0.0001, k=1.0)

pool2_3x3_s2 = MaxPoolingLayer(net, conv2_norm2, kernel=3, stride=2, pad=0)

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
loss3_classifier =  FullyConnectedLayer(net, pool5_7x7_s1, 1000)
#prob = SoftmaxLossLayer(net, loss3_classifier)


net.compile() 
        
weights= conv_params["conv1/7x7_s2"][0]
bias = conv_params["conv1/7x7_s2"][1]
conv1_7x7_s2.set_weights(weights)
bias = bias.reshape(conv1_7x7_s2.get_bias().shape)
conv1_7x7_s2.set_bias(bias)
weights= conv_params["conv2/3x3_reduce"][0]
bias = conv_params["conv2/3x3_reduce"][1]

conv2_3x3_reduce.set_weights(weights)
bias = bias.reshape(conv2_3x3_reduce.get_bias().shape)
conv2_3x3_reduce.set_bias(bias)

weights= conv_params["conv2/3x3"][0]
bias = conv_params["conv2/3x3"][1]
conv2_3x3.set_weights(weights)
bias = bias.reshape(conv2_3x3.get_bias().shape)
conv2_3x3.set_bias(bias)

weights= conv_params["inception_3a/1x1"][0]
bias = conv_params["inception_3a/1x1"][1]
inception_3a_1x1.set_weights(weights)
bias = bias.reshape(inception_3a_1x1.get_bias().shape)
inception_3a_1x1.set_bias(bias)

weights= conv_params["inception_3a/3x3_reduce"][0]
bias = conv_params["inception_3a/3x3_reduce"][1]
inception_3a_3x3_reduce.set_weights(weights)
bias = bias.reshape(inception_3a_3x3_reduce.get_bias().shape)
inception_3a_3x3_reduce.set_bias(bias)

weights= conv_params["inception_3a/3x3"][0]
bias = conv_params["inception_3a/3x3"][1]
inception_3a_3x3.set_weights(weights)
bias = bias.reshape(inception_3a_3x3.get_bias().shape)
inception_3a_3x3.set_bias(bias)

weights= conv_params["inception_3a/5x5_reduce"][0]
bias = conv_params["inception_3a/5x5_reduce"][1]
inception_3a_5x5_reduce.set_weights(weights)
bias = bias.reshape(inception_3a_5x5_reduce.get_bias().shape)
inception_3a_5x5_reduce.set_bias(bias)
weights= conv_params["inception_3a/5x5"][0]
bias = conv_params["inception_3a/5x5"][1]
inception_3a_5x5.set_weights(weights)
bias = bias.reshape(inception_3a_5x5.get_bias().shape)
inception_3a_5x5.set_bias(bias)
weights= conv_params["inception_3a/pool_proj"][0]
bias = conv_params["inception_3a/pool_proj"][1]
inception_3a_pool_proj.set_weights(weights)
bias = bias.reshape(inception_3a_pool_proj.get_bias().shape)
inception_3a_pool_proj.set_bias(bias)
weights= conv_params["inception_3b/1x1"][0]
bias = conv_params["inception_3b/1x1"][1]
inception_3b_1x1.set_weights(weights)
bias = bias.reshape(inception_3b_1x1.get_bias().shape)
inception_3b_1x1.set_bias(bias)
weights= conv_params["inception_3b/3x3_reduce"][0]
bias = conv_params["inception_3b/3x3_reduce"][1]
inception_3b_3x3_reduce.set_weights(weights)
bias = bias.reshape(inception_3b_3x3_reduce.get_bias().shape)
inception_3b_3x3_reduce.set_bias(bias)
weights= conv_params["inception_3b/3x3"][0]
bias = conv_params["inception_3b/3x3"][1]
inception_3b_3x3.set_weights(weights)
bias = bias.reshape(inception_3b_3x3.get_bias().shape)
inception_3b_3x3.set_bias(bias)
weights= conv_params["inception_3b/5x5_reduce"][0]
bias = conv_params["inception_3b/5x5_reduce"][1]
inception_3b_5x5_reduce.set_weights(weights)
bias = bias.reshape(inception_3b_5x5_reduce.get_bias().shape)
inception_3b_5x5_reduce.set_bias(bias)
weights= conv_params["inception_3b/5x5"][0]
bias = conv_params["inception_3b/5x5"][1]
inception_3b_5x5.set_weights(weights)
bias = bias.reshape(inception_3b_5x5.get_bias().shape)
inception_3b_5x5.set_bias(bias)
weights= conv_params["inception_3b/pool_proj"][0]
bias = conv_params["inception_3b/pool_proj"][1]
inception_3b_pool_proj.set_weights(weights)
bias = bias.reshape(inception_3b_pool_proj.get_bias().shape)
inception_3b_pool_proj.set_bias(bias)
weights= conv_params["inception_4a/1x1"][0]
bias = conv_params["inception_4a/1x1"][1]
inception_4a_1x1.set_weights(weights)
bias = bias.reshape(inception_4a_1x1.get_bias().shape)
inception_4a_1x1.set_bias(bias)
weights= conv_params["inception_4a/3x3_reduce"][0]
bias = conv_params["inception_4a/3x3_reduce"][1]
inception_4a_3x3_reduce.set_weights(weights)
bias = bias.reshape(inception_4a_3x3_reduce.get_bias().shape)
inception_4a_3x3_reduce.set_bias(bias)
weights= conv_params["inception_4a/3x3"][0]
bias = conv_params["inception_4a/3x3"][1]
inception_4a_3x3.set_weights(weights)
bias = bias.reshape(inception_4a_3x3.get_bias().shape)
inception_4a_3x3.set_bias(bias)
weights= conv_params["inception_4a/5x5_reduce"][0]
bias = conv_params["inception_4a/5x5_reduce"][1]
inception_4a_5x5_reduce.set_weights(weights)
bias = bias.reshape(inception_4a_5x5_reduce.get_bias().shape)
inception_4a_5x5_reduce.set_bias(bias)
weights= conv_params["inception_4a/5x5"][0]
bias = conv_params["inception_4a/5x5"][1]
inception_4a_5x5.set_weights(weights)
bias = bias.reshape(inception_4a_5x5.get_bias().shape)
inception_4a_5x5.set_bias(bias)
weights= conv_params["inception_4a/pool_proj"][0]
bias = conv_params["inception_4a/pool_proj"][1]
inception_4a_pool_proj.set_weights(weights)
bias = bias.reshape(inception_4a_pool_proj.get_bias().shape)
inception_4a_pool_proj.set_bias(bias)
weights= conv_params["inception_4b/1x1"][0]
bias = conv_params["inception_4b/1x1"][1]
inception_4b_1x1.set_weights(weights)
bias = bias.reshape(inception_4b_1x1.get_bias().shape)
inception_4b_1x1.set_bias(bias)
weights= conv_params["inception_4b/3x3_reduce"][0]
bias = conv_params["inception_4b/3x3_reduce"][1]
inception_4b_3x3_reduce.set_weights(weights)
bias = bias.reshape(inception_4b_3x3_reduce.get_bias().shape)
inception_4b_3x3_reduce.set_bias(bias)
weights= conv_params["inception_4b/3x3"][0]
bias = conv_params["inception_4b/3x3"][1]
inception_4b_3x3.set_weights(weights)
bias = bias.reshape(inception_4b_3x3.get_bias().shape)
inception_4b_3x3.set_bias(bias)
weights= conv_params["inception_4b/5x5_reduce"][0]
bias = conv_params["inception_4b/5x5_reduce"][1]
inception_4b_5x5_reduce.set_weights(weights)
bias = bias.reshape(inception_4b_5x5_reduce.get_bias().shape)
inception_4b_5x5_reduce.set_bias(bias)
weights= conv_params["inception_4b/5x5"][0]
bias = conv_params["inception_4b/5x5"][1]
inception_4b_5x5.set_weights(weights)
bias = bias.reshape(inception_4b_5x5.get_bias().shape)
inception_4b_5x5.set_bias(bias)
weights= conv_params["inception_4b/pool_proj"][0]
bias = conv_params["inception_4b/pool_proj"][1]
inception_4b_pool_proj.set_weights(weights)
bias = bias.reshape(inception_4b_pool_proj.get_bias().shape)
inception_4b_pool_proj.set_bias(bias)
weights= conv_params["inception_4c/1x1"][0]
bias = conv_params["inception_4c/1x1"][1]
inception_4c_1x1.set_weights(weights)
bias = bias.reshape(inception_4c_1x1.get_bias().shape)
inception_4c_1x1.set_bias(bias)
weights= conv_params["inception_4c/3x3_reduce"][0]
bias = conv_params["inception_4c/3x3_reduce"][1]
inception_4c_3x3_reduce.set_weights(weights)
bias = bias.reshape(inception_4c_3x3_reduce.get_bias().shape)
inception_4c_3x3_reduce.set_bias(bias)
weights= conv_params["inception_4c/3x3"][0]
bias = conv_params["inception_4c/3x3"][1]
inception_4c_3x3.set_weights(weights)
bias = bias.reshape(inception_4c_3x3.get_bias().shape)
inception_4c_3x3.set_bias(bias)
weights= conv_params["inception_4c/5x5_reduce"][0]
bias = conv_params["inception_4c/5x5_reduce"][1]
inception_4c_5x5_reduce.set_weights(weights)
bias = bias.reshape(inception_4c_5x5_reduce.get_bias().shape)
inception_4c_5x5_reduce.set_bias(bias)
weights= conv_params["inception_4c/5x5"][0]
bias = conv_params["inception_4c/5x5"][1]
inception_4c_5x5.set_weights(weights)
bias = bias.reshape(inception_4c_5x5.get_bias().shape)
inception_4c_5x5.set_bias(bias)
weights= conv_params["inception_4c/pool_proj"][0]
bias = conv_params["inception_4c/pool_proj"][1]
inception_4c_pool_proj.set_weights(weights)
bias = bias.reshape(inception_4c_pool_proj.get_bias().shape)
inception_4c_pool_proj.set_bias(bias)
weights= conv_params["inception_4d/1x1"][0]
bias = conv_params["inception_4d/1x1"][1]
inception_4d_1x1.set_weights(weights)
bias = bias.reshape(inception_4d_1x1.get_bias().shape)
inception_4d_1x1.set_bias(bias)
weights= conv_params["inception_4d/3x3_reduce"][0]
bias = conv_params["inception_4d/3x3_reduce"][1]
inception_4d_3x3_reduce.set_weights(weights)
bias = bias.reshape(inception_4d_3x3_reduce.get_bias().shape)
inception_4d_3x3_reduce.set_bias(bias)
weights= conv_params["inception_4d/3x3"][0]
bias = conv_params["inception_4d/3x3"][1]
inception_4d_3x3.set_weights(weights)
bias = bias.reshape(inception_4d_3x3.get_bias().shape)
inception_4d_3x3.set_bias(bias)
weights= conv_params["inception_4d/5x5_reduce"][0]
bias = conv_params["inception_4d/5x5_reduce"][1]
inception_4d_5x5_reduce.set_weights(weights)
bias = bias.reshape(inception_4d_5x5_reduce.get_bias().shape)
inception_4d_5x5_reduce.set_bias(bias)
weights= conv_params["inception_4d/5x5"][0]
bias = conv_params["inception_4d/5x5"][1]
inception_4d_5x5.set_weights(weights)
bias = bias.reshape(inception_4d_5x5.get_bias().shape)
inception_4d_5x5.set_bias(bias)
weights= conv_params["inception_4d/pool_proj"][0]
bias = conv_params["inception_4d/pool_proj"][1]
inception_4d_pool_proj.set_weights(weights)
bias = bias.reshape(inception_4d_pool_proj.get_bias().shape)
inception_4d_pool_proj.set_bias(bias)
weights= conv_params["inception_4e/1x1"][0]
bias = conv_params["inception_4e/1x1"][1]
inception_4e_1x1.set_weights(weights)
bias = bias.reshape(inception_4e_1x1.get_bias().shape)
inception_4e_1x1.set_bias(bias)
weights= conv_params["inception_4e/3x3_reduce"][0]
bias = conv_params["inception_4e/3x3_reduce"][1]
inception_4e_3x3_reduce.set_weights(weights)
bias = bias.reshape(inception_4e_3x3_reduce.get_bias().shape)
inception_4e_3x3_reduce.set_bias(bias)
weights= conv_params["inception_4e/3x3"][0]
bias = conv_params["inception_4e/3x3"][1]
inception_4e_3x3.set_weights(weights)
bias = bias.reshape(inception_4e_3x3.get_bias().shape)
inception_4e_3x3.set_bias(bias)
weights= conv_params["inception_4e/5x5_reduce"][0]
bias = conv_params["inception_4e/5x5_reduce"][1]
inception_4e_5x5_reduce.set_weights(weights)
bias = bias.reshape(inception_4e_5x5_reduce.get_bias().shape)
inception_4e_5x5_reduce.set_bias(bias)
weights= conv_params["inception_4e/5x5"][0]
bias = conv_params["inception_4e/5x5"][1]
inception_4e_5x5.set_weights(weights)
bias = bias.reshape(inception_4e_5x5.get_bias().shape)
inception_4e_5x5.set_bias(bias)
weights= conv_params["inception_4e/pool_proj"][0]
bias = conv_params["inception_4e/pool_proj"][1]
inception_4e_pool_proj.set_weights(weights)
bias = bias.reshape(inception_4e_pool_proj.get_bias().shape)
inception_4e_pool_proj.set_bias(bias)
weights= conv_params["inception_5a/1x1"][0]
bias = conv_params["inception_5a/1x1"][1]
inception_5a_1x1.set_weights(weights)
bias = bias.reshape(inception_5a_1x1.get_bias().shape)
inception_5a_1x1.set_bias(bias)
weights= conv_params["inception_5a/3x3_reduce"][0]
bias = conv_params["inception_5a/3x3_reduce"][1]
inception_5a_3x3_reduce.set_weights(weights)
bias = bias.reshape(inception_5a_3x3_reduce.get_bias().shape)
inception_5a_3x3_reduce.set_bias(bias)
weights= conv_params["inception_5a/3x3"][0]
bias = conv_params["inception_5a/3x3"][1]
inception_5a_3x3.set_weights(weights)
bias = bias.reshape(inception_5a_3x3.get_bias().shape)
inception_5a_3x3.set_bias(bias)
weights= conv_params["inception_5a/5x5_reduce"][0]
bias = conv_params["inception_5a/5x5_reduce"][1]
inception_5a_5x5_reduce.set_weights(weights)
bias = bias.reshape(inception_5a_5x5_reduce.get_bias().shape)
inception_5a_5x5_reduce.set_bias(bias)
weights= conv_params["inception_5a/5x5"][0]
bias = conv_params["inception_5a/5x5"][1]
inception_5a_5x5.set_weights(weights)
bias = bias.reshape(inception_5a_5x5.get_bias().shape)
inception_5a_5x5.set_bias(bias)
weights= conv_params["inception_5a/pool_proj"][0]
bias = conv_params["inception_5a/pool_proj"][1]
inception_5a_pool_proj.set_weights(weights)
bias = bias.reshape(inception_5a_pool_proj.get_bias().shape)
inception_5a_pool_proj.set_bias(bias)
weights= conv_params["inception_5b/1x1"][0]
bias = conv_params["inception_5b/1x1"][1]
inception_5b_1x1.set_weights(weights)
bias = bias.reshape(inception_5b_1x1.get_bias().shape)
inception_5b_1x1.set_bias(bias)
weights= conv_params["inception_5b/3x3_reduce"][0]
bias = conv_params["inception_5b/3x3_reduce"][1]
inception_5b_3x3_reduce.set_weights(weights)
bias = bias.reshape(inception_5b_3x3_reduce.get_bias().shape)
inception_5b_3x3_reduce.set_bias(bias)
weights= conv_params["inception_5b/3x3"][0]
bias = conv_params["inception_5b/3x3"][1]
inception_5b_3x3.set_weights(weights)
bias = bias.reshape(inception_5b_3x3.get_bias().shape)
inception_5b_3x3.set_bias(bias)
weights= conv_params["inception_5b/5x5_reduce"][0]
bias = conv_params["inception_5b/5x5_reduce"][1]
inception_5b_5x5_reduce.set_weights(weights)
bias = bias.reshape(inception_5b_5x5_reduce.get_bias().shape)
inception_5b_5x5_reduce.set_bias(bias)
weights= conv_params["inception_5b/5x5"][0]
bias = conv_params["inception_5b/5x5"][1]
inception_5b_5x5.set_weights(weights)
bias = bias.reshape(inception_5b_5x5.get_bias().shape)
inception_5b_5x5.set_bias(bias)
weights= conv_params["inception_5b/pool_proj"][0]
bias = conv_params["inception_5b/pool_proj"][1]
inception_5b_pool_proj.set_weights(weights)
bias = bias.reshape(inception_5b_pool_proj.get_bias().shape)
inception_5b_pool_proj.set_bias(bias)
weights= conv_params["loss3/classifier"][0]
bias = conv_params["loss3/classifier"][1]
weights = np.expand_dims(weights, axis = 2)
weights = np.expand_dims(weights, axis = 3)

loss3_classifier.set_weights(weights)
bias = bias.reshape(loss3_classifier.get_bias().shape)
loss3_classifier.set_bias(bias)

data.set_value(batched_image)
print("Finished Copying\n")
net.forward()
print("Finished forward computation\n") 
net.backward()
print("Finished backward computation\n")

output = loss3_classifier.get_value()

x = output[0]
e_x = np.exp(x - np.max(x))
latte_prob = e_x / e_x.sum()

print( 'output label:', labels[latte_prob.argmax()])

assert np.allclose(net2.blobs["data"].data, data.get_value(), 1e-1, 1e-1)
assert np.allclose(net2.blobs["conv1/7x7_s2"].data, conv1_7x7_s2.get_value(), 1e-1, 1e-1)
assert np.allclose(net2.blobs["pool1/3x3_s2"].data, pool1_3x3_s2.get_value(), 1e-1, 1e-1)
assert np.allclose(net2.blobs["pool1/norm1"].data, pool1_norm1.get_value(), 1e-1, 1e-1)
assert np.allclose(net2.blobs["conv2/3x3_reduce"].data, conv2_3x3_reduce.get_value(), 1e-1, 1e-1)
assert np.allclose(net2.blobs["conv2/3x3"].data, conv2_3x3.get_value(), 1e-1, 1e-1)
assert np.allclose(net2.blobs["conv2/norm2"].data, conv2_norm2.get_value(), 1e-1, 1e-1)
assert np.allclose(net2.blobs["pool2/3x3_s2"].data, pool2_3x3_s2.get_value(), 1e-1, 1e-1)
assert np.allclose(net2.blobs["inception_3a/1x1"].data, inception_3a_1x1.get_value(), 1e-1, 1e-1)
assert np.allclose(net2.blobs["inception_3a/3x3"].data, inception_3a_3x3.get_value(), 1e-1, 1e-1)
assert np.allclose(net2.blobs["inception_3a/5x5_reduce"].data, inception_3a_5x5_reduce.get_value(), 1e-1, 1e-1)
assert np.allclose(net2.blobs["inception_3a/5x5"].data, inception_3a_5x5.get_value(), 1e-1, 1e-1)
assert np.allclose(net2.blobs["inception_3a/pool"].data, inception_3a_pool.get_value(), 1e-1, 1e-1)
assert np.allclose(net2.blobs["inception_3a/pool_proj"].data, inception_3a_pool_proj.get_value(), 1e-1, 1e-1)
assert np.allclose(net2.blobs["inception_3a/output"].data, inception_3a_output.get_value(), 1e-1, 1e-1)
assert np.allclose(net2.blobs["inception_3b/1x1"].data, inception_3b_1x1.get_value(), 1e-1, 1e-1)
assert np.allclose(net2.blobs["inception_3b/3x3_reduce"].data, inception_3b_3x3_reduce.get_value(), 1e-1, 1e-1)
assert np.allclose(net2.blobs["inception_3b/3x3"].data, inception_3b_3x3.get_value(), 1e-1, 1e-1)
assert np.allclose(net2.blobs["inception_3b/5x5_reduce"].data, inception_3b_5x5_reduce.get_value(), 1e-1, 1e-1)
assert np.allclose(net2.blobs["inception_3b/5x5"].data, inception_3b_5x5.get_value(), 1e-1, 1e-1)
assert np.allclose(net2.blobs["inception_3b/pool"].data, inception_3b_pool.get_value(), 1e-1, 1e-1)
assert np.allclose(net2.blobs["inception_3b/pool_proj"].data, inception_3b_pool_proj.get_value(), 1e-1, 1e-1)
assert np.allclose(net2.blobs["inception_3b/output"].data, inception_3b_output.get_value(), 1e-1, 1e-1)
assert np.allclose(net2.blobs["pool3/3x3_s2"].data, pool3_3x3_s2.get_value(), 1e-1, 1e-1)
assert np.allclose(net2.blobs["inception_4a/1x1"].data, inception_4a_1x1.get_value(), 1e-1, 1e-1)
assert np.allclose(net2.blobs["inception_4a/3x3_reduce"].data, inception_4a_3x3_reduce.get_value(), 1e-1, 1e-1)
assert np.allclose(net2.blobs["inception_4a/3x3"].data, inception_4a_3x3.get_value(), 1e-1, 1e-1)
assert np.allclose(net2.blobs["inception_4a/5x5_reduce"].data, inception_4a_5x5_reduce.get_value(), 1e-1, 1e-1)
assert np.allclose(net2.blobs["inception_4a/5x5"].data, inception_4a_5x5.get_value(), 1e-1, 1e-1)
assert np.allclose(net2.blobs["inception_4a/pool"].data, inception_4a_pool.get_value(), 1e-1, 1e-1)
assert np.allclose(net2.blobs["inception_4a/pool_proj"].data, inception_4a_pool_proj.get_value(), 1e-1, 1e-1)
assert np.allclose(net2.blobs["inception_4a/output"].data, inception_4a_output.get_value(), 1e-1, 1e-1)
assert np.allclose(net2.blobs["inception_4b/1x1"].data, inception_4b_1x1.get_value(), 1e-1, 1e-1)
assert np.allclose(net2.blobs["inception_4b/3x3_reduce"].data, inception_4b_3x3_reduce.get_value(), 1e-1, 1e-1)
assert np.allclose(net2.blobs["inception_4b/3x3"].data, inception_4b_3x3.get_value(), 1e-1, 1e-1)
assert np.allclose(net2.blobs["inception_4b/5x5_reduce"].data, inception_4b_5x5_reduce.get_value(), 1e-1, 1e-1)
assert np.allclose(net2.blobs["inception_4b/5x5"].data, inception_4b_5x5.get_value(), 1e-1, 1e-1)
assert np.allclose(net2.blobs["inception_4b/pool"].data, inception_4b_pool.get_value(), 1e-1, 1e-1)
assert np.allclose(net2.blobs["inception_4b/pool_proj"].data, inception_4b_pool_proj.get_value(), 1e-1, 1e-1)
assert np.allclose(net2.blobs["inception_4b/output"].data, inception_4b_output.get_value(), 1e-1, 1e-1)
assert np.allclose(net2.blobs["inception_4c/1x1"].data, inception_4c_1x1.get_value(), 1e-1, 1e-1)
assert np.allclose(net2.blobs["inception_4c/3x3_reduce"].data, inception_4c_3x3_reduce.get_value(), 1e-1, 1e-1)
assert np.allclose(net2.blobs["inception_4c/3x3"].data, inception_4c_3x3.get_value(), 1e-1, 1e-1)
assert np.allclose(net2.blobs["inception_4c/5x5_reduce"].data, inception_4c_5x5_reduce.get_value(), 1e-1, 1e-1)
assert np.allclose(net2.blobs["inception_4c/5x5"].data, inception_4c_5x5.get_value(), 1e-1, 1e-1)
assert np.allclose(net2.blobs["inception_4c/pool"].data, inception_4c_pool.get_value(), 1e-1, 1e-1)
assert np.allclose(net2.blobs["inception_4c/pool_proj"].data, inception_4c_pool_proj.get_value(), 1e-1, 1e-1)
assert np.allclose(net2.blobs["inception_4c/output"].data, inception_4c_output.get_value(), 1e-1, 1e-1)
assert np.allclose(net2.blobs["inception_4d/1x1"].data, inception_4d_1x1.get_value(), 1e-1, 1e-1)
assert np.allclose(net2.blobs["inception_4d/3x3_reduce"].data, inception_4d_3x3_reduce.get_value(), 1e-1, 1e-1)
assert np.allclose(net2.blobs["inception_4d/3x3"].data, inception_4d_3x3.get_value(), 1e-1, 1e-1)
assert np.allclose(net2.blobs["inception_4d/5x5_reduce"].data, inception_4d_5x5_reduce.get_value(), 1e-1, 1e-1)
assert np.allclose(net2.blobs["inception_4d/5x5"].data, inception_4d_5x5.get_value(), 1e-1, 1e-1)
assert np.allclose(net2.blobs["inception_4d/pool"].data, inception_4d_pool.get_value(), 1e-1, 1e-1)
assert np.allclose(net2.blobs["inception_4d/pool_proj"].data, inception_4d_pool_proj.get_value(), 1e-1, 1e-1)
assert np.allclose(net2.blobs["inception_4d/output"].data, inception_4d_output.get_value(), 1e-1, 1e-1)
assert np.allclose(net2.blobs["inception_4e/1x1"].data, inception_4e_1x1.get_value(), 1e-1, 1e-1)
assert np.allclose(net2.blobs["inception_4e/3x3_reduce"].data, inception_4e_3x3_reduce.get_value(), 1e-1, 1e-1)
assert np.allclose(net2.blobs["inception_4e/3x3"].data, inception_4e_3x3.get_value(), 1e-1, 1e-1)
assert np.allclose(net2.blobs["inception_4e/5x5_reduce"].data, inception_4e_5x5_reduce.get_value(), 1e-1, 1e-1)
assert np.allclose(net2.blobs["inception_4e/5x5"].data, inception_4e_5x5.get_value(), 1e-1, 1e-1)
assert np.allclose(net2.blobs["inception_4e/pool"].data, inception_4e_pool.get_value(), 1e-1, 1e-1)
assert np.allclose(net2.blobs["inception_4e/pool_proj"].data, inception_4e_pool_proj.get_value(), 1e-1, 1e-1)
assert np.allclose(net2.blobs["inception_4e/output"].data, inception_4e_output.get_value(), 1e-1, 1e-1)
assert np.allclose(net2.blobs["pool4/3x3_s2"].data, pool4_3x3_s2.get_value(), 1e-1, 1e-1)
assert np.allclose(net2.blobs["inception_5a/1x1"].data, inception_5a_1x1.get_value(), 1e-1, 1e-1)
assert np.allclose(net2.blobs["inception_5a/3x3_reduce"].data, inception_5a_3x3_reduce.get_value(), 1e-1, 1e-1)
assert np.allclose(net2.blobs["inception_5a/3x3"].data, inception_5a_3x3.get_value(), 1e-1, 1e-1)
assert np.allclose(net2.blobs["inception_5a/5x5_reduce"].data, inception_5a_5x5_reduce.get_value(), 1e-1, 1e-1)
assert np.allclose(net2.blobs["inception_5a/pool"].data, inception_5a_pool.get_value(), 1e-1, 1e-1)
assert np.allclose(net2.blobs["inception_5a/pool_proj"].data, inception_5a_pool_proj.get_value(), 1e-1, 1e-1)
assert np.allclose(net2.blobs["inception_5a/output"].data, inception_5a_output.get_value(), 1e-1, 1e-1)
assert np.allclose(net2.blobs["inception_5b/1x1"].data, inception_5b_1x1.get_value(), 1e-1, 1e-1)
assert np.allclose(net2.blobs["inception_5b/3x3_reduce"].data, inception_5b_3x3_reduce.get_value(), 1e-1, 1e-1)
assert np.allclose(net2.blobs["inception_5b/3x3"].data, inception_5b_3x3.get_value(), 1e-1, 1e-1)
assert np.allclose(net2.blobs["inception_5b/5x5_reduce"].data, inception_5b_5x5_reduce.get_value(), 1e-1, 1e-1)
assert np.allclose(net2.blobs["inception_5b/5x5"].data, inception_5b_5x5.get_value(), 1e-1, 1e-1)
assert np.allclose(net2.blobs["inception_5b/pool"].data, inception_5b_pool.get_value(), 1e-1, 1e-1)
assert np.allclose(net2.blobs["inception_5b/pool_proj"].data, inception_5b_pool_proj.get_value(), 1e-1, 1e-1)
assert np.allclose(net2.blobs["inception_5b/output"].data, inception_5b_output.get_value(), 1e-1, 1e-1)
assert np.allclose(net2.blobs["pool5/7x7_s1"].data, pool5_7x7_s1.get_value(), 1e-1, 1e-1)
assert np.allclose(net2.blobs["loss3/classifier"].data, loss3_classifier.get_value(), 1e-1, 1e-1)

