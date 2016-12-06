import numpy as np
from latte import *
import caffe
from latte.math import compute_softmax_loss
from latte.solvers import sgd_update
import os



def reference_pooling_forward(_input, kernel, pad, stride):
    stride_h, stride_w = stride, stride
    pad_h, pad_w = pad, pad
    kernel_h, kernel_w = kernel, kernel
    batch_size, in_channels, in_height, in_width = _input.shape
    output_width = ((in_width - kernel_w + 2 * pad_w) // stride_w) + 1
    output_height = ((in_height - kernel_h + 2 * pad_h) // stride_h) + 1
    output = np.zeros((batch_size, in_channels, output_height, output_width), dtype=np.float32)
    output_mask = np.zeros((batch_size, in_channels, output_height, output_width, 2), dtype=np.int32)
    for n in range(batch_size):
        for o in range(in_channels):
            for y in range(output_height):
                for x in range(output_width):
                    in_y = y*stride_h - pad
                    in_x = x*stride_w - pad
                    out_y = in_y + kernel_h
                    out_x = in_x + kernel_w
                    sumval = 0.0
                    #idx = ()
                    for i, p in enumerate(range(in_y, out_y)):
                        p = min(max(p, 0), in_height - 1)
                        for j, q in enumerate(range(in_x, out_x)):
                            q = min(max(q, 0), in_width - 1)
                            curr = _input[n, o, p, q]
                            #rint(curr)
                            sumval += curr
                            #if curr > maxval:
                            #    idx = (i, j)
                            #    maxval = curr
                    output[n, o, y, x] = sumval/(kernel_h*kernel_w)
                    #output_mask[n, o, y, x, :] = idx
    return output


def check_equal(actual, expected, atol=1e-2):
    assert np.allclose(actual, expected, atol=atol)



 
caffe_root ='/home/avenkat/caffe/'

 
caffe.set_mode_cpu()
 
#model_def = './deploy.prototxt'
#model_weights = caffe_root + 'models/bvlc_googlenet/bvlc_googlenet.caffemodel'
 
#net2 = solver.net
#net2 = caffe.Net(model_def, # caffe.TRAIN)    # defines the structure of the model
#                model_weights,  # contains the trained weights
#                caffe.TRAIN)   
#caffe_output = net2.forward()
#solver = caffe.get_solver('solver.prototxt')
#solver.step(1)
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
"inception_5b/pool_proj","loss3/classifier", "loss1/conv", "loss1/fc", "loss1/classifier",\
"loss2/conv", "loss2/fc", "loss2/classifier"
]

latte_convs = ["conv1_7x7_s2","conv2_3x3_reduce","conv2_3x3","inception_3a_1x1",\
"inception_3a_3x3_reduce","inception_3a_3x3","inception_3a_5x5_reduce","inception_3a_5x5",\
"inception_3a_pool_proj","inception_3b_1x1","inception_3b_3x3_reduce","inception_3b_3x3", \
"inception_3b_5x5_reduce","inception_3b_5x5","inception_3b_pool_proj","inception_4a_1x1", \
"inception_4a_3x3_reduce","inception_4a_3x3","inception_4a_5x5_reduce","inception_4a_5x5",\
"inception_4a_pool_proj","inception_4b_1x1","inception_4b_3x3_reduce","inception_4b_3x3",\
"inception_4b_5x5_reduce","inception_4b_5x5","inception_4b_pool_proj","inception_4c_1x1",\
"inception_4c_3x3_reduce","inception_4c_3x3","inception_4c_5x5_reduce","inception_4c_5x5",\
"inception_4c_pool_proj","inception_4d_1x1","inception_4d_3x3_reduce","inception_4d_3x3",\
"inception_4d_5x5_reduce","inception_4d_5x5","inception_4d_pool_proj","inception_4e_1x1",\
"inception_4e_3x3_reduce","inception_4e_3x3","inception_4e_5x5_reduce","inception_4e_5x5",\
"inception_4e_pool_proj","inception_5a_1x1","inception_5a_3x3_reduce","inception_5a_3x3",\
"inception_5a_5x5_reduce","inception_5a_5x5","inception_5a_pool_proj","inception_5b_1x1",\
"inception_5b_3x3_reduce","inception_5b_3x3","inception_5b_5x5_reduce","inception_5b_5x5",\
"inception_5b_pool_proj","loss3_classifier", "loss1_conv", "loss1_fc", "loss1/classifier",\
"loss2_conv", "loss2_fc", "loss2_classifier"
]


#ioss3_classifier.set_bias(bias)
##print("HIGHLIGHT\n")
##print(net2.blobs["inception_5b/output"].data.shape)
###print(net2.blobs["inception_3a/1x1"].data.shape)


#for key, value inconv_params.items():#    ##print(key, value)#    ##print("\n")

#for i in net2.params:
#    if net2.params[i][0].type == 'Convolution':
#        ##print( net2.params[i].type)

###print(conv_params)

labels_file = caffe_root + 'data/ilsvrc12/synset_words.txt'
if not os.path.exists(labels_file):
    os.system("...data/ilsvrc12/get_ilsvrc_aux.sh")
#label = np.loadtxt(labels_file,str, comments='\t', delimiter='\t', converters=None, skiprows=0, usecols=None, unpack=False, ndmin=2) 
labels = np.loadtxt(labels_file, str, delimiter='\t')
###print(label.shape)
###print(label[0,0])
###print( 'output label:', labels[output_prob.argmax()])
'''
batch_size = 1
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
''' 
##print ('mean-subtracted values:', test2)
solver = caffe.SGDSolver('solver.prototxt')
##solver.net.blobs[.data'].data[...] = transformed_image
net2 = solver.net
#before = net2.blobs["conv1/7x7_s2"].diff
#print(before) 
#net2 = solver.net
#conv_params = {pr: (net2.params[pr][0].data, net2.params[pr][1].data) for pr in convs}

#solver.step(1)
net2.forward()
#before = net2.blobs["conv1/7x7_s2"].diff
net2.backward()
#after = net2.blobs["conv1/7x7_s2"].diff
#print(after)
#assert(check_equal(before, after))
#net2 = solver.net
#print(net2.blobs["conv1/7x7_s2"].diff)
#net2 = solver.net
conv_params = {pr: (net2.params[pr][0].data, net2.params[pr][1].data) for pr in convs}

batch_size = 1
net = Net(batch_size)
channels=3
height=224
width=224

'''
# create transformer for the input called .data'
transformer = caffe.io.Transformer({'data': net2.blobs['data'].data.shape})
 
transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
transformer.set_mean('data', mu)            # subtract the dataset-mean value in each channel
transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR
 
# set the size of the input (we can skip this if we're happy
#  with the default; we can also change it later, e.g., for.dataerent batch sizes)
#net.blobs[.data'].reshape(50,        # batch size
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

# copy the image.data into the memory allocated for the net


net2.blobs['data'].data[...] = transformed_image
#solver = caffe.get_solver('solver.prototxt')
#.blobs[.data'].data[...] = transformed_image

#solver.step(1)
#net2 = solver.net

ionv_params = {pr: (net2.params[pr][0].data, net2.params[pr][1].data) for pr in convs}
#solver.step(2)
#net2 = solver.net
#conv_params2 = {pr: (net2.params[pr][0].data, net2.params[pr][1].data) for pr in convs}

#for pr in convs:
#    check_equal(net2.params[pr][0].data, conv_params[pr][0])
#    check_equal(net2.params[pr][1].data, conv_params[pr][1]) 
lr = 100
layer_types = []
for ll in net2.layers:
    layer_types.append(ll.type)

# Get the indices of layers that have weights in them
weight_layer_idx = [idx for idx,l in enumerate(layer_types) if 'Convolution' in l or 'InnerProduct' in l]

#for it in range(1, niter+1):
net2.forward()  # fprop
net2.backward()  # bprop

for k in weight_layer_idx:
    net2.layers[k].blobs[0].data[...] *= lr
    net2.layers[k].blobs[1].data[...] *= lr
    net2.layers[k].blobs[0].data[...] -= net2.layers[k].blobs[0].diff
    net2.layers[k].blobs[1].data[...] -= net2.layers[k].blobs[1].diff





net2.forward()
#net2.backward()

'''


#net2.forward()
##print("caffe weights\n") 
##print(conv_params["loss3/classifier"][0].size)


data = MemoryDataLayer(net, (3, 224, 224))
conv1_7x7_s2 = ConvLayer(net, data, num_filters=64, kernel=7, stride=2, pad=3)

conv1_relu_7x7 = ReLULayer(net, conv1_7x7_s2)
pool1_3x3_s2 = MaxPoolingLayer(net, conv1_7x7_s2, kernel=3, stride=2, pad=0)
pool1_norm1 = LRNLayer(net, pool1_3x3_s2, n=5, beta=0.75, alpha=0.0001, k=1.0)
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
loss1_ave_pool = MeanPoolingLayer(net, inception_4a_output, kernel=5, stride=3, pad=0)
loss1_conv = ConvLayer(net, loss1_ave_pool, num_filters=128, kernel=1, stride=1, pad=0)
loss1_relu_conv = ReLULayer(net, loss1_conv)
loss1_fc =  FullyConnectedLayer(net, loss1_relu_conv, 1024)
loss1_relu_fc = ReLULayer(net, loss1_fc)
loss1_classifier =  FullyConnectedLayer(net, loss1_relu_fc, 1000)
#loss1_loss = SoftmaxLossLayer(net, loss1_classifier, label)
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
loss2_ave_pool = MeanPoolingLayer(net, inception_4d_output, kernel=5, stride=3, pad=0)
loss2_conv = ConvLayer(net, loss2_ave_pool, num_filters=128, kernel=1, stride=1, pad=0)
loss2_relu_conv = ReLULayer(net, loss2_conv)
loss2_fc =  FullyConnectedLayer(net, loss2_relu_conv, 1024)
loss2_relu_fc = ReLULayer(net, loss2_fc)
loss2_classifier =  FullyConnectedLayer(net, loss2_relu_fc, 1000)
#loss2_loss = SoftmaxLossLayer(net, loss2_classifier, label)
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
loss3_classifier =  FullyConnectedLayer(net, pool5_7x7_s1, 1000)
#loss3_loss3 = SoftmaxLossLayer(net, loss3_classifier, label)
















































net.compile() 

#net2 = solver.net
#conv_params = {pr: (net2.params[pr][0].data, net2.params[pr][1].data) for pr in convs}
        
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



weights= conv_params["loss1/conv"][0]
bias = conv_params["loss1/conv"][1]
loss1_conv.set_weights(weights)
bias = bias.reshape(loss1_conv.get_bias().shape)
loss1_conv.set_bias(bias)



weights= conv_params["loss1/classifier"][0]
bias = conv_params["loss1/classifier"][1]
#weights = np.expand_dims(weights, axis = 2)
#weights = np.expand_dims(weights, axis = 3)
 
loss1_classifier.set_weights(weights)
bias = bias.reshape(loss1_classifier.get_bias().shape)
loss1_classifier.set_bias(bias)

weights= conv_params["loss1/fc"][0]
bias = conv_params["loss1/fc"][1]
#weights = np.expand_dims(weights, axis = 2)
#weights = np.expand_dims(weights, axis = 3)

print(weights.shape) 
print (loss1_fc.get_weights().shape)
loss1_fc.set_weights(weights.reshape(loss1_fc.get_weights().shape))
bias = bias.reshape(loss1_fc.get_bias().shape)
loss1_fc.set_bias(bias)

weights= conv_params["loss2/conv"][0]
bias = conv_params["loss2/conv"][1]
loss2_conv.set_weights(weights)
bias = bias.reshape(loss2_conv.get_bias().shape)
loss2_conv.set_bias(bias)
 
 
 
weights= conv_params["loss2/classifier"][0]
bias = conv_params["loss2/classifier"][1]
#weights = np.expand_dims(weights, axis = 2)
#weights = np.expand_dims(weights, axis = 3)
 
loss2_classifier.set_weights(weights)
bias = bias.reshape(loss2_classifier.get_bias().shape)
loss2_classifier.set_bias(bias)
 
weights= conv_params["loss2/fc"][0]
bias = conv_params["loss2/fc"][1]
#weights = np.expand_dims(weights, axis = 2)
#weights = np.expand_dims(weights, axis = 3)
 
print(weights.shape)
print (loss2_fc.get_weights().shape)
loss2_fc.set_weights(weights.reshape(loss2_fc.get_weights().shape))
bias = bias.reshape(loss2_fc.get_bias().shape)
loss2_fc.set_bias(bias)

weights= conv_params["loss3/classifier"][0]
bias = conv_params["loss3/classifier"][1]
#weights = np.expand_dims(weights, axis = 2)
#weights = np.expand_dims(weights, axis = 3)
 
#loss3_classifier.set_weights(weights)
#bias = bias.reshape(loss3_classifier.get_bias().shape)
#loss3_classifier.set_bias(bias)


data.set_value(net2.blobs["data"].data)
print("Finished Copying\n")
net.forward()
#solver.step(1)
print("Finished forward computation\n") 
loss1_classifier.set_grad(net2.blobs["loss1/classifier"].diff) 
loss2_classifier.set_grad(net2.blobs["loss2/classifier"].diff) 
loss3_classifier.set_grad(net2.blobs["loss3/classifier"].diff)

net.backward()

base_lr = 0.01
momentum = 0.9
weight_decay = 0.0002
lr_w_mult = 1
lr_b_mult = 2

momentum_hist = {}
for layer in net2.params:
    m_w = np.zeros_like(net2.params[layer][0].data)
    m_b = np.zeros_like(net2.params[layer][1].data)
    momentum_hist[layer] = [m_w, m_b]

for layer in solver.net.params:
    momentum_hist[layer][0] = momentum_hist[layer][0] * momentum + (solver.net.params[layer][0].diff + weight_decay *
                                                       solver.net.params[layer][0].data) * base_lr * lr_w_mult
    momentum_hist[layer][1] = momentum_hist[layer][1] * momentum + (net2.params[layer][1].diff + weight_decay *
                                                       net2.params[layer][1].data) * base_lr * lr_b_mult
    net2.params[layer][0].data[...] -= momentum_hist[layer][0]
    net2.params[layer][1].data[...] -= momentum_hist[layer][1]
    #solver.net.params[layer][0].diff[...] *= 0
    #solver.net.params[layer][1].diff[...] *= 0
    #print(layer)
#base_lr = .001
#gamma = 0.1
#power = .75


#for param in latte_convs:
#    sgd_update(latte_convs[param][0], latte_convs[param][1],np.zeros_like(latte_convs[param][0]), base_lr, momentum, batch_size)
sgd_update(conv1_7x7_s2.get_weights_view(), conv1_7x7_s2.get_grad_weights_view(),np.zeros_like(conv1_7x7_s2.get_weights_view()), base_lr, momentum, batch_size)
sgd_update(conv2_3x3_reduce.get_weights_view(), conv2_3x3_reduce.get_grad_weights_view(),np.zeros_like(conv2_3x3_reduce.get_weights_view()), base_lr, momentum, batch_size)
sgd_update(conv2_3x3.get_weights_view(), conv2_3x3.get_grad_weights_view(),np.zeros_like(conv2_3x3.get_weights_view()), base_lr, momentum, batch_size)
sgd_update(inception_3a_1x1.get_weights_view(), inception_3a_1x1.get_grad_weights_view(),np.zeros_like(inception_3a_1x1.get_weights_view()), base_lr, momentum, batch_size)
sgd_update(inception_3a_3x3_reduce.get_weights_view(), inception_3a_3x3_reduce.get_grad_weights_view(),np.zeros_like(inception_3a_3x3_reduce.get_weights_view()), base_lr, momentum, batch_size)
sgd_update(inception_3a_3x3.get_weights_view(), inception_3a_3x3.get_grad_weights_view(),np.zeros_like(inception_3a_3x3.get_weights_view()), base_lr, momentum, batch_size)
sgd_update(inception_3a_5x5_reduce.get_weights_view(), inception_3a_5x5_reduce.get_grad_weights_view(),np.zeros_like(inception_3a_5x5_reduce.get_weights_view()), base_lr, momentum, batch_size)
sgd_update(inception_3a_5x5.get_weights_view(), inception_3a_5x5.get_grad_weights_view(),np.zeros_like(inception_3a_5x5.get_weights_view()), base_lr, momentum, batch_size)
sgd_update(inception_3a_pool_proj.get_weights_view(), inception_3a_pool_proj.get_grad_weights_view(),np.zeros_like(inception_3a_pool_proj.get_weights_view()), base_lr, momentum, batch_size)
sgd_update(inception_3b_1x1.get_weights_view(), inception_3b_1x1.get_grad_weights_view(),np.zeros_like(inception_3b_1x1.get_weights_view()), base_lr, momentum, batch_size)
sgd_update(inception_3b_3x3_reduce.get_weights_view(), inception_3b_3x3_reduce.get_grad_weights_view(),np.zeros_like(inception_3b_3x3_reduce.get_weights_view()), base_lr, momentum, batch_size)
sgd_update(inception_3b_3x3.get_weights_view(), inception_3b_3x3.get_grad_weights_view(),np.zeros_like(inception_3b_3x3.get_weights_view()), base_lr, momentum, batch_size)
sgd_update(inception_3b_5x5_reduce.get_weights_view(), inception_3b_5x5_reduce.get_grad_weights_view(),np.zeros_like(inception_3b_5x5_reduce.get_weights_view()), base_lr, momentum, batch_size)
sgd_update(inception_3b_5x5.get_weights_view(), inception_3b_5x5.get_grad_weights_view(),np.zeros_like(inception_3b_5x5.get_weights_view()), base_lr, momentum, batch_size)
sgd_update(inception_3b_pool_proj.get_weights_view(), inception_3b_pool_proj.get_grad_weights_view(),np.zeros_like(inception_3b_pool_proj.get_weights_view()), base_lr, momentum, batch_size)
sgd_update(inception_4a_1x1.get_weights_view(), inception_4a_1x1.get_grad_weights_view(),np.zeros_like(inception_4a_1x1.get_weights_view()), base_lr, momentum, batch_size)
sgd_update(inception_4a_3x3_reduce.get_weights_view(), inception_4a_3x3_reduce.get_grad_weights_view(),np.zeros_like(inception_4a_3x3_reduce.get_weights_view()), base_lr, momentum, batch_size)
sgd_update(inception_4a_3x3.get_weights_view(), inception_4a_3x3.get_grad_weights_view(),np.zeros_like(inception_4a_3x3.get_weights_view()), base_lr, momentum, batch_size)
sgd_update(inception_4a_5x5_reduce.get_weights_view(), inception_4a_5x5_reduce.get_grad_weights_view(),np.zeros_like(inception_4a_5x5_reduce.get_weights_view()), base_lr, momentum, batch_size)
sgd_update(inception_4a_5x5.get_weights_view(), inception_4a_5x5.get_grad_weights_view(),np.zeros_like(inception_4a_5x5.get_weights_view()), base_lr, momentum, batch_size)
sgd_update(inception_4a_pool_proj.get_weights_view(), inception_4a_pool_proj.get_grad_weights_view(),np.zeros_like(inception_4a_pool_proj.get_weights_view()), base_lr, momentum, batch_size)
sgd_update(loss1_conv.get_weights_view(), loss1_conv.get_grad_weights_view(),np.zeros_like(loss1_conv.get_weights_view()), base_lr, momentum, batch_size)
sgd_update(loss1_fc.get_weights_view(), loss1_fc.get_grad_weights_view(),np.zeros_like(loss1_fc.get_weights_view()), base_lr, momentum, batch_size)
sgd_update(loss1_classifier.get_weights_view(), loss1_classifier.get_grad_weights_view(),np.zeros_like(loss1_classifier.get_weights_view()), base_lr, momentum, batch_size)
sgd_update(inception_4b_1x1.get_weights_view(), inception_4b_1x1.get_grad_weights_view(),np.zeros_like(inception_4b_1x1.get_weights_view()), base_lr, momentum, batch_size)
sgd_update(inception_4b_3x3_reduce.get_weights_view(), inception_4b_3x3_reduce.get_grad_weights_view(),np.zeros_like(inception_4b_3x3_reduce.get_weights_view()), base_lr, momentum, batch_size)
sgd_update(inception_4b_3x3.get_weights_view(), inception_4b_3x3.get_grad_weights_view(),np.zeros_like(inception_4b_3x3.get_weights_view()), base_lr, momentum, batch_size)
sgd_update(inception_4b_5x5_reduce.get_weights_view(), inception_4b_5x5_reduce.get_grad_weights_view(),np.zeros_like(inception_4b_5x5_reduce.get_weights_view()), base_lr, momentum, batch_size)
sgd_update(inception_4b_5x5.get_weights_view(), inception_4b_5x5.get_grad_weights_view(),np.zeros_like(inception_4b_5x5.get_weights_view()), base_lr, momentum, batch_size)
sgd_update(inception_4b_pool_proj.get_weights_view(), inception_4b_pool_proj.get_grad_weights_view(),np.zeros_like(inception_4b_pool_proj.get_weights_view()), base_lr, momentum, batch_size)
sgd_update(inception_4c_1x1.get_weights_view(), inception_4c_1x1.get_grad_weights_view(),np.zeros_like(inception_4c_1x1.get_weights_view()), base_lr, momentum, batch_size)
sgd_update(inception_4c_3x3_reduce.get_weights_view(), inception_4c_3x3_reduce.get_grad_weights_view(),np.zeros_like(inception_4c_3x3_reduce.get_weights_view()), base_lr, momentum, batch_size)
sgd_update(inception_4c_3x3.get_weights_view(), inception_4c_3x3.get_grad_weights_view(),np.zeros_like(inception_4c_3x3.get_weights_view()), base_lr, momentum, batch_size)
sgd_update(inception_4c_5x5_reduce.get_weights_view(), inception_4c_5x5_reduce.get_grad_weights_view(),np.zeros_like(inception_4c_5x5_reduce.get_weights_view()), base_lr, momentum, batch_size)
sgd_update(inception_4c_5x5.get_weights_view(), inception_4c_5x5.get_grad_weights_view(),np.zeros_like(inception_4c_5x5.get_weights_view()), base_lr, momentum, batch_size)
sgd_update(inception_4c_pool_proj.get_weights_view(), inception_4c_pool_proj.get_grad_weights_view(),np.zeros_like(inception_4c_pool_proj.get_weights_view()), base_lr, momentum, batch_size)
sgd_update(inception_4d_1x1.get_weights_view(), inception_4d_1x1.get_grad_weights_view(),np.zeros_like(inception_4d_1x1.get_weights_view()), base_lr, momentum, batch_size)
sgd_update(inception_4d_3x3_reduce.get_weights_view(), inception_4d_3x3_reduce.get_grad_weights_view(),np.zeros_like(inception_4d_3x3_reduce.get_weights_view()), base_lr, momentum, batch_size)
sgd_update(inception_4d_3x3.get_weights_view(), inception_4d_3x3.get_grad_weights_view(),np.zeros_like(inception_4d_3x3.get_weights_view()), base_lr, momentum, batch_size)
sgd_update(inception_4d_5x5_reduce.get_weights_view(), inception_4d_5x5_reduce.get_grad_weights_view(),np.zeros_like(inception_4d_5x5_reduce.get_weights_view()), base_lr, momentum, batch_size)
sgd_update(inception_4d_5x5.get_weights_view(), inception_4d_5x5.get_grad_weights_view(),np.zeros_like(inception_4d_5x5.get_weights_view()), base_lr, momentum, batch_size)
sgd_update(inception_4d_pool_proj.get_weights_view(), inception_4d_pool_proj.get_grad_weights_view(),np.zeros_like(inception_4d_pool_proj.get_weights_view()), base_lr, momentum, batch_size)
sgd_update(loss2_conv.get_weights_view(), loss2_conv.get_grad_weights_view(),np.zeros_like(loss2_conv.get_weights_view()), base_lr, momentum, batch_size)
sgd_update(loss2_fc.get_weights_view(), loss2_fc.get_grad_weights_view(),np.zeros_like(loss2_fc.get_weights_view()), base_lr, momentum, batch_size)
sgd_update(loss2_classifier.get_weights_view(), loss2_classifier.get_grad_weights_view(),np.zeros_like(loss2_classifier.get_weights_view()), base_lr, momentum, batch_size)
sgd_update(inception_4e_1x1.get_weights_view(), inception_4e_1x1.get_grad_weights_view(),np.zeros_like(inception_4e_1x1.get_weights_view()), base_lr, momentum, batch_size)
sgd_update(inception_4e_3x3_reduce.get_weights_view(), inception_4e_3x3_reduce.get_grad_weights_view(),np.zeros_like(inception_4e_3x3_reduce.get_weights_view()), base_lr, momentum, batch_size)
sgd_update(inception_4e_3x3.get_weights_view(), inception_4e_3x3.get_grad_weights_view(),np.zeros_like(inception_4e_3x3.get_weights_view()), base_lr, momentum, batch_size)
sgd_update(inception_4e_5x5_reduce.get_weights_view(), inception_4e_5x5_reduce.get_grad_weights_view(),np.zeros_like(inception_4e_5x5_reduce.get_weights_view()), base_lr, momentum, batch_size)
sgd_update(inception_4e_5x5.get_weights_view(), inception_4e_5x5.get_grad_weights_view(),np.zeros_like(inception_4e_5x5.get_weights_view()), base_lr, momentum, batch_size)
sgd_update(inception_4e_pool_proj.get_weights_view(), inception_4e_pool_proj.get_grad_weights_view(),np.zeros_like(inception_4e_pool_proj.get_weights_view()), base_lr, momentum, batch_size)
sgd_update(inception_5a_1x1.get_weights_view(), inception_5a_1x1.get_grad_weights_view(),np.zeros_like(inception_5a_1x1.get_weights_view()), base_lr, momentum, batch_size)
sgd_update(inception_5a_3x3_reduce.get_weights_view(), inception_5a_3x3_reduce.get_grad_weights_view(),np.zeros_like(inception_5a_3x3_reduce.get_weights_view()), base_lr, momentum, batch_size)
sgd_update(inception_5a_3x3.get_weights_view(), inception_5a_3x3.get_grad_weights_view(),np.zeros_like(inception_5a_3x3.get_weights_view()), base_lr, momentum, batch_size)
sgd_update(inception_5a_5x5_reduce.get_weights_view(), inception_5a_5x5_reduce.get_grad_weights_view(),np.zeros_like(inception_5a_5x5_reduce.get_weights_view()), base_lr, momentum, batch_size)
sgd_update(inception_5a_5x5.get_weights_view(), inception_5a_5x5.get_grad_weights_view(),np.zeros_like(inception_5a_5x5.get_weights_view()), base_lr, momentum, batch_size)
sgd_update(inception_5a_pool_proj.get_weights_view(), inception_5a_pool_proj.get_grad_weights_view(),np.zeros_like(inception_5a_pool_proj.get_weights_view()), base_lr, momentum, batch_size)
sgd_update(inception_5b_1x1.get_weights_view(), inception_5b_1x1.get_grad_weights_view(),np.zeros_like(inception_5b_1x1.get_weights_view()), base_lr, momentum, batch_size)
sgd_update(inception_5b_3x3_reduce.get_weights_view(), inception_5b_3x3_reduce.get_grad_weights_view(),np.zeros_like(inception_5b_3x3_reduce.get_weights_view()), base_lr, momentum, batch_size)
sgd_update(inception_5b_3x3.get_weights_view(), inception_5b_3x3.get_grad_weights_view(),np.zeros_like(inception_5b_3x3.get_weights_view()), base_lr, momentum, batch_size)
sgd_update(inception_5b_5x5_reduce.get_weights_view(), inception_5b_5x5_reduce.get_grad_weights_view(),np.zeros_like(inception_5b_5x5_reduce.get_weights_view()), base_lr, momentum, batch_size)
sgd_update(inception_5b_5x5.get_weights_view(), inception_5b_5x5.get_grad_weights_view(),np.zeros_like(inception_5b_5x5.get_weights_view()), base_lr, momentum, batch_size)
sgd_update(inception_5b_pool_proj.get_weights_view(), inception_5b_pool_proj.get_grad_weights_view(),np.zeros_like(inception_5b_pool_proj.get_weights_view()), base_lr, momentum, batch_size)
sgd_update(loss3_classifier.get_weights_view(), loss3_classifier.get_grad_weights_view(),np.zeros_like(loss3_classifier.get_weights_view()), base_lr, momentum, batch_size)

assert np.allclose(net2.blobs["data"].data, data.get_value(), 1e-3, 1e-3)
assert np.allclose(net2.blobs["conv1/7x7_s2"].data, conv1_7x7_s2.get_value(), 1e-3, 1e-3)
assert np.allclose(net2.blobs["pool1/3x3_s2"].data, pool1_3x3_s2.get_value(), 1e-3, 1e-3)
assert np.allclose(net2.blobs["pool1/norm1"].data, pool1_norm1.get_value(), 1e-3, 1e-3)
assert np.allclose(net2.blobs["conv2/3x3_reduce"].data, conv2_3x3_reduce.get_value(), 1e-3, 1e-3)
assert np.allclose(net2.blobs["conv2/3x3"].data, conv2_3x3.get_value(), 1e-3, 1e-3)
assert np.allclose(net2.blobs["conv2/norm2"].data, conv2_norm2.get_value(), 1e-3, 1e-3)
assert np.allclose(net2.blobs["pool2/3x3_s2"].data, pool2_3x3_s2.get_value(), 1e-3, 1e-3)
assert np.allclose(net2.blobs["inception_3a/1x1"].data, inception_3a_1x1.get_value(), 1e-3, 1e-3)
assert np.allclose(net2.blobs["inception_3a/3x3_reduce"].data, inception_3a_3x3_reduce.get_value(), 1e-3, 1e-3)
assert np.allclose(net2.blobs["inception_3a/3x3"].data, inception_3a_3x3.get_value(), 1e-3, 1e-3)
assert np.allclose(net2.blobs["inception_3a/5x5_reduce"].data, inception_3a_5x5_reduce.get_value(), 1e-3, 1e-3)
assert np.allclose(net2.blobs["inception_3a/5x5"].data, inception_3a_5x5.get_value(), 1e-3, 1e-3)
assert np.allclose(net2.blobs["inception_3a/pool"].data, inception_3a_pool.get_value(), 1e-3, 1e-3)
assert np.allclose(net2.blobs["inception_3a/pool_proj"].data, inception_3a_pool_proj.get_value(), 1e-3, 1e-3)
assert np.allclose(net2.blobs["inception_3a/output"].data, inception_3a_output.get_value(), 1e-3, 1e-3)
assert np.allclose(net2.blobs["inception_3b/1x1"].data, inception_3b_1x1.get_value(), 1e-3, 1e-3)
assert np.allclose(net2.blobs["inception_3b/3x3_reduce"].data, inception_3b_3x3_reduce.get_value(), 1e-3, 1e-3)
assert np.allclose(net2.blobs["inception_3b/3x3"].data, inception_3b_3x3.get_value(), 1e-3, 1e-3)
assert np.allclose(net2.blobs["inception_3b/5x5_reduce"].data, inception_3b_5x5_reduce.get_value(), 1e-3, 1e-3)
assert np.allclose(net2.blobs["inception_3b/5x5"].data, inception_3b_5x5.get_value(), 1e-3, 1e-3)
assert np.allclose(net2.blobs["inception_3b/pool"].data, inception_3b_pool.get_value(), 1e-3, 1e-3)
assert np.allclose(net2.blobs["inception_3b/pool_proj"].data, inception_3b_pool_proj.get_value(), 1e-3, 1e-3)
assert np.allclose(net2.blobs["inception_3b/output"].data, inception_3b_output.get_value(), 1e-3, 1e-3)
assert np.allclose(net2.blobs["pool3/3x3_s2"].data, pool3_3x3_s2.get_value(), 1e-3, 1e-3)
assert np.allclose(net2.blobs["inception_4a/1x1"].data, inception_4a_1x1.get_value(), 1e-3, 1e-3)
assert np.allclose(net2.blobs["inception_4a/3x3_reduce"].data, inception_4a_3x3_reduce.get_value(), 1e-3, 1e-3)
assert np.allclose(net2.blobs["inception_4a/3x3"].data, inception_4a_3x3.get_value(), 1e-3, 1e-3)
assert np.allclose(net2.blobs["inception_4a/5x5_reduce"].data, inception_4a_5x5_reduce.get_value(), 1e-3, 1e-3)
assert np.allclose(net2.blobs["inception_4a/5x5"].data, inception_4a_5x5.get_value(), 1e-3, 1e-3)
assert np.allclose(net2.blobs["inception_4a/pool"].data, inception_4a_pool.get_value(), 1e-3, 1e-3)
assert np.allclose(net2.blobs["inception_4a/pool_proj"].data, inception_4a_pool_proj.get_value(), 1e-3, 1e-3)
assert np.allclose(net2.blobs["inception_4a/output"].data, inception_4a_output.get_value(), 1e-3, 1e-3)
assert np.allclose(net2.blobs["loss1/ave_pool"].data, loss1_ave_pool.get_value(), 1e-3, 1e-3)
assert np.allclose(net2.blobs["loss1/conv"].data, loss1_conv.get_value(), 1e-3, 1e-3)
assert np.allclose(net2.blobs["loss1/fc"].data, loss1_fc.get_value(), 1e-3, 1e-3)
assert np.allclose(net2.blobs["loss1/classifier"].data, loss1_classifier.get_value(), 1e-3, 1e-3)
assert np.allclose(net2.blobs["inception_4b/1x1"].data, inception_4b_1x1.get_value(), 1e-3, 1e-3)
assert np.allclose(net2.blobs["inception_4b/3x3_reduce"].data, inception_4b_3x3_reduce.get_value(), 1e-3, 1e-3)
assert np.allclose(net2.blobs["inception_4b/3x3"].data, inception_4b_3x3.get_value(), 1e-3, 1e-3)
assert np.allclose(net2.blobs["inception_4b/5x5_reduce"].data, inception_4b_5x5_reduce.get_value(), 1e-3, 1e-3)
assert np.allclose(net2.blobs["inception_4b/5x5"].data, inception_4b_5x5.get_value(), 1e-3, 1e-3)
assert np.allclose(net2.blobs["inception_4b/5x5"].data, inception_4b_5x5.get_value(), 1e-3, 1e-3)
assert np.allclose(net2.blobs["inception_4b/pool"].data, inception_4b_pool.get_value(), 1e-3, 1e-3)
assert np.allclose(net2.blobs["inception_4b/pool"].data, inception_4b_pool.get_value(), 1e-3, 1e-3)
assert np.allclose(net2.blobs["inception_4b/pool_proj"].data, inception_4b_pool_proj.get_value(), 1e-3, 1e-3)
assert np.allclose(net2.blobs["inception_4b/pool_proj"].data, inception_4b_pool_proj.get_value(), 1e-3, 1e-3)
assert np.allclose(net2.blobs["inception_4b/output"].data, inception_4b_output.get_value(), 1e-3, 1e-3)
assert np.allclose(net2.blobs["inception_4b/output"].data, inception_4b_output.get_value(), 1e-3, 1e-3)
assert np.allclose(net2.blobs["inception_4c/1x1"].data, inception_4c_1x1.get_value(), 1e-3, 1e-3)
assert np.allclose(net2.blobs["inception_4c/1x1"].data, inception_4c_1x1.get_value(), 1e-3, 1e-3)
assert np.allclose(net2.blobs["inception_4c/3x3_reduce"].data, inception_4c_3x3_reduce.get_value(), 1e-3, 1e-3)
assert np.allclose(net2.blobs["inception_4c/3x3"].data, inception_4c_3x3.get_value(), 1e-3, 1e-3)
assert np.allclose(net2.blobs["inception_4c/5x5_reduce"].data, inception_4c_5x5_reduce.get_value(), 1e-3, 1e-3)
assert np.allclose(net2.blobs["inception_4c/5x5"].data, inception_4c_5x5.get_value(), 1e-3, 1e-3)
assert np.allclose(net2.blobs["inception_4c/pool"].data, inception_4c_pool.get_value(), 1e-3, 1e-3)
assert np.allclose(net2.blobs["inception_4c/pool_proj"].data, inception_4c_pool_proj.get_value(), 1e-3, 1e-3)
assert np.allclose(net2.blobs["inception_4c/output"].data, inception_4c_output.get_value(), 1e-3, 1e-3)
assert np.allclose(net2.blobs["inception_4d/1x1"].data, inception_4d_1x1.get_value(), 1e-3, 1e-3)
assert np.allclose(net2.blobs["inception_4d/3x3_reduce"].data, inception_4d_3x3_reduce.get_value(), 1e-3, 1e-3)
assert np.allclose(net2.blobs["inception_4d/3x3"].data, inception_4d_3x3.get_value(), 1e-3, 1e-3)
assert np.allclose(net2.blobs["inception_4d/5x5_reduce"].data, inception_4d_5x5_reduce.get_value(), 1e-3, 1e-3)
assert np.allclose(net2.blobs["inception_4d/5x5"].data, inception_4d_5x5.get_value(), 1e-3, 1e-3)
assert np.allclose(net2.blobs["inception_4d/pool"].data, inception_4d_pool.get_value(), 1e-3, 1e-3)
assert np.allclose(net2.blobs["inception_4d/pool_proj"].data, inception_4d_pool_proj.get_value(), 1e-3, 1e-3)
assert np.allclose(net2.blobs["inception_4d/output"].data, inception_4d_output.get_value(), 1e-3, 1e-3)
assert np.allclose(net2.blobs["loss2/ave_pool"].data, loss2_ave_pool.get_value(), 1e-3, 1e-3)
assert np.allclose(net2.blobs["loss2/conv"].data, loss2_conv.get_value(), 1e-3, 1e-3)
assert np.allclose(net2.blobs["loss2/fc"].data, loss2_fc.get_value(), 1e-3, 1e-3)
assert np.allclose(net2.blobs["loss2/classifier"].data, loss2_classifier.get_value(), 1e-3, 1e-3)
assert np.allclose(net2.blobs["inception_4e/1x1"].data, inception_4e_1x1.get_value(), 1e-3, 1e-3)
assert np.allclose(net2.blobs["inception_4e/3x3_reduce"].data, inception_4e_3x3_reduce.get_value(), 1e-3, 1e-3)
assert np.allclose(net2.blobs["inception_4e/3x3"].data, inception_4e_3x3.get_value(), 1e-3, 1e-3)
assert np.allclose(net2.blobs["inception_4e/5x5_reduce"].data, inception_4e_5x5_reduce.get_value(), 1e-3, 1e-3)
assert np.allclose(net2.blobs["inception_4e/5x5"].data, inception_4e_5x5.get_value(), 1e-3, 1e-3)
assert np.allclose(net2.blobs["inception_4e/pool"].data, inception_4e_pool.get_value(), 1e-3, 1e-3)
assert np.allclose(net2.blobs["inception_4e/pool_proj"].data, inception_4e_pool_proj.get_value(), 1e-3, 1e-3)
assert np.allclose(net2.blobs["inception_4e/output"].data, inception_4e_output.get_value(), 1e-3, 1e-3)
assert np.allclose(net2.blobs["pool4/3x3_s2"].data, pool4_3x3_s2.get_value(), 1e-3, 1e-3)
assert np.allclose(net2.blobs["inception_5a/1x1"].data, inception_5a_1x1.get_value(), 1e-3, 1e-3)
assert np.allclose(net2.blobs["inception_5a/3x3_reduce"].data, inception_5a_3x3_reduce.get_value(), 1e-3, 1e-3)
assert np.allclose(net2.blobs["inception_5a/3x3"].data, inception_5a_3x3.get_value(), 1e-3, 1e-3)
assert np.allclose(net2.blobs["inception_5a/5x5_reduce"].data, inception_5a_5x5_reduce.get_value(), 1e-3, 1e-3)
assert np.allclose(net2.blobs["inception_5a/5x5"].data, inception_5a_5x5.get_value(), 1e-3, 1e-3)
assert np.allclose(net2.blobs["inception_5a/pool"].data, inception_5a_pool.get_value(), 1e-3, 1e-3)
assert np.allclose(net2.blobs["inception_5a/pool_proj"].data, inception_5a_pool_proj.get_value(), 1e-3, 1e-3)
assert np.allclose(net2.blobs["inception_5a/output"].data, inception_5a_output.get_value(), 1e-3, 1e-3)
assert np.allclose(net2.blobs["inception_5b/1x1"].data, inception_5b_1x1.get_value(), 1e-3, 1e-3)
assert np.allclose(net2.blobs["inception_5b/3x3_reduce"].data, inception_5b_3x3_reduce.get_value(), 1e-3, 1e-3)
assert np.allclose(net2.blobs["inception_5b/3x3"].data, inception_5b_3x3.get_value(), 1e-3, 1e-3)
assert np.allclose(net2.blobs["inception_5b/5x5_reduce"].data, inception_5b_5x5_reduce.get_value(), 1e-3, 1e-3)
assert np.allclose(net2.blobs["inception_5b/5x5"].data, inception_5b_5x5.get_value(), 1e-3, 1e-3)
assert np.allclose(net2.blobs["inception_5b/pool"].data, inception_5b_pool.get_value(), 1e-3, 1e-3)
assert np.allclose(net2.blobs["inception_5b/pool_proj"].data, inception_5b_pool_proj.get_value(), 1e-3, 1e-3)
assert np.allclose(net2.blobs["inception_5b/output"].data, inception_5b_output.get_value(), 1e-3, 1e-3)
assert np.allclose(net2.blobs["pool5/7x7_s1"].data, pool5_7x7_s1.get_value(), 1e-3, 1e-3)
assert np.allclose(net2.blobs["loss3/classifier"].data, loss3_classifier.get_value(), 1e-3, 1e-3)



#assert np.allclose(net2.blobs["data"].data, data.get_value(), 1e-3, 1e-3)
#print (net2.blobs["conv1/7x7_s2"].diff)
#print(conv1_7x7_s2.get_grad())
assert np.allclose(net2.blobs["conv1/7x7_s2"].diff, conv1_7x7_s2.get_grad(), 1e-3, 1e-3)
assert np.allclose(net2.blobs["pool1/3x3_s2"].diff, pool1_3x3_s2.get_grad(), 1e-3, 1e-3)
assert np.allclose(net2.blobs["pool1/norm1"].diff, pool1_norm1.get_grad(), 1e-3, 1e-3)
assert np.allclose(net2.blobs["conv2/3x3_reduce"].diff, conv2_3x3_reduce.get_grad(), 1e-3, 1e-3)
assert np.allclose(net2.blobs["conv2/3x3"].diff, conv2_3x3.get_grad(), 1e-3, 1e-3)
assert np.allclose(net2.blobs["conv2/norm2"].diff, conv2_norm2.get_grad(), 1e-3, 1e-3)
assert np.allclose(net2.blobs["pool2/3x3_s2"].diff, pool2_3x3_s2.get_grad(), 1e-3, 1e-3)
assert np.allclose(net2.blobs["inception_3a/1x1"].diff, inception_3a_1x1.get_grad(), 1e-3, 1e-3)
assert np.allclose(net2.blobs["inception_3a/3x3_reduce"].diff, inception_3a_3x3_reduce.get_grad(), 1e-3, 1e-3)
assert np.allclose(net2.blobs["inception_3a/3x3"].diff, inception_3a_3x3.get_grad(), 1e-3, 1e-3)
assert np.allclose(net2.blobs["inception_3a/5x5_reduce"].diff, inception_3a_5x5_reduce.get_grad(), 1e-3, 1e-3)
assert np.allclose(net2.blobs["inception_3a/5x5"].diff, inception_3a_5x5.get_grad(), 1e-3, 1e-3)
assert np.allclose(net2.blobs["inception_3a/pool"].diff, inception_3a_pool.get_grad(), 1e-3, 1e-3)
assert np.allclose(net2.blobs["inception_3a/pool_proj"].diff, inception_3a_pool_proj.get_grad(), 1e-3, 1e-3)
assert np.allclose(net2.blobs["inception_3a/output"].diff, inception_3a_output.get_grad(), 1e-3, 1e-3)
assert np.allclose(net2.blobs["inception_3b/1x1"].diff, inception_3b_1x1.get_grad(), 1e-3, 1e-3)
assert np.allclose(net2.blobs["inception_3b/3x3_reduce"].diff, inception_3b_3x3_reduce.get_grad(), 1e-3, 1e-3)
assert np.allclose(net2.blobs["inception_3b/3x3"].diff, inception_3b_3x3.get_grad(), 1e-3, 1e-3)
assert np.allclose(net2.blobs["inception_3b/5x5_reduce"].diff, inception_3b_5x5_reduce.get_grad(), 1e-3, 1e-3)
assert np.allclose(net2.blobs["inception_3b/5x5"].diff, inception_3b_5x5.get_grad(), 1e-3, 1e-3)
assert np.allclose(net2.blobs["inception_3b/pool"].diff, inception_3b_pool.get_grad(), 1e-3, 1e-3)
assert np.allclose(net2.blobs["inception_3b/pool_proj"].diff, inception_3b_pool_proj.get_grad(), 1e-3, 1e-3)
assert np.allclose(net2.blobs["inception_3b/output"].diff, inception_3b_output.get_grad(), 1e-3, 1e-3)
assert np.allclose(net2.blobs["pool3/3x3_s2"].diff, pool3_3x3_s2.get_grad(), 1e-3, 1e-3)
assert np.allclose(net2.blobs["inception_4a/1x1"].diff, inception_4a_1x1.get_grad(), 1e-3, 1e-3)
assert np.allclose(net2.blobs["inception_4a/3x3_reduce"].diff, inception_4a_3x3_reduce.get_grad(), 1e-3, 1e-3)
assert np.allclose(net2.blobs["inception_4a/3x3"].diff, inception_4a_3x3.get_grad(), 1e-3, 1e-3)
assert np.allclose(net2.blobs["inception_4a/5x5_reduce"].diff, inception_4a_5x5_reduce.get_grad(), 1e-3, 1e-3)
assert np.allclose(net2.blobs["inception_4a/5x5"].diff, inception_4a_5x5.get_grad(), 1e-3, 1e-3)
assert np.allclose(net2.blobs["inception_4a/pool"].diff, inception_4a_pool.get_grad(), 1e-3, 1e-3)
assert np.allclose(net2.blobs["inception_4a/pool_proj"].diff, inception_4a_pool_proj.get_grad(), 1e-3, 1e-3)
assert np.allclose(net2.blobs["inception_4a/output"].diff, inception_4a_output.get_grad(), 1e-3, 1e-3)
assert np.allclose(net2.blobs["loss1/ave_pool"].diff, loss1_ave_pool.get_grad(), 1e-3, 1e-3)
assert np.allclose(net2.blobs["loss1/conv"].diff, loss1_conv.get_grad(), 1e-3, 1e-3)
assert np.allclose(net2.blobs["loss1/fc"].diff, loss1_fc.get_grad(), 1e-3, 1e-3)
assert np.allclose(net2.blobs["loss1/classifier"].diff, loss1_classifier.get_grad(), 1e-3, 1e-3)
assert np.allclose(net2.blobs["inception_4b/1x1"].diff, inception_4b_1x1.get_grad(), 1e-3, 1e-3)
assert np.allclose(net2.blobs["inception_4b/3x3_reduce"].diff, inception_4b_3x3_reduce.get_grad(), 1e-3, 1e-3)
assert np.allclose(net2.blobs["inception_4b/3x3"].diff, inception_4b_3x3.get_grad(), 1e-3, 1e-3)
assert np.allclose(net2.blobs["inception_4b/5x5_reduce"].diff, inception_4b_5x5_reduce.get_grad(), 1e-3, 1e-3)
assert np.allclose(net2.blobs["inception_4b/5x5"].diff, inception_4b_5x5.get_grad(), 1e-3, 1e-3)
assert np.allclose(net2.blobs["inception_4b/5x5"].diff, inception_4b_5x5.get_grad(), 1e-3, 1e-3)
assert np.allclose(net2.blobs["inception_4b/pool"].diff, inception_4b_pool.get_grad(), 1e-3, 1e-3)
assert np.allclose(net2.blobs["inception_4b/pool"].diff, inception_4b_pool.get_grad(), 1e-3, 1e-3)
assert np.allclose(net2.blobs["inception_4b/pool_proj"].diff, inception_4b_pool_proj.get_grad(), 1e-3, 1e-3)
assert np.allclose(net2.blobs["inception_4b/pool_proj"].diff, inception_4b_pool_proj.get_grad(), 1e-3, 1e-3)
assert np.allclose(net2.blobs["inception_4b/output"].diff, inception_4b_output.get_grad(), 1e-3, 1e-3)
assert np.allclose(net2.blobs["inception_4b/output"].diff, inception_4b_output.get_grad(), 1e-3, 1e-3)
assert np.allclose(net2.blobs["inception_4c/1x1"].diff, inception_4c_1x1.get_grad(), 1e-3, 1e-3)
assert np.allclose(net2.blobs["inception_4c/1x1"].diff, inception_4c_1x1.get_grad(), 1e-3, 1e-3)
assert np.allclose(net2.blobs["inception_4c/3x3_reduce"].diff, inception_4c_3x3_reduce.get_grad(), 1e-3, 1e-3)
assert np.allclose(net2.blobs["inception_4c/3x3"].diff, inception_4c_3x3.get_grad(), 1e-3, 1e-3)
assert np.allclose(net2.blobs["inception_4c/5x5_reduce"].diff, inception_4c_5x5_reduce.get_grad(), 1e-3, 1e-3)
assert np.allclose(net2.blobs["inception_4c/5x5"].diff, inception_4c_5x5.get_grad(), 1e-3, 1e-3)
assert np.allclose(net2.blobs["inception_4c/pool"].diff, inception_4c_pool.get_grad(), 1e-3, 1e-3)
assert np.allclose(net2.blobs["inception_4c/pool_proj"].diff, inception_4c_pool_proj.get_grad(), 1e-3, 1e-3)
assert np.allclose(net2.blobs["inception_4c/output"].diff, inception_4c_output.get_grad(), 1e-3, 1e-3)
assert np.allclose(net2.blobs["inception_4d/1x1"].diff, inception_4d_1x1.get_grad(), 1e-3, 1e-3)
assert np.allclose(net2.blobs["inception_4d/3x3_reduce"].diff, inception_4d_3x3_reduce.get_grad(), 1e-3, 1e-3)
assert np.allclose(net2.blobs["inception_4d/3x3"].diff, inception_4d_3x3.get_grad(), 1e-3, 1e-3)
assert np.allclose(net2.blobs["inception_4d/5x5_reduce"].diff, inception_4d_5x5_reduce.get_grad(), 1e-3, 1e-3)
assert np.allclose(net2.blobs["inception_4d/5x5"].diff, inception_4d_5x5.get_grad(), 1e-3, 1e-3)
assert np.allclose(net2.blobs["inception_4d/pool"].diff, inception_4d_pool.get_grad(), 1e-3, 1e-3)
assert np.allclose(net2.blobs["inception_4d/pool_proj"].diff, inception_4d_pool_proj.get_grad(), 1e-3, 1e-3)
assert np.allclose(net2.blobs["inception_4d/output"].diff, inception_4d_output.get_grad(), 1e-3, 1e-3)
assert np.allclose(net2.blobs["loss2/ave_pool"].diff, loss2_ave_pool.get_grad(), 1e-3, 1e-3)
assert np.allclose(net2.blobs["loss2/conv"].diff, loss2_conv.get_grad(), 1e-3, 1e-3)
assert np.allclose(net2.blobs["loss2/fc"].diff, loss2_fc.get_grad(), 1e-3, 1e-3)
assert np.allclose(net2.blobs["loss2/classifier"].diff, loss2_classifier.get_grad(), 1e-3, 1e-3)
assert np.allclose(net2.blobs["inception_4e/1x1"].diff, inception_4e_1x1.get_grad(), 1e-3, 1e-3)
assert np.allclose(net2.blobs["inception_4e/3x3_reduce"].diff, inception_4e_3x3_reduce.get_grad(), 1e-3, 1e-3)
assert np.allclose(net2.blobs["inception_4e/3x3"].diff, inception_4e_3x3.get_grad(), 1e-3, 1e-3)
assert np.allclose(net2.blobs["inception_4e/5x5_reduce"].diff, inception_4e_5x5_reduce.get_grad(), 1e-3, 1e-3)
assert np.allclose(net2.blobs["inception_4e/5x5"].diff, inception_4e_5x5.get_grad(), 1e-3, 1e-3)
assert np.allclose(net2.blobs["inception_4e/pool"].diff, inception_4e_pool.get_grad(), 1e-3, 1e-3)
assert np.allclose(net2.blobs["inception_4e/pool_proj"].diff, inception_4e_pool_proj.get_grad(), 1e-3, 1e-3)
assert np.allclose(net2.blobs["inception_4e/output"].diff, inception_4e_output.get_grad(), 1e-3, 1e-3)
assert np.allclose(net2.blobs["pool4/3x3_s2"].diff, pool4_3x3_s2.get_grad(), 1e-3, 1e-3)
assert np.allclose(net2.blobs["inception_5a/1x1"].diff, inception_5a_1x1.get_grad(), 1e-3, 1e-3)
assert np.allclose(net2.blobs["inception_5a/3x3_reduce"].diff, inception_5a_3x3_reduce.get_grad(), 1e-3, 1e-3)
assert np.allclose(net2.blobs["inception_5a/3x3"].diff, inception_5a_3x3.get_grad(), 1e-3, 1e-3)
assert np.allclose(net2.blobs["inception_5a/5x5_reduce"].diff, inception_5a_5x5_reduce.get_grad(), 1e-3, 1e-3)
assert np.allclose(net2.blobs["inception_5a/5x5"].diff, inception_5a_5x5.get_grad(), 1e-3, 1e-3)
assert np.allclose(net2.blobs["inception_5a/pool"].diff, inception_5a_pool.get_grad(), 1e-3, 1e-3)
assert np.allclose(net2.blobs["inception_5a/pool_proj"].diff, inception_5a_pool_proj.get_grad(), 1e-3, 1e-3)
assert np.allclose(net2.blobs["inception_5a/output"].diff, inception_5a_output.get_grad(), 1e-3, 1e-3)
assert np.allclose(net2.blobs["inception_5b/1x1"].diff, inception_5b_1x1.get_grad(), 1e-3, 1e-3)
assert np.allclose(net2.blobs["inception_5b/3x3_reduce"].diff, inception_5b_3x3_reduce.get_grad(), 1e-3, 1e-3)
assert np.allclose(net2.blobs["inception_5b/3x3"].diff, inception_5b_3x3.get_grad(), 1e-3, 1e-3)
assert np.allclose(net2.blobs["inception_5b/5x5_reduce"].diff, inception_5b_5x5_reduce.get_grad(), 1e-3, 1e-3)
assert np.allclose(net2.blobs["inception_5b/5x5"].diff, inception_5b_5x5.get_grad(), 1e-3, 1e-3)
assert np.allclose(net2.blobs["inception_5b/pool"].diff, inception_5b_pool.get_grad(), 1e-3, 1e-3)
assert np.allclose(net2.blobs["inception_5b/pool_proj"].diff, inception_5b_pool_proj.get_grad(), 1e-3, 1e-3)
assert np.allclose(net2.blobs["inception_5b/output"].diff, inception_5b_output.get_grad(), 1e-3, 1e-3)
assert np.allclose(net2.blobs["pool5/7x7_s1"].diff, pool5_7x7_s1.get_grad(), 1e-3, 1e-3)
assert np.allclose(net2.blobs["loss3/classifier"].diff, loss3_classifier.get_grad(), 1e-3, 1e-3)
'''
print(net2.blobs["pool1/3x3_s2"].diff)
print(pool1_3x3_s2.get_grad())
print(net2.blobs["pool1/norm1"].diff)
print(pool1_norm1.get_grad())
print(net2.blobs["conv2/3x3_reduce"].diff)
print(conv2_3x3_reduce.get_grad())
print(net2.blobs["conv2/3x3"].diff)
print(conv2_3x3.get_grad())

print(net2.blobs["conv2/norm2"].diff)

print(conv2_norm2.get_grad())
print(net2.blobs["pool2/3x3_s2"].diff)
print(pool2_3x3_s2.get_grad())
print(net2.blobs["inception_3a/1x1"].diff)
print(inception_3a_1x1.get_grad())
print(net2.blobs["inception_3a/3x3_reduce"].diff)
print(inception_3a_3x3_reduce.get_grad())
print(net2.blobs["inception_3a/3x3"].diff)
print(inception_3a_3x3.get_grad())
print(net2.blobs["inception_3a/5x5_reduce"].diff)
print(inception_3a_5x5_reduce.get_grad())

print(net2.blobs["inception_3a/5x5"].diff)
print(inception_3a_5x5.get_grad())
print(net2.blobs["inception_3a/pool"].diff)
print(inception_3a_pool.get_grad())
print(net2.blobs["inception_3a/pool_proj"].diff)
print(inception_3a_pool_proj.get_grad())
print(net2.blobs["inception_3a/output"].diff)
print(inception_3a_output.get_grad())
print(net2.blobs["inception_3b/1x1"].diff)
print(inception_3b_1x1.get_grad())
print(net2.blobs["inception_3b/3x3_reduce"].diff)
print(inception_3b_3x3_reduce.get_grad())
print(net2.blobs["inception_3b/3x3"].diff)
print(inception_3b_3x3.get_grad())
print(net2.blobs["inception_3b/5x5_reduce"].diff)
print(inception_3b_5x5_reduce.get_grad())
print(net2.blobs["inception_3b/5x5"].diff)
print(inception_3b_5x5.get_grad())
print(net2.blobs["inception_3b/pool"].diff)
print(inception_3b_pool.get_grad())
print(net2.blobs["inception_3b/pool_proj"].diff)
print(inception_3b_pool_proj.get_grad())
print(net2.blobs["inception_3b/output"].diff)
print(inception_3b_output.get_grad())
print(net2.blobs["pool3/3x3_s2"].diff)
print(pool3_3x3_s2.get_grad())
print(net2.blobs["inception_4a/1x1"].diff)
print(inception_4a_1x1.get_grad())
print(net2.blobs["inception_4a/3x3_reduce"].diff)
print(inception_4a_3x3_reduce.get_grad())
print(net2.blobs["inception_4a/3x3"].diff)
print(inception_4a_3x3.get_grad())
print(net2.blobs["inception_4a/5x5_reduce"].diff)
print(inception_4a_5x5_reduce.get_grad())
print(net2.blobs["inception_4a/5x5"].diff)
print(inception_4a_5x5.get_grad())
print(net2.blobs["inception_4a/pool"].diff)
print(inception_4a_pool.get_grad())
print(net2.blobs["inception_4a/pool_proj"].diff)
print(inception_4a_pool_proj.get_grad())
print(net2.blobs["inception_4a/output"].diff)
print(inception_4a_output.get_grad())
print(net2.blobs["loss1/ave_pool"].diff)
print(loss1_ave_pool.get_grad())
print(net2.blobs["loss1/conv"].diff)
print(loss1_conv.get_grad())
print(net2.blobs["loss1/fc"].diff)
print(loss1_fc.get_grad())
print(net2.blobs["inception_4b/1x1"].diff)
print(inception_4b_1x1.get_grad())
print(net2.blobs["inception_4b/3x3_reduce"].diff)
print(inception_4b_3x3_reduce.get_grad())
print(net2.blobs["inception_4b/3x3"].diff)
print(inception_4b_3x3.get_grad())
print(net2.blobs["inception_4b/5x5_reduce"].diff)
print(inception_4b_5x5_reduce.get_grad())
print(net2.blobs["inception_4b/5x5"].diff)
print(inception_4b_5x5.get_grad())
print(net2.blobs["inception_4b/pool"].diff)
print(inception_4b_pool.get_grad())
print(net2.blobs["inception_4b/pool_proj"].diff)
print(inception_4b_pool_proj.get_grad())
print(net2.blobs["inception_4b/output"].diff)
print(inception_4b_output.get_grad())
print(net2.blobs["inception_4c/1x1"].diff)
print(inception_4c_1x1.get_grad())
print(net2.blobs["inception_4c/3x3_reduce"].diff)
print(inception_4c_3x3_reduce.get_grad())
print(net2.blobs["inception_4c/3x3"].diff)
print(inception_4c_3x3.get_grad())
print(net2.blobs["inception_4c/5x5_reduce"].diff)
print(inception_4c_5x5_reduce.get_grad())
print(net2.blobs["inception_4c/5x5"].diff)
print(inception_4c_5x5.get_grad())
print(net2.blobs["inception_4c/pool"].diff)
print(inception_4c_pool.get_grad())
print(net2.blobs["inception_4c/pool_proj"].diff)
print(inception_4c_pool_proj.get_grad())
print(net2.blobs["inception_4c/output"].diff)
print(inception_4c_output.get_grad())
print(net2.blobs["inception_4d/1x1"].diff)
print(inception_4d_1x1.get_grad())
print(net2.blobs["inception_4d/3x3_reduce"].diff)
print(inception_4d_3x3_reduce.get_grad())
print(net2.blobs["inception_4d/3x3"].diff)
print(inception_4d_3x3.get_grad())
print(net2.blobs["inception_4d/5x5_reduce"].diff)
print(inception_4d_5x5_reduce.get_grad())
print(net2.blobs["inception_4d/5x5"].diff)
print(inception_4d_5x5.get_grad())
print(net2.blobs["inception_4d/pool"].diff)
print(inception_4d_pool.get_grad())
print(net2.blobs["inception_4d/pool_proj"].diff)
print(inception_4d_pool_proj.get_grad())
print(net2.blobs["inception_4d/output"].diff)
print(inception_4d_output.get_grad())
print(net2.blobs["loss2/ave_pool"].diff)
print(loss2_ave_pool.get_grad())
print(net2.blobs["loss2/conv"].diff)
print(loss2_conv.get_grad())
print(net2.blobs["loss2/fc"].diff)
print(loss2_fc.get_grad())
print(net2.blobs["inception_4e/1x1"].diff)
print(inception_4e_1x1.get_grad())
print(net2.blobs["inception_4e/3x3_reduce"].diff)
print(inception_4e_3x3_reduce.get_grad())
print(net2.blobs["inception_4e/3x3"].diff)
print(inception_4e_3x3.get_grad())
print(net2.blobs["inception_4e/5x5_reduce"].diff)
print(inception_4e_5x5_reduce.get_grad())
print(net2.blobs["inception_4e/5x5"].diff)
print(inception_4e_5x5.get_grad())
print(net2.blobs["inception_4e/pool"].diff)
print(inception_4e_pool.get_grad())
print(net2.blobs["inception_4e/pool_proj"].diff)
print(inception_4e_pool_proj.get_grad())
print(net2.blobs["inception_4e/output"].diff)
print(inception_4e_output.get_grad())
print(net2.blobs["pool4/3x3_s2"].diff)
print(pool4_3x3_s2.get_grad())
print(net2.blobs["inception_5a/1x1"].diff)
print(inception_5a_1x1.get_grad())
print(net2.blobs["inception_5a/3x3_reduce"].diff)
print(inception_5a_3x3_reduce.get_grad())
print(net2.blobs["inception_5a/3x3"].diff)
print(inception_5a_3x3.get_grad())
print(net2.blobs["inception_5a/5x5_reduce"].diff)
print(inception_5a_5x5_reduce.get_grad())
print(net2.blobs["inception_5a/5x5"].diff)
print(inception_5a_5x5.get_grad())
print(net2.blobs["inception_5a/pool"].diff)
print(inception_5a_pool.get_grad())
print(net2.blobs["inception_5a/pool_proj"].diff)
print(inception_5a_pool_proj.get_grad())
print(net2.blobs["inception_5a/output"].diff)
print(inception_5a_output.get_grad())
print(net2.blobs["inception_5b/1x1"].diff)
print(inception_5b_1x1.get_grad())

print(net2.blobs["inception_5b/3x3_reduce"].diff)
print(inception_5b_3x3_reduce.get_grad())

print(net2.blobs["inception_5b/3x3"].diff)
print(inception_5b_3x3.get_grad())
print(net2.blobs["inception_5b/5x5_reduce"].diff)
print(inception_5b_5x5_reduce.get_grad())
print(net2.blobs["inception_5b/5x5"].diff)

print(inception_5b_5x5.get_grad())
print(net2.blobs["inception_5b/pool"].diff)
print(inception_5b_pool.get_grad())
print(net2.blobs["inception_5b/pool_proj"].diff)
print(inception_5b_pool_proj.get_grad())


print(net2.blobs["inception_5b/output"].diff)

print(inception_5b_output.get_grad())
print(net2.blobs["pool5/7x7_s1"].diff)
print(pool5_7x7_s1.get_grad())
'''

#assert np.allclose(net2.params["conv1/7x7_s2"][0].data, conv1_7x7_s2.get_weights(), 1e-3, 1e-3)
assert np.allclose(net2.params["conv2/3x3_reduce"][0].data, conv2_3x3_reduce.get_weights(), 1e-3, 1e-3)
assert np.allclose(net2.params["conv2/3x3"][0].data, conv2_3x3.get_weights(), 1e-3, 1e-3)
assert np.allclose(net2.params["inception_3a/1x1"][0].data, inception_3a_1x1.get_weights(), 1e-3, 1e-3)
assert np.allclose(net2.params["inception_3a/3x3_reduce"][0].data, inception_3a_3x3_reduce.get_weights(), 1e-3, 1e-3)
assert np.allclose(net2.params["inception_3a/3x3"][0].data, inception_3a_3x3.get_weights(), 1e-3, 1e-3)
assert np.allclose(net2.params["inception_3a/5x5_reduce"][0].data, inception_3a_5x5_reduce.get_weights(), 1e-3, 1e-3)
assert np.allclose(net2.params["inception_3a/5x5"][0].data, inception_3a_5x5.get_weights(), 1e-3, 1e-3)
assert np.allclose(net2.params["inception_3a/pool_proj"][0].data, inception_3a_pool_proj.get_weights(), 1e-3, 1e-3)
assert np.allclose(net2.params["inception_3b/1x1"][0].data, inception_3b_1x1.get_weights(), 1e-3, 1e-3)
assert np.allclose(net2.params["inception_3b/3x3_reduce"][0].data, inception_3b_3x3_reduce.get_weights(), 1e-3, 1e-3)
assert np.allclose(net2.params["inception_3b/3x3"][0].data, inception_3b_3x3.get_weights(), 1e-3, 1e-3)
assert np.allclose(net2.params["inception_3b/5x5_reduce"][0].data, inception_3b_5x5_reduce.get_weights(), 1e-3, 1e-3)
assert np.allclose(net2.params["inception_3b/5x5"][0].data, inception_3b_5x5.get_weights(), 1e-3, 1e-3)
assert np.allclose(net2.params["inception_3b/pool_proj"][0].data, inception_3b_pool_proj.get_weights(), 1e-3, 1e-3)
assert np.allclose(net2.params["inception_4a/1x1"][0].data, inception_4a_1x1.get_weights(), 1e-3, 1e-3)
assert np.allclose(net2.params["inception_4a/3x3_reduce"][0].data, inception_4a_3x3_reduce.get_weights(), 1e-3, 1e-3)
assert np.allclose(net2.params["inception_4a/3x3"][0].data, inception_4a_3x3.get_weights(), 1e-3, 1e-3)
assert np.allclose(net2.params["inception_4a/5x5_reduce"][0].data, inception_4a_5x5_reduce.get_weights(), 1e-3, 1e-3)
assert np.allclose(net2.params["inception_4a/5x5"][0].data, inception_4a_5x5.get_weights(), 1e-3, 1e-3)
assert np.allclose(net2.params["inception_4a/pool_proj"][0].data, inception_4a_pool_proj.get_weights(), 1e-3, 1e-3)
assert np.allclose(net2.params["loss1/conv"][0].data, loss1_conv.get_weights(), 1e-3, 1e-3)
temp = net2.params["loss1/fc"][0].data.reshape(loss1_fc.get_weights().shape)
assert np.allclose(temp, loss1_fc.get_weights(), 1e-3, 1e-3)
temp = net2.params["loss1/classifier"][0].data.reshape(loss1_classifier.get_weights().shape)
assert np.allclose(temp, loss1_classifier.get_weights(), 1e-3, 1e-3)
assert np.allclose(net2.params["inception_4b/1x1"][0].data, inception_4b_1x1.get_weights(), 1e-3, 1e-3)
assert np.allclose(net2.params["inception_4b/3x3_reduce"][0].data, inception_4b_3x3_reduce.get_weights(), 1e-3, 1e-3)
assert np.allclose(net2.params["inception_4b/3x3"][0].data, inception_4b_3x3.get_weights(), 1e-3, 1e-3)
assert np.allclose(net2.params["inception_4b/5x5_reduce"][0].data, inception_4b_5x5_reduce.get_weights(), 1e-3, 1e-3)
assert np.allclose(net2.params["inception_4b/5x5"][0].data, inception_4b_5x5.get_weights(), 1e-3, 1e-3)
assert np.allclose(net2.params["inception_4b/pool_proj"][0].data, inception_4b_pool_proj.get_weights(), 1e-3, 1e-3)
assert np.allclose(net2.params["inception_4c/1x1"][0].data, inception_4c_1x1.get_weights(), 1e-3, 1e-3)
assert np.allclose(net2.params["inception_4c/3x3_reduce"][0].data, inception_4c_3x3_reduce.get_weights(), 1e-3, 1e-3)
assert np.allclose(net2.params["inception_4c/3x3"][0].data, inception_4c_3x3.get_weights(), 1e-3, 1e-3)
assert np.allclose(net2.params["inception_4c/5x5_reduce"][0].data, inception_4c_5x5_reduce.get_weights(), 1e-3, 1e-3)
assert np.allclose(net2.params["inception_4c/5x5"][0].data, inception_4c_5x5.get_weights(), 1e-3, 1e-3)
assert np.allclose(net2.params["inception_4c/pool_proj"][0].data, inception_4c_pool_proj.get_weights(), 1e-3, 1e-3)
assert np.allclose(net2.params["inception_4d/1x1"][0].data, inception_4d_1x1.get_weights(), 1e-3, 1e-3)
assert np.allclose(net2.params["inception_4d/3x3_reduce"][0].data, inception_4d_3x3_reduce.get_weights(), 1e-3, 1e-3)
assert np.allclose(net2.params["inception_4d/3x3"][0].data, inception_4d_3x3.get_weights(), 1e-3, 1e-3)
assert np.allclose(net2.params["inception_4d/5x5_reduce"][0].data, inception_4d_5x5_reduce.get_weights(), 1e-3, 1e-3)
assert np.allclose(net2.params["inception_4d/5x5"][0].data, inception_4d_5x5.get_weights(), 1e-3, 1e-3)
assert np.allclose(net2.params["inception_4d/pool_proj"][0].data, inception_4d_pool_proj.get_weights(), 1e-3, 1e-3)
assert np.allclose(net2.params["loss2/conv"][0].data, loss2_conv.get_weights(), 1e-3, 1e-3)
temp = net2.params["loss2/fc"][0].data.reshape(loss2_fc.get_weights().shape)

assert np.allclose(temp, loss2_fc.get_weights(), 1e-3, 1e-3)
temp = net2.params["loss2/classifier"][0].data.reshape(loss2_classifier.get_weights().shape) 
assert np.allclose(temp, loss2_classifier.get_weights(), 1e-3, 1e-3)

#assert np.allclose(net2.params["loss2/classifier"][0].data, loss2_classifier.get_weights(), 1e-3, 1e-3)
assert np.allclose(net2.params["inception_4e/1x1"][0].data, inception_4e_1x1.get_weights(), 1e-3, 1e-3)
assert np.allclose(net2.params["inception_4e/3x3_reduce"][0].data, inception_4e_3x3_reduce.get_weights(), 1e-3, 1e-3)
assert np.allclose(net2.params["inception_4e/3x3"][0].data, inception_4e_3x3.get_weights(), 1e-3, 1e-3)
assert np.allclose(net2.params["inception_4e/5x5_reduce"][0].data, inception_4e_5x5_reduce.get_weights(), 1e-3, 1e-3)
assert np.allclose(net2.params["inception_4e/5x5"][0].data, inception_4e_5x5.get_weights(), 1e-3, 1e-3)
assert np.allclose(net2.params["inception_4e/pool_proj"][0].data, inception_4e_pool_proj.get_weights(), 1e-3, 1e-3)
assert np.allclose(net2.params["inception_5a/1x1"][0].data, inception_5a_1x1.get_weights(), 1e-3, 1e-3)
assert np.allclose(net2.params["inception_5a/3x3_reduce"][0].data, inception_5a_3x3_reduce.get_weights(), 1e-3, 1e-3)
assert np.allclose(net2.params["inception_5a/3x3"][0].data, inception_5a_3x3.get_weights(), 1e-3, 1e-3)
assert np.allclose(net2.params["inception_5a/5x5_reduce"][0].data, inception_5a_5x5_reduce.get_weights(), 1e-3, 1e-3)
assert np.allclose(net2.params["inception_5a/5x5"][0].data, inception_5a_5x5.get_weights(), 1e-3, 1e-3)
assert np.allclose(net2.params["inception_5a/pool_proj"][0].data, inception_5a_pool_proj.get_weights(), 1e-3, 1e-3)
assert np.allclose(net2.params["inception_5b/1x1"][0].data, inception_5b_1x1.get_weights(), 1e-3, 1e-3)
assert np.allclose(net2.params["inception_5b/3x3_reduce"][0].data, inception_5b_3x3_reduce.get_weights(), 1e-3, 1e-3)
assert np.allclose(net2.params["inception_5b/3x3"][0].data, inception_5b_3x3.get_weights(), 1e-3, 1e-3)
assert np.allclose(net2.params["inception_5b/5x5_reduce"][0].data, inception_5b_5x5_reduce.get_weights(), 1e-3, 1e-3)
assert np.allclose(net2.params["inception_5b/5x5"][0].data, inception_5b_5x5.get_weights(), 1e-3, 1e-3)
assert np.allclose(net2.params["inception_5b/pool_proj"][0].data, inception_5b_pool_proj.get_weights(), 1e-3, 1e-3)
#assert np.allclose(net2.params["loss3/classifier"][0].data, loss3_classifier.get_weights(), 1e-3, 1e-3)
temp = net2.params["loss3/classifier"][0].data.reshape(loss3_classifier.get_weights().shape) 
assert np.allclose(temp, loss3_classifier.get_weights(), 1e-3, 1e-3)


print("Success\n")

