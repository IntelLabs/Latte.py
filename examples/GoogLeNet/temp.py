import numpy as np
#import caffe
from latte import *
from latte.math import compute_softmax_loss, softmax_loss_backprop
import time
from latte.solvers import sgd_update
import os


 
def weight_update_and_check_output(convs, layers, batch_size):
 
    base_lr = 0.01
    momentum = 0.9
    weight_decay = 0
    lr_w_mult = 1
    lr_b_mult = 1
    momentum_hist = {}
    
    '''
    for layer in net2.params:
        m_w = np.zeros_like(net2.params[layer][0].data)
        m_b = np.zeros_like(net2.params[layer][1].data)
        momentum_hist[layer] = [m_w, m_b]
 
    for layer in net2.params:
        momentum_hist[layer][0] = momentum_hist[layer][0] * momentum + (solver.net.params[layer][0].diff + weight_decay *
                                                       solver.net.params[layer][0].data) * base_lr * lr_w_mult
        momentum_hist[layer][1] = momentum_hist[layer][1] * momentum + (net2.params[layer][1].diff + weight_decay *
                                                       net2.params[layer][1].data) * base_lr * lr_b_mult
        net2.params[layer][0].data[...] -= momentum_hist[layer][0]
        net2.params[layer][1].data[...] -= momentum_hist[layer][1]
    '''

    for param in convs:
        sgd_update(param[1].get_weights_view(), param[1].get_grad_weights_view(),np.zeros_like(param[1].get_weights_view()), base_lr, momentum, batch_size)
        sgd_update(param[1].get_bias_view(), param[1].get_grad_bias_view(),np.zeros_like(param[1].get_bias_view()), base_lr, momentum, batch_size)
    

    ''' 
    for layer in layers:
        #print(layer[0])
        assert np.allclose(net2.blobs[layer[0]].data, layer[1].get_value(), 1e-7, 1e-1)
        assert np.allclose(net2.blobs[layer[0]].diff, layer[1].get_grad(), 1e-7, 1e-1)
    for conv in convs:
        if not (conv[0] == "conv1/7x7_s2"):
            weights = net2.params[conv[0]][0].data
            bias = net2.params[conv[0]][1].data
            grad_wt  = net2.params[conv[0]][0].diff
            grad_bias = net2.params[conv[0]][1].diff
            #print(conv[0])
            if weights.shape == conv[1].get_weights().shape:
                assert np.allclose(net2.params[conv[0]][0].data, conv[1].get_weights(), 1e-7, 1e-2)
            else:
                assert np.allclose(weights.reshape(conv[1].get_weights().shape), conv[1].get_weights(), 1e-7, 1e-2)
            if bias.shape == conv[1].get_bias().shape:
                assert np.allclose(net2.params[conv[0]][1].data, conv[1].get_bias(), 1e-7, 1e-2)
            else:
                assert np.allclose(bias.reshape(conv[1].get_bias().shape), conv[1].get_bias(), 1e-7, 1e-2)
 
            if grad_wt.shape == conv[1].get_grad_weights().shape:
                if not np.allclose(grad_wt, conv[1].get_grad_weights(), 1e-4, 1e-1):
                    assert False
            else:
                assert np.allclose(grad_wt.reshape(conv[1].get_grad_weights().shape), conv[1].get_grad_weights(), 1e-4, 1e-1)
            if grad_bias.shape == conv[1].get_grad_bias().shape:
                assert np.allclose(grad_bias, conv[1].get_grad_bias(), 1e-4, 1e-1)
            else:
                assert np.allclose(grad_bias.reshape(conv[1].get_grad_bias().shape), conv[1].get_grad_bias(), 1e-4, 1e-1)
    '''
def copy_weights(net2, convs):
 
    for conv in convs:
        weights = net2.params[conv[0]][0].data
        bias = net2.params[conv[0]][1].data
 
 
        #print(conv[1])
        #print(bias.shape)
        #print(conv[1].get_bias().shape)
        if conv[0] == "conv1/7x7_s2":
 
            if channels < 8:
                input_channel_pad = 8 - channels
            elif channels % 8 != 0:
                input_channel_pad = 8 - (channels % 8)
            else:
                input_channel_pad = 0
 
            #wt =  np.lib.pad(weights, ((0,0), (0,input_channel_pad), (0, 0), (0, 0)), 'constant', constant_values=(0,))
            wt = conv[1].get_weights()
            for i in range(64):
                for j in range(channels):
                    for k in range(7):
                        for l in range(7):
                            wt[i,j,k,l] = weights[i,j,k,l]
            #for j in [3,4,5,6,7]:
            #      wt[i,j,:,:] = 0.0
 
            conv[1].set_weights(wt)
            #print(weights.shape)
            #print(conv[1].get_weights().shape)
        else:
            if weights.shape == conv[1].get_weights().shape:
                conv[1].set_weights(weights)
            else:
                conv[1].set_weights(weights.reshape(conv[1].get_weights().shape))
 
        if bias.shape == conv[1].get_bias().shape:
            conv[1].set_bias(bias)
        else:
            conv[1].set_bias(bias.reshape(conv[1].get_bias().shape))


batch_size = 128
net = Net(batch_size)
channels=3
height=224
width=224

#caffe_root ='/nfs_home/avenkat/caffe/'
#caffe.set_mode_cpu()
 
data = MemoryDataLayer(net, (3, 224, 224))
conv1_7x7_s2 = ConvLayer(net, data, num_filters=64, kernel=7, stride=2, pad=3)
conv1_relu_7x7 = ReLULayer(net, conv1_7x7_s2)
pool1_3x3_s2 = MaxPoolingLayer(net, conv1_7x7_s2, kernel=3, stride=2, pad=0)

#pool1_norm1 = LRNLayer(net, pool1_3x3_s2, n=5, beta=0.75, alpha=0.0001, k=1.0)
#conv2_3x3_reduce = ConvLayer(net, pool1_norm1, num_filters=64, kernel=1, stride=1, pad=0)
conv2_3x3_reduce = ConvLayer(net, pool1_3x3_s2, num_filters=64, kernel=1, stride=1, pad=0)

conv2_relu_3x3_reduce = ReLULayer(net, conv2_3x3_reduce)
conv2_3x3 = ConvLayer(net, conv2_relu_3x3_reduce, num_filters=192, kernel=3, stride=1, pad=1)
conv2_relu_3x3 = ReLULayer(net, conv2_3x3)

#conv2_norm2 = LRNLayer(net, conv2_relu_3x3, n=5, beta=0.75, alpha=0.0001, k=1.0)
#pool2_3x3_s2 = MaxPoolingLayer(net, conv2_norm2, kernel=3, stride=2, pad=0)
pool2_3x3_s2 = MaxPoolingLayer(net, conv2_relu_3x3, kernel=3, stride=2, pad=0)

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
loss1_classifier =  FullyConnectedLayer(net, loss1_relu_fc, 1008)#ANAND: Changed from 1000 to 1024 for testing
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
'''
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
loss2_classifier =  FullyConnectedLayer(net, loss2_relu_fc, 1008)#ANAND: Changed from 1000 to 1024 for testing
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
loss3_classifier =  FullyConnectedLayer(net, pool5_7x7_s1, 1008)#ANAND: Changed from 1000 to 1024 for testing
#loss3_loss3 = SoftmaxLossLayer(net, loss3_classifier, label)
'''
'''
layers = [("conv1/7x7_s2",conv1_7x7_s2),("pool1/3x3_s2",pool1_3x3_s2),("pool1/norm1",pool1_norm1),
("conv2/3x3_reduce",conv2_3x3_reduce),("conv2/3x3",conv2_3x3), ("conv2/norm2",conv2_norm2),
("pool2/3x3_s2",pool2_3x3_s2),("inception_3a/1x1",inception_3a_1x1),("inception_3a/3x3_reduce",inception_3a_3x3_reduce),
("inception_3a/3x3",inception_3a_3x3),("inception_3a/5x5_reduce",inception_3a_5x5_reduce),("inception_3a/5x5",inception_3a_5x5),
("inception_3a/pool",inception_3a_pool),("inception_3a/pool_proj",inception_3a_pool_proj),("inception_3a/output",inception_3a_output),
("inception_3b/1x1",inception_3b_1x1),("inception_3b/3x3_reduce",inception_3b_3x3_reduce),("inception_3b/3x3",inception_3b_3x3),
("inception_3b/5x5_reduce",inception_3b_5x5_reduce),("inception_3b/5x5",inception_3b_5x5),("inception_3b/pool",inception_3b_pool),
("inception_3b/pool_proj",inception_3b_pool_proj),("inception_3b/output",inception_3b_output),("pool3/3x3_s2",pool3_3x3_s2),
("inception_4a/1x1",inception_4a_1x1),("inception_4a/3x3_reduce",inception_4a_3x3_reduce),("inception_4a/3x3",inception_4a_3x3),
("inception_4a/5x5_reduce",inception_4a_5x5_reduce),("inception_4a/5x5",inception_4a_5x5),("inception_4a/pool",inception_4a_pool),
("inception_4a/pool_proj",inception_4a_pool_proj),("inception_4a/output",inception_4a_output),("loss1/ave_pool",loss1_ave_pool),
("loss1/conv",loss1_conv),("loss1/fc",loss1_fc),("loss1/classifier",loss1_classifier),("inception_4b/1x1",inception_4b_1x1),
("inception_4b/3x3_reduce",inception_4b_3x3_reduce),("inception_4b/3x3",inception_4b_3x3),("inception_4b/5x5_reduce",inception_4b_5x5_reduce),
("inception_4b/5x5",inception_4b_5x5),("inception_4b/pool",inception_4b_pool),("inception_4b/pool_proj",inception_4b_pool_proj),
("inception_4b/output",inception_4b_output),("inception_4c/1x1",inception_4c_1x1),("inception_4c/3x3_reduce",inception_4c_3x3_reduce),
("inception_4c/3x3",inception_4c_3x3),("inception_4c/5x5_reduce",inception_4c_5x5_reduce),("inception_4c/5x5",inception_4c_5x5),
("inception_4c/pool",inception_4c_pool),("inception_4c/pool_proj",inception_4c_pool_proj),("inception_4c/output",inception_4c_output),
("inception_4d/1x1",inception_4d_1x1),("inception_4d/3x3_reduce",inception_4d_3x3_reduce),("inception_4d/3x3",inception_4d_3x3),
("inception_4d/5x5_reduce",inception_4d_5x5_reduce),("inception_4d/5x5",inception_4d_5x5),("inception_4d/pool",inception_4d_pool),
("inception_4d/pool_proj",inception_4d_pool_proj),("inception_4d/output",inception_4d_output),("loss2/ave_pool",loss2_ave_pool),
("loss2/conv",loss2_conv),("loss2/fc",loss2_fc),("loss2/classifier",loss2_classifier),("inception_4e/1x1",inception_4e_1x1),
("inception_4e/3x3_reduce",inception_4e_3x3_reduce),("inception_4e/3x3",inception_4e_3x3),("inception_4e/5x5_reduce",inception_4e_5x5_reduce),
("inception_4e/5x5",inception_4e_5x5),("inception_4e/pool",inception_4e_pool),("inception_4e/pool_proj",inception_4e_pool_proj),
("inception_4e/output",inception_4e_output),("pool4/3x3_s2",pool4_3x3_s2),("inception_5a/1x1",inception_5a_1x1),
("inception_5a/3x3_reduce",inception_5a_3x3_reduce),("inception_5a/3x3",inception_5a_3x3),("inception_5a/5x5_reduce",inception_5a_5x5_reduce),
("inception_5a/5x5",inception_5a_5x5),("inception_5a/pool",inception_5a_pool),("inception_5a/pool_proj",inception_5a_pool_proj),
("inception_5a/output",inception_5a_output),("inception_5b/1x1",inception_5b_1x1),("inception_5b/3x3_reduce",inception_5b_3x3_reduce),
("inception_5b/3x3",inception_5b_3x3),("inception_5b/5x5_reduce",inception_5b_5x5_reduce),("inception_5b/5x5",inception_5b_5x5),
("inception_5b/pool",inception_5b_pool),("inception_5b/pool_proj",inception_5b_pool_proj),("inception_5b/output",inception_5b_output),
("pool5/7x7_s1",pool5_7x7_s1),("loss3/classifier",loss3_classifier)]


convs=[("conv1/7x7_s2",conv1_7x7_s2),("conv2/3x3_reduce",conv2_3x3_reduce),("conv2/3x3",conv2_3x3),
("inception_3a/1x1",inception_3a_1x1),("inception_3a/3x3_reduce",inception_3a_3x3_reduce),
("inception_3a/3x3",inception_3a_3x3),("inception_3a/5x5_reduce",inception_3a_5x5_reduce),
("inception_3a/5x5",inception_3a_5x5),("inception_3a/pool_proj",inception_3a_pool_proj),
("inception_3b/1x1",inception_3b_1x1),("inception_3b/3x3_reduce",inception_3b_3x3_reduce),
("inception_3b/3x3",inception_3b_3x3),("inception_3b/5x5_reduce",inception_3b_5x5_reduce),
("inception_3b/5x5",inception_3b_5x5),("inception_3b/pool_proj",inception_3b_pool_proj),
("inception_4a/1x1",inception_4a_1x1),("inception_4a/3x3_reduce",inception_4a_3x3_reduce),
("inception_4a/3x3",inception_4a_3x3),("inception_4a/5x5_reduce",inception_4a_5x5_reduce),
("inception_4a/5x5",inception_4a_5x5),("inception_4a/pool_proj",inception_4a_pool_proj),
("loss1/conv",loss1_conv),("loss1/fc",loss1_fc),("loss1/classifier",loss1_classifier),("inception_4b/1x1",inception_4b_1x1),
("inception_4b/3x3_reduce",inception_4b_3x3_reduce),("inception_4b/3x3",inception_4b_3x3),("inception_4b/5x5_reduce",inception_4b_5x5_reduce),
("inception_4b/5x5",inception_4b_5x5),("inception_4b/pool_proj",inception_4b_pool_proj),("inception_4c/1x1",inception_4c_1x1),
("inception_4c/3x3_reduce",inception_4c_3x3_reduce),("inception_4c/3x3",inception_4c_3x3),("inception_4c/5x5_reduce",inception_4c_5x5_reduce),
("inception_4c/5x5",inception_4c_5x5),("inception_4c/pool_proj",inception_4c_pool_proj),("inception_4d/1x1",inception_4d_1x1),
("inception_4d/3x3_reduce",inception_4d_3x3_reduce),("inception_4d/3x3",inception_4d_3x3),("inception_4d/5x5_reduce",inception_4d_5x5_reduce),
("inception_4d/5x5",inception_4d_5x5),("inception_4d/pool_proj",inception_4d_pool_proj),("loss2/conv",loss2_conv),("loss2/fc",loss2_fc),
("loss2/classifier",loss2_classifier),("inception_4e/1x1",inception_4e_1x1),("inception_4e/3x3_reduce",inception_4e_3x3_reduce),
("inception_4e/3x3",inception_4e_3x3),("inception_4e/5x5_reduce",inception_4e_5x5_reduce),("inception_4e/5x5",inception_4e_5x5),
("inception_4e/pool_proj",inception_4e_pool_proj),("inception_5a/1x1",inception_5a_1x1),("inception_5a/3x3_reduce",inception_5a_3x3_reduce),
("inception_5a/3x3",inception_5a_3x3),("inception_5a/5x5_reduce",inception_5a_5x5_reduce),("inception_5a/5x5",inception_5a_5x5),
("inception_5a/pool_proj",inception_5a_pool_proj),("inception_5b/1x1",inception_5b_1x1),("inception_5b/3x3_reduce",inception_5b_3x3_reduce),
("inception_5b/3x3",inception_5b_3x3),("inception_5b/5x5_reduce",inception_5b_5x5_reduce),("inception_5b/5x5",inception_5b_5x5),
("inception_5b/pool_proj",inception_5b_pool_proj),("loss3/classifier",loss3_classifier)]
'''


net.compile() 

num_train = 1
 
 
#train_batches = [i for i in range(0, num_train, batch_size)]
total_forward_time = 0.0
total_backward_time = 0.0
epoch_size = 2
timing_info = True
 
#train_images = np.random.randint(0, 255, (num_train, 3, 306, 306)).astype(np.float32)
#train_labels = np.random.randint(0, 255, (num_train, 1, 306, 306)).astype(np.float32)

label = np.zeros((batch_size,1), dtype=int)

 
print("Training ...")
#forward_time = 0.0
#backward_time = 0.0
'''
labels_file = caffe_root + 'data/ilsvrc12/synset_words.txt'
if not os.path.exists(labels_file):
    os.system("...data/ilsvrc12/get_ilsvrc_aux.sh")
labels = np.loadtxt(labels_file, str, delimiter='\t')
solver = caffe.SGDSolver('solver.prototxt')
net2 = solver.net
'''

for epoch in range(epoch_size):
 
 
    for i in range(1):
        forward_time = 0.0
        backward_time = 0.0
 
        data.set_value(np.random.rand(batch_size, 3, 224, 224))
 
        #train_data = train_images[n:n+batch_size]
        #train_label = train_labels[n:n+batch_size]
        #data.set_value(train_data)
        #label.set_value(train_label)
        #data.set_value(net2.blobs["data"].data)

        #net2.forward()

        #print(net2.blobs["loss1/classifier"].diff)

        #net2.backward()
            
        #if i == 0:
             #copy_weights(net2,convs)
             #label = net2.blobs["label"].data
             #label = label.reshape(batch_size,1)
             #data.set_value(net2.blobs["data"].data)

        #print(label[0][0])

        t = time.time()
        net.forward()
        forward_time += time.time() - t

        #forward_time += time.time() - t
 
        # Compute loss
        '''
        output1 = loss1_classifier.get_value()
        output2 = loss2_classifier.get_value()
        output3 = loss3_classifier.get_value()
        

        prob1 = np.zeros_like(output1)
        prob2 = np.zeros_like(output2)
        prob3 = np.zeros_like(output3)

        output_grad1 = np.zeros_like(output1)
        output_grad2 = np.zeros_like(output2)
        output_grad3 = np.zeros_like(output3)
        
        loss1 = compute_softmax_loss(output1, prob1, label)
        loss2 = compute_softmax_loss(output2, prob2, label)
        loss3 = compute_softmax_loss(output3, prob3, label)
        #forward_time += time.time() - t 
    
        #acc = compute_seg_accuracy(output, shrink_label.get_value(), ignore_label)
 
        # Initialize gradients
        

       # t = time.time()


        softmax_loss_backprop(output_grad1, prob1, label)
        softmax_loss_backprop(output_grad2, prob2, label)
        softmax_loss_backprop(output_grad3, prob3, label)

        output_grad1 *= (0.3)
        output_grad2 *= (0.3)


        #print(output_grad1)
        #print(net2.blobs["loss1/classifier"].diff) 

        #assert np.allclose(net2.blobs["loss1/classifier"].diff, output_grad1, 1e-7, 1e-1)
        #assert np.allclose(net2.blobs["loss2/loss"].data, loss2, 1e-7, 1e-1)
        #assert np.allclose(net2.blobs["loss3/loss"].data, loss3, 1e-7, 1e-1)



        loss1_classifier.set_grad(output_grad1)
        loss2_classifier.set_grad(output_grad2)
        loss3_classifier.set_grad(output_grad3)
        '''
        t = time.time()
        
        net.backward()
        
        backward_time += time.time() - t
        #lr = base_lr * (1 + gamma * i)**power
        #mom = .9
        #for param in params:
        #    sgd_update(param[0], param[1], param[2], lr, mom, batch_size)
        
        #weight_update_and_check_output(convs, layers, batch_size)

      
        #net.clear_values()
        #net.clear_grad()
        #net.loss = 0.0
       
        '''
        for layer in layers:
          net2.blobs[layer[0]].data[...] *= 0
          net2.blobs[layer[0]].diff[...] *= 0
        for layer in net2.params:
          net2.params[layer][0].diff[...] *= 0
          net2.params[layer][1].diff[...] *= 0
        '''  
        if timing_info:
            print("Iteration {} -- ".format(epoch))
            print("FP                   : {0:.3f} ms".format(forward_time * 1000))
            print("BP+WU                : {0:.3f} ms".format(backward_time * 1000))
            print("Total Time           : {0:.3f} ms".format((forward_time+backward_time)*1000))
            print("")
 
        total_forward_time += forward_time
        total_backward_time += backward_time
 
print("Total FP                   : {0:.3f} ms".format(total_forward_time * 1000))
print("Total BP+WU                : {0:.3f} ms".format(total_backward_time * 1000))
print("Total Time                 : {0:.3f} ms".format((total_forward_time+total_backward_time)*1000))
print("Total Inference Throughput : {0:.3f} images/second".format((num_train * epoch_size*batch_size)/(total_forward_time)))
print("Total Training Throughput  : {0:.3f} images/second".format((num_train * epoch_size*batch_size)/(total_forward_time + total_backward_time)))

