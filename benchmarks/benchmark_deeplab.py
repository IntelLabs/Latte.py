import numpy as np
from latte import *
from latte.solvers import sgd_update
import random
import time
from latte.math import compute_seg_softmax_loss, seg_softmax_loss_backprop, compute_seg_accuracy

batch_size = 20
net = Net(batch_size)

data     = MemoryDataLayer(net, (3, 306, 306))
label    = MemoryDataLayer(net, (1, 306, 306))

conv1_1 = ConvLayer(      net, data, num_filters=64, kernel=3, stride=1, pad=1)
relu1_1 = ReLULayer(      net, conv1_1)
conv1_2 = ConvLayer(      net, relu1_1, num_filters=64, kernel=3, stride=1, pad=1)
relu1_2 = ReLULayer(      net, conv1_2)
pool1   = MaxPoolingLayer(net, relu1_2, kernel=2, stride=2, pad=1)

conv2_1 = ConvLayer(net,     pool1, num_filters=128, kernel=3, stride=1, pad=1)
relu2_1 = ReLULayer(net,     conv2_1)
conv2_2 = ConvLayer(net,     relu2_1, num_filters=128, kernel=3, stride=1, pad=1)
relu2_2 = ReLULayer(net,     conv2_2)
pool2 = MaxPoolingLayer(net, relu2_2, kernel=2, stride=2, pad=1)

conv3_1 = ConvLayer(net, pool2, num_filters=256, kernel=3, stride=1, pad=1)
relu3_1 = ReLULayer(net, conv3_1)
conv3_2 = ConvLayer(net, relu3_1, num_filters=256, kernel=3, stride=1, pad=1)
relu3_2 = ReLULayer(net, conv3_2)
conv3_3 = ConvLayer(net, relu3_2, num_filters=256, kernel=3, stride=1, pad=1)
relu3_3 = ReLULayer(net, conv3_3)
pool3 = MaxPoolingLayer(net, relu3_3, kernel=2, stride=2, pad=1)

conv4_1 = ConvLayer(net, pool3, num_filters=512, kernel=3, stride=1, pad=1)
relu4_1 = ReLULayer(net, conv4_1)
conv4_2 = ConvLayer(net, relu4_1, num_filters=512, kernel=3, stride=1, pad=1)
relu4_2 = ReLULayer(net, conv4_2)
conv4_3 = ConvLayer(net, relu4_2, num_filters=512, kernel=3, stride=1)
relu4_3 = ReLULayer(net, conv4_3)
pool4 = MaxPoolingLayer(net, relu4_3, kernel=2, stride=1, pad=0)

conv5_1 = ConvLayer(net, pool4, num_filters=512, kernel=3, stride=1, pad=2, dilation=2)
relu5_1 = ReLULayer(net, conv5_1)
conv5_2 = ConvLayer(net, relu5_1, num_filters=512, kernel=3, stride=1, pad=2, dilation=2)
relu5_2 = ReLULayer(net, conv5_2)
conv5_3 = ConvLayer(net, relu5_2, num_filters=512, kernel=3, stride=1, pad=2, dilation=2)
relu5_3 = ReLULayer(net, conv5_3)
pool5 = MaxPoolingLayer(net, relu5_3, kernel=3, stride=1, pad=1)

fc6 = ConvLayer(net, pool5, num_filters=4096, kernel=4, stride=1, pad=6, dilation=4)
relu6 = ReLULayer(net, fc6)
drop6 = DropoutLayer(net, relu6, ratio=0.5)
fc7 = ConvLayer(net, drop6, num_filters=4096, kernel=1, stride=1, pad=0)
relu7 = ReLULayer(net, fc7)
drop7 = DropoutLayer(net, relu7, ratio=0.5)

fc8_pascal = ConvLayer(net, drop7, num_filters=19, kernel=1, stride=1, pad=0)

shrink_label = InterpolationLayer(net, label, pad=-1, resize_factor=0.125)


print("Compiling...")
net.compile()

def make_param(buffer, grad):
    return buffer, grad, np.zeros_like(buffer)

params = [
    make_param(conv1_1.get_weights_view() , conv1_1.get_grad_weights_view() ), 
    make_param(conv1_1.get_bias_view()    , conv1_1.get_grad_bias_view()    ),
    make_param(conv1_2.get_weights_view() , conv1_2.get_grad_weights_view() ),
    make_param(conv1_2.get_bias_view()    , conv1_2.get_grad_bias_view()    ),
    make_param(conv2_1.get_weights_view() , conv2_1.get_grad_weights_view() ),
    make_param(conv2_1.get_bias_view()    , conv2_1.get_grad_bias_view()    ),
    make_param(conv2_2.get_weights_view() , conv2_2.get_grad_weights_view() ),
    make_param(conv2_2.get_bias_view()    , conv2_2.get_grad_bias_view()    ),
    make_param(conv3_1.get_weights_view() , conv3_1.get_grad_weights_view() ),
    make_param(conv3_1.get_bias_view()    , conv3_1.get_grad_bias_view()    ),
    make_param(conv3_2.get_weights_view() , conv3_2.get_grad_weights_view() ),
    make_param(conv3_2.get_bias_view()    , conv3_2.get_grad_bias_view()    ),
    make_param(conv3_3.get_weights_view() , conv3_3.get_grad_weights_view() ),
    make_param(conv3_3.get_bias_view()    , conv3_3.get_grad_bias_view()    ),
    make_param(conv4_1.get_weights_view() , conv4_1.get_grad_weights_view() ),
    make_param(conv4_1.get_bias_view()    , conv4_1.get_grad_bias_view()    ),
    make_param(conv4_2.get_weights_view() , conv4_2.get_grad_weights_view() ),
    make_param(conv4_2.get_bias_view()    , conv4_2.get_grad_bias_view()    ),
    make_param(conv4_3.get_weights_view() , conv4_3.get_grad_weights_view() ),
    make_param(conv4_3.get_bias_view()    , conv4_3.get_grad_bias_view()    ),
    make_param(conv5_1.get_weights_view() , conv5_1.get_grad_weights_view() ),
    make_param(conv5_1.get_bias_view()    , conv5_1.get_grad_bias_view()    ),
    make_param(conv5_2.get_weights_view() , conv5_2.get_grad_weights_view() ),
    make_param(conv5_2.get_bias_view()    , conv5_2.get_grad_bias_view()    ),
    make_param(conv5_3.get_weights_view() , conv5_3.get_grad_weights_view() ),
    make_param(conv5_3.get_bias_view()    , conv5_3.get_grad_bias_view()    ),
    make_param(fc6.get_weights_view()     , fc6.get_grad_weights_view()   ),
    make_param(fc6.get_bias_view()        , fc6.get_grad_bias_view()      ),
    make_param(fc7.get_weights_view()     , fc7.get_grad_weights_view()   ),
    make_param(fc7.get_bias_view()        , fc7.get_grad_bias_view()      ),
    make_param(fc8_pascal.get_weights_view()   , fc8_pascal.get_grad_weights_view()   ),
    make_param(fc8_pascal.get_bias_view()      , fc8_pascal.get_grad_bias_view()      ),

]

base_lr = .001
gamma = 0.1
power = .75

output = fc8_pascal.get_value()
prob = np.zeros_like(output)

output_grad = np.zeros_like(output)

num_train = 60

train_batches = [i for i in range(0, num_train, batch_size)]

total_forward_time = 0.0
total_backward_time = 0.0
epoch_size = 1
timing_info = False

ignore_label = 255

train_images = np.random.randint(0, 255, (num_train, 3, 306, 306)).astype(np.float32)
train_labels = np.random.randint(0, 255, (num_train, 1, 306, 306)).astype(np.float32)

print("Training ...")
for epoch in range(epoch_size):


    for i, n in enumerate(train_batches):

        forward_time = 0.0
        backward_time = 0.0
        
        train_data = train_images[n:n+batch_size]
        train_label = train_labels[n:n+batch_size]
        data.set_value(train_data)
        label.set_value(train_label)
 
        t = time.time()
        net.forward()
        forward_time += time.time() - t

        # Compute loss
        output = fc8_pascal.get_value()
        loss = compute_seg_softmax_loss(output, prob, shrink_label.get_value(), ignore_label)
        acc = compute_seg_accuracy(output, shrink_label.get_value(), ignore_label)

        # Initialize gradients
        seg_softmax_loss_backprop(output_grad, prob, shrink_label.get_value(), ignore_label)
        fc8_pascal.set_grad(output_grad)

        t = time.time()
        net.backward()
        backward_time += time.time() - t

        lr = base_lr * (1 + gamma * i)**power
        mom = .9
        for param in params:
            sgd_update(param[0], param[1], param[2], lr, mom, batch_size)
        net.clear_values()
        net.clear_grad()
        net.loss = 0.0
       
        if timing_info:
            print("Iteration {} -- ".format(i))
            print("FP                   : {0:.3f} ms".format(forward_time * 1000))
            print("BP+WU                : {0:.3f} ms".format(backward_time * 1000))
            print("Total Time           : {0:.3f} ms".format((forward_time+backward_time)*1000))
            print("")

        total_forward_time += forward_time
        total_backward_time += backward_time
    
print("Total FP                   : {0:.3f} ms".format(total_forward_time * 1000))
print("Total BP+WU                : {0:.3f} ms".format(total_backward_time * 1000))
print("Total Time                 : {0:.3f} ms".format((total_forward_time+total_backward_time)*1000))
print("Total Training Throughput  : {0:.3f} images/second".format((num_train * epoch_size)/(total_forward_time + total_backward_time)))
