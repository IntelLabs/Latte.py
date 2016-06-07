import numpy as np
from latte import *
from latte.solvers import sgd_update
import random
from data_loader import load_deeplab
from latte.math import compute_softmax_loss, softmax_loss_backprop, compute_seg_accuracy

print("Loading data...")
train_data, train_label = load_deeplab(dataset="training", path="CityScapes/list/", data_folder="./data/", is_color=True, crop_size=306)
test_data, test_label   = load_deeplab(dataset="testing", path="CityScapes/list/", data_folder="./data/", is_color=True, crop_size=306)

num_train = len(train_data)
num_test = len(test_data)

batch_size = 20
net = Net(batch_size)

data     = MemoryDataLayer(net, (8, 306, 306))
conv1_1 = ConvLayer(net, data, num_filters=64, kernel=3, stride=1, pad=1)
relu1_1 = ReLULayer(net, conv1_1)
conv1_2 = ConvLayer(net, conv1_1, num_filters=64, kernel=3, stride=1, pad=1)
relu1_2 = ReLULayer(net, conv1_2)
pool1 = MaxPoolingLayer(net, conv1_2, kernel=2, stride=2, pad=1)
conv2_1 = ConvLayer(net, pool1, num_filters=128, kernel=3, stride=1, pad=1)
relu2_1 = ReLULayer(net, conv2_1)
conv2_2 = ConvLayer(net, conv2_1, num_filters=128, kernel=3, stride=1, pad=1)
relu2_2 = ReLULayer(net, conv2_2)
pool2 = MaxPoolingLayer(net, conv2_2, kernel=2, stride=2, pad=1)
conv3_1 = ConvLayer(net, pool2, num_filters=256, kernel=3, stride=1, pad=1)
relu3_1 = ReLULayer(net, conv3_1)
conv3_2 = ConvLayer(net, conv3_1, num_filters=256, kernel=3, stride=1, pad=1)
relu3_2 = ReLULayer(net, conv3_2)
conv3_3 = ConvLayer(net, conv3_2, num_filters=256, kernel=3, stride=1, pad=1)
relu3_3 = ReLULayer(net, conv3_3)
pool3 = MaxPoolingLayer(net, conv3_3, kernel=2, stride=2, pad=1)
conv4_1 = ConvLayer(net, pool3, num_filters=512, kernel=3, stride=1, pad=1)
relu4_1 = ReLULayer(net, conv4_1)
conv4_2 = ConvLayer(net, conv4_1, num_filters=512, kernel=3, stride=1, pad=1)
relu4_2 = ReLULayer(net, conv4_2)
conv4_3 = ConvLayer(net, conv4_2, num_filters=512, kernel=3, stride=1, pad=1)
relu4_3 = ReLULayer(net, conv4_3)
pool4 = MaxPoolingLayer(net, conv4_3, kernel=2, stride=1, pad=0)
conv5_1 = ConvLayer(net, pool4, num_filters=512, kernel=3, stride=1, pad=2)
relu5_1 = ReLULayer(net, conv5_1)
conv5_2 = ConvLayer(net, conv5_1, num_filters=512, kernel=3, stride=1, pad=2)
relu5_2 = ReLULayer(net, conv5_2)
conv5_3 = ConvLayer(net, conv5_2, num_filters=512, kernel=3, stride=1, pad=2)
relu5_3 = ReLULayer(net, conv5_3)
pool5 = MaxPoolingLayer(net, conv5_3, kernel=3, stride=1, pad=1)
fc6 = ConvLayer(net, pool5, num_filters=4096, kernel=4, stride=1, pad=6)
relu6 = ReLULayer(net, fc6)
drop6 = DropoutLayer(net, fc6, ratio=0.5)
fc7 = ConvLayer(net, fc6, num_filters=4096, kernel=1, stride=1, pad=0)
relu7 = ReLULayer(net, fc7)
drop7 = DropoutLayer(net, fc7, ratio=0.5)
fc8_pascal = ConvLayer(net, fc7, num_filters=19, kernel=1, stride=1, pad=0)
#label_shrink = InterpolationLayer(net, np.asarray(train_label), resize_factor=8)

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
    make_param(fc6.get_weights_view()   , fc6.get_grad_weights_view()   ),
    make_param(fc6.get_bias_view()      , fc6.get_grad_bias_view()      ),
    make_param(fc7.get_weights_view()   , fc7.get_grad_weights_view()   ), 
    make_param(fc7.get_bias_view()      , fc7.get_grad_bias_view()      ),
    make_param(fc8_pascal.get_weights_view()   , fc8_pascal.get_grad_weights_view()   ), 
    make_param(fc8_pascal.get_bias_view()      , fc8_pascal.get_grad_bias_view()      ),
]

base_lr = .01
gamma = .0001
power = .75

train_batches = [i for i in range(0, num_train, batch_size)]

output = fc8_pascal.get_value()
prob = np.zeros_like(output)

output_grad = np.zeros_like(output)

for epoch in range(10):
    random.shuffle(train_batches)
    print("Epoch {} - Training...".format(epoch))
    for i, n in enumerate(train_batches):
        data.set_value(train_data[n:n+batch_size])
        label_value = train_label[n:n+batch_size]
        net.forward()

        # Compute loss
        output = fc8_pascal.get_value()

        loss = compute_softmax_loss(output, prob, label_value)

        if i % 100 == 0:
            print("Epoch {}, Train Iteration {} - Loss = {}".format(epoch, i, loss))

        # Initialize gradients
        softmax_loss_backprop(output_grad, prob, label_value)
        fc4.set_grad(output_grad)

        net.backward()
        lr = base_lr * (1 + gamma * i)**power
        mom = .9
        for param in params:
            # expected = param[0] - (param[2] * mom + np.sum(param[1], axis=0) * lr)
            sgd_update(param[0], param[1], param[2], lr, mom, batch_size)
        net.clear_values()
        net.clear_grad()
        net.loss = 0.0

    print("Epoch {} - Testing...".format(epoch))
    acc = 0
    for i, n in enumerate(range(0, num_test, batch_size)):
        data.set_value(test_data[n:n+batch_size])
        label_value = test_label[n:n+batch_size]
        net.test()

        acc += compute_seg_accuracy(fc8_pascal.get_value(), label_value)
        net.clear_values()
    acc /= (num_test / batch_size)
    acc *= 100
    print("Epoch {} - Validation accuracy = {:.3f}%".format(epoch, acc))
