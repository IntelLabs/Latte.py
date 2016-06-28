import numpy as np
from latte import *
from latte.solvers import sgd_update
import random
from data_loader import load_data, load_images
from latte.math import compute_seg_softmax_loss, seg_softmax_loss_backprop, compute_seg_accuracy


batch_size = 1
net = Net(batch_size)

data     = MemoryDataLayer(net, (3, 306, 306))
label     = MemoryDataLayer(net, (3, 306, 306))
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
conv4_3 = ConvLayer(net, relu4_2, num_filters=512, kernel=3, stride=1, pad=1)
relu4_3 = ReLULayer(net, conv4_3)
pool4 = MaxPoolingLayer(net, relu4_3, kernel=2, stride=1, pad=0)

conv5_1 = ConvLayer(net, pool4, num_filters=512, kernel=3, stride=1, pad=2)
relu5_1 = ReLULayer(net, conv5_1)
conv5_2 = ConvLayer(net, relu5_1, num_filters=512, kernel=3, stride=1, pad=2)
relu5_2 = ReLULayer(net, conv5_2)
conv5_3 = ConvLayer(net, relu5_2, num_filters=512, kernel=3, stride=1, pad=2)
relu5_3 = ReLULayer(net, conv5_3)
pool5 = MaxPoolingLayer(net, relu5_3, kernel=3, stride=1, pad=1)

fc6 = ConvLayer(net, pool5, num_filters=4096, kernel=4, stride=1, pad=6)
relu6 = ReLULayer(net, fc6)
drop6 = DropoutLayer(net, relu6, ratio=0.5)
fc7 = ConvLayer(net, drop6, num_filters=4096, kernel=1, stride=1, pad=0)
relu7 = ReLULayer(net, fc7)
drop7 = DropoutLayer(net, relu7, ratio=0.5)
fc8_pascal = ConvLayer(net, drop7, num_filters=24, kernel=1, stride=1, pad=0)
shrink_label = InterpolationLayer(net, label, pad=1, resize_factor=8)
print("Compiling...")


conv1_1.name = 'conv1_1'
relu1_1.name = 'relu1_1'
conv1_2.name = 'conv1_2'
relu1_2.name = 'relu1_2'
pool1.name = 'pool1'

conv2_1.name = 'conv2_1'
relu2_1.name = 'relu2_1'
conv2_2.name = 'conv2_2'
relu2_2.name = 'relu2_2'
pool2.name = 'pool2'

conv3_1.name = 'conv3_1'
relu3_1.name = 'relu3_1'
conv3_2.name = 'conv3_2'
relu3_2.name = 'relu3_2'
conv3_3.name = 'conv3_3'
relu3_3.name = 'relu3_3'
pool3.name = 'pool3'

conv4_1.name = 'conv4_1'
relu4_1.name = 'relu4_1'
conv4_2.name = 'conv4_2'
relu4_2.name = 'relu4_2'
conv4_3.name = 'conv4_3'
relu4_3.name = 'relu4_3'
pool4.name = 'pool4'

conv5_1.name = 'conv5_1'
relu5_1.name = 'relu5_1'
conv5_2.name = 'conv5_2'
relu5_2.name = 'relu5_2'
conv5_3.name = 'conv5_3'
relu5_3.name = 'relu5_3'
pool5.name = 'pool5'

fc6.name = 'fc6'
relu6.name = 'relu6'
drop6.name = 'drop6'
fc7.name = 'fc7'
drop7.name = 'drop7'
fc8_pascal.name = 'fc8'


net.compile()

sizes = dict()
bytes = 0
for key, value in net.buffers.items():
    bytes += value.nbytes
    sizes[key] = value.nbytes

#for key in sorted(sizes, key=sizes.get, reverse=True):
#    print(key + ": " + str(sizes[key]))

print("TOTAL MEM: " + str(bytes))

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

base_lr = .01
gamma = .0001
power = .75

output = fc8_pascal.get_value()
prob = np.zeros_like(output)

output_grad = np.zeros_like(output)

training_images_list = load_data(dataset="training", path="CityScapes/list/")
test_images_list = load_data(dataset="testing", path="CityScapes/list/")
#training_images_list = [np.random.rand(8, 306, 306) for _ in range(100)]
#test_images_list = [np.random.rand(8, 306, 306) for _ in range(100)]

num_train = len(training_images_list)
num_test = len(test_images_list)

train_batches = [i for i in range(0, num_train, batch_size)]

for epoch in range(10):
    random.shuffle(train_batches)
    print("Epoch {} - Training...".format(epoch))
    for i, n in enumerate(train_batches):
        train_data, train_label = load_images(training_images_list, data_folder="./data/", is_color=True, crop_size=306, start=n, batch_size=batch_size)
        #train_data = np.array(training_images_list[n:n+batch_size])
        #train_label = np.random.rand(batch_size, 8, 306, 306) * 100
        data.set_value(train_data)
        label.set_value(train_label)
        net.forward()

        # Compute loss
        output = fc8_pascal.get_value()
        loss = compute_seg_softmax_loss(output, prob, shrink_label.get_value(), 255)

        #if i % 100 == 0:
        print("Epoch {}, Train Iteration {} - Loss = {}".format(epoch, i, loss))

        # Initialize gradients
        seg_softmax_loss_backprop(output_grad, prob, shrink_label.get_value(), 255)
        fc8_pascal.set_grad(output_grad)

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
        test_data, test_label  = load_images(test_images_list, data_folder="./data/", is_color=True, crop_size=306, start=n, batch_size=batch_size)
        #test_data = np.array(test_images_list[n:n+batch_size])
        #test_label = np.random.rand(batch_size, 8, 306, 306) * 100
        data.set_value(test_data)
        label.set_value(test_label)
        net.test()

        acc += compute_seg_accuracy(fc8_pascal.get_value(), shrink_label.get_value(), ignore_label)
        net.clear_values()
    acc /= (num_test / batch_size)
    acc *= 100
    print("Epoch {} - Validation accuracy = {:.3f}%".format(epoch, acc))
