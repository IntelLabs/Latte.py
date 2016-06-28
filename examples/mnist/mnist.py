import numpy as np
from latte import *
from latte.solvers import sgd_update
import random
from data_loader import load_mnist
from latte.math import compute_softmax_loss, softmax_loss_backprop, compute_accuracy

train_data, train_label = load_mnist(dataset="training", path="./data")
test_data, test_label   = load_mnist(dataset="testing", path="./data")

num_train = train_data.shape[0]
train_data = np.pad(train_data.reshape(num_train, 1, 28, 28), [(0, 0), (0, 7), (0, 0), (0, 0)], mode='constant')
train_label = train_label.reshape(num_train, 1)

num_test = test_data.shape[0]
test_data = np.pad(test_data.reshape(num_test, 1, 28, 28), [(0, 0), (0, 7), (0, 0), (0, 0)], mode='constant')
test_label = test_label.reshape(num_test, 1)


batch_size = 50
net = Net(batch_size)

data  = MemoryDataLayer(net, train_data[0].shape)

conv1 = ConvLayer(net, data, num_filters=16, kernel=5, stride=1, pad=0)
relu1 = ReLULayer(net, conv1)
pool1 = MaxPoolingLayer(net, relu1, kernel=2, stride=2, pad=0)

conv2 = ConvLayer(net, pool1, num_filters=64, kernel=5, stride=1, pad=0)
relu2 = ReLULayer(net, conv2)
pool2 = MaxPoolingLayer(net, relu2, kernel=2, stride=2, pad=0)

fc3   = FullyConnectedLayer(net, pool2, 128)
relu3 = ReLULayer(net, fc3)
drop3 = DropoutLayer(net, relu3, .5)
fc4   = FullyConnectedLayer(net, drop3, 16)

net.compile()

def make_param(buffer, grad):
    return buffer, grad, np.zeros_like(buffer)

params = [
    make_param(conv1.get_weights_view() , conv1.get_grad_weights_view() ), 
    make_param(conv1.get_bias_view()    , conv1.get_grad_bias_view()    ),
    make_param(conv2.get_weights_view() , conv2.get_grad_weights_view() ),
    make_param(conv2.get_bias_view()    , conv2.get_grad_bias_view()    ),
    make_param(fc3.get_weights_view()   , fc3.get_grad_weights_view()   ),
    make_param(fc3.get_bias_view()      , fc3.get_grad_bias_view()      ),
    make_param(fc4.get_weights_view()   , fc4.get_grad_weights_view()   ), 
    make_param(fc4.get_bias_view()      , fc4.get_grad_bias_view()      ),
]

base_lr = .01
gamma = .0001
power = .75

train_batches = [i for i in range(0, num_train, batch_size)]

output = fc4.get_value()
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
        output = fc4.get_value()

        loss = compute_softmax_loss(output, prob, label_value)

        if i % 10 == 0:
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

        acc += compute_accuracy(fc4.get_value(), label_value)
        net.clear_values()
    acc /= (num_test / batch_size)
    acc *= 100
    print("Epoch {} - Validation accuracy = {:.3f}%".format(epoch, acc))
