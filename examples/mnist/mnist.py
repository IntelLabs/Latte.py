import numpy as np
from latte import *
from latte.solvers import sgd_update
import random
from data_loader import load_mnist

train_data, train_label = load_mnist(dataset="training", path="./data")
test_data, test_label   = load_mnist(dataset="testing", path="./data")

num_train = train_data.shape[0]
train_data = np.pad(train_data.reshape(num_train, 1, 28, 28), [(0, 0), (0, 7), (0, 0), (0, 0)], mode='constant')
train_label = train_label.reshape(num_train, 1)

num_test = test_data.shape[0]
test_data = np.pad(test_data.reshape(num_test, 1, 28, 28), [(0, 0), (0, 7), (0, 0), (0, 0)], mode='constant')
test_label = test_label.reshape(num_test, 1)


batch_size = 100
net = Net(batch_size)

data     = MemoryDataLayer(net, train_data[0].shape)
label    = MemoryDataLayer(net, train_label[0].shape)

_, conv1 = ConvLayer(net, data, num_filters=16, kernel=5, stride=1, pad=0)
relu1    = ReLULayer(net, conv1)
pool1    = MaxPoolingLayer(net, relu1, kernel=2, stride=2, pad=0)

_, conv2 = ConvLayer(net, pool1, num_filters=64, kernel=5, stride=1, pad=0)
relu2    = ReLULayer(net, conv2)
pool2    = MaxPoolingLayer(net, relu2, kernel=2, stride=2, pad=0)

_, fc3   = FullyConnectedLayer(net, pool2, 512)
relu3    = ReLULayer(net, fc3)
_, fc4   = FullyConnectedLayer(net, relu3, 16)

# loss     = SoftmaxLossLayer(net, fc4, label)
# acc      = AccuracyLayer(net, fc4, label)

net.compile()

params = []
for name in net.buffers.keys():
    if name.endswith("weights") and "grad_" not in name:
        ensemble_name = name[:-len("weights")]
        grad = net.buffers[ensemble_name + "grad_weights"]
        params.append((net.buffers[name],
                       grad, 
                       np.zeros_like(net.buffers[name])))
    elif name.endswith("bias") and "grad_" not in name:
        ensemble_name = name[:-len("bias")]
        grad = net.buffers[ensemble_name + "grad_bias"]
        params.append((net.buffers[name],
                       grad, 
                       np.zeros_like(net.buffers[name])))

base_lr = .01
gamma = .0001
power = .75

train_batches = [i for i in range(0, num_train, batch_size)]

output = net.buffers[fc4.name + "value"].reshape(batch_size, 16)
prob = np.zeros_like(output)

output_grad = net.buffers[fc4.name + "grad"].reshape(batch_size, 16)

for epoch in range(10):
    random.shuffle(train_batches)
    print("Epoch {} - Training...".format(epoch))
    for i, n in enumerate(train_batches):
        data.set_value(train_data[n:n+batch_size])
        label_value = train_label[n:n+batch_size]
        label.set_value(label_value)
        net.forward()

        loss = 0.0
        for n in range(batch_size):
            x = output[n]
            e_x = np.exp(x - np.max(x))
            prob[n] = e_x / e_x.sum()
            loss -= np.log(max(prob[n, int(label_value[n, 0])], np.finfo(np.float32).min))
        loss /= batch_size

        if i % 100 == 0:
            print("Epoch {}, Train Iteration {} - Loss = {}".format(epoch, i, loss))

        np.copyto(output_grad, prob)
        for n in range(batch_size):
            output_grad[n, int(label_value[n, 0])] -= 1
        output_grad /= batch_size

        net.backward()
        lr = base_lr * (1 + gamma * i)**power
        mom = .9
        for param in params:
            # expected = param[0] - (param[2] * mom + np.sum(param[1], axis=0) * lr)
            sgd_update(param[0], param[1], param[2], lr, mom)
        net.clear_values()
        net.clear_grad()
        net.loss = 0.0

    print("Epoch {} - Testing...".format(epoch))
    acc = 0
    for i, n in enumerate(range(0, num_test, batch_size)):
        data.set_value(test_data[n:n+batch_size])
        label_value = test_label[n:n+batch_size]
        label.set_value(label_value)
        net.test()

        accuracy = 0.0
        for n in range(batch_size):
            if np.argmax(output[n]) == label_value[n, 0]:
                accuracy += 1
        acc += accuracy / batch_size
        net.clear_values()
    acc /= (num_test / batch_size)
    acc *= 100
    print("Epoch {} - Validation accuracy = {:.3f}%".format(epoch, acc))
