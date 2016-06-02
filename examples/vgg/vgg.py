import numpy as np
from latte import *
import time
from latte.solvers import sgd_update
from mpi4py import MPI
from latte.math import compute_softmax_loss, softmax_loss_backprop, compute_accuracy
from collections import namedtuple
from scipy.ndimage import imread
from scipy.misc import imresize

comm = MPI.COMM_WORLD
batch_size = 128 // comm.size
net = Net(batch_size)

data     = MemoryDataLayer(net, (8, 224, 224))

_, conv1 = ConvLayer(       net, data, num_filters=64, kernel=3, stride=1, pad=1)
relu1    = ReLULayer(       net, conv1)
pool1    = MaxPoolingLayer( net, relu1, kernel=2, stride=2, pad=0)

_, conv2 = ConvLayer(       net, pool1, num_filters=128, kernel=3, stride=1, pad=1)
relu2    = ReLULayer(       net, conv2)
pool2    = MaxPoolingLayer( net, relu2, kernel=2, stride=2, pad=0)

_, conv3_1 = ConvLayer(      net, pool2,   num_filters=256, kernel=3, stride=1, pad=1)
relu3_1    = ReLULayer(      net, conv3_1)
_, conv3_2 = ConvLayer(      net, relu3_1, num_filters=256, kernel=3, stride=1, pad=1)
relu3_2    = ReLULayer(      net, conv3_2)
pool3      = MaxPoolingLayer(net, relu3_2, kernel=2, stride=2, pad=0)

_, conv4_1 = ConvLayer(      net, pool3,   num_filters=512, kernel=3, stride=1, pad=1)
relu4_1    = ReLULayer(      net, conv4_1)
_, conv4_2 = ConvLayer(      net, relu4_1, num_filters=512, kernel=3, stride=1, pad=1)
relu4_2    = ReLULayer(      net, conv4_2)
pool4      = MaxPoolingLayer(net, relu4_2, kernel=2, stride=2, pad=0)

_, conv5_1 = ConvLayer(      net, pool4,   num_filters=512, kernel=3, stride=1, pad=1)
relu5_1    = ReLULayer(      net, conv5_1)
_, conv5_2 = ConvLayer(      net, relu5_1, num_filters=512, kernel=3, stride=1, pad=1)
relu5_2    = ReLULayer(      net, conv5_2)
pool5      = MaxPoolingLayer(net, relu5_2, kernel=2, stride=2, pad=0)

fc6, fc6bias = FullyConnectedLayer(net, pool5, 4096)
fc7, fc7bias = FullyConnectedLayer(net, fc6bias, 4096)
fc8, fc8bias = FullyConnectedLayer(net, fc7bias, 1000)

data.set_value(np.random.rand(batch_size, 8, 224, 224))

print("Compiling...")
net.compile()

base_lr = .01
gamma = .0001
power = .75

Param = namedtuple('Param', ['value', 'grad', 'hist', 'request'])

params = []
for name in net.buffers.keys():
    if name.endswith("weights") and "grad_" not in name:
        ensemble_name = name[:-len("weights")]
        grad = net.buffers[ensemble_name + "grad_weights"]
        params.append(Param(net.buffers[name],
                            grad, 
                            np.zeros_like(net.buffers[name]),
                            None))
    elif name.endswith("bias") and "grad_" not in name:
        ensemble_name = name[:-len("bias")]
        grad = net.buffers[ensemble_name + "grad_bias"]
        params.append(Param(net.buffers[name],
                            grad, 
                            np.zeros_like(net.buffers[name]),
                            None))

output = net.buffers[fc8.name + "value"].reshape(batch_size, 1000)
prob = np.zeros_like(output)
output_grad = net.buffers[fc8.name + "grad"].reshape(batch_size, 1000)

train_info = open("data/train_metadata.txt", "r").readlines()
test_info = open("data/test_metadata.txt", "r").readlines()

num_train = len(train_info)
num_train = len(test_info)
train_batches = [i for i in range(0, num_train, batch_size)]
test_batches = [i for i in range(0, num_train, batch_size)]

train_chunk = num_train / comm.Get_size()
if num_train % comm.Get_size() != 0:
    chunk += 1
train_slice = comm.Get_rank() * train_chunk

test_chunk = num_test / comm.Get_size()
if num_test % comm.Get_size() != 0:
    chunk += 1
test_slice = comm.Get_rank() * test_chunk

data_value = np.zeros((batch_size, 8, 224, 224), np.float32)
label_value = np.zeros((batch_size, 1), np.int32)
prefix = ""

for epoch in range(10):
    if comm.Get_rank() == 0:
        random.shuffle(train_batches)
        comm.bcast(train_batches, root=0)
    else:
        comm.bcast(train_batches, root=0)
    print("Epoch {} - Training...".format(epoch))
    for i, n in enumerate(train_batches[train_slice:train_slice+train_chunk]):
        t = time.time()
        for j, info in enumerate(train_info[n:n+batch_size]):
            path, label = info.split()
            data_value[j, :3, :, :] = resize(imread(prefix + path), (3, 224, 224))
            label_value[j, 0] = int(label)
        data.set_value(data_value)

        net.forward()

        loss = compute_softmax_loss(output, prob, label_value)

        if i % 100 == 0:
            print("Epoch {}, Train Iteration {} - Loss = {}".format(epoch, i, loss))

        softmax_loss_backprop(output_grad, prob, label_value)

        net.backward()

        lr = base_lr * (1 + gamma * i)**power
        mom = .9

        for param in params:
            param[3] = comm.Iallreduce(MPI.IN_PLACE, param[1], op=MPI.SUM)
        for param in params:
            param[3].Wait()
            sgd_update(param[0], param[1], param[2], lr, mom, batch_size)
        t_total += time.time() - t 

        net.clear_values()
        net.clear_grad()

    print("Epoch {} - Testing...".format(epoch))
    acc = 0
    for i, n in enumerate(test_batches[test_slice:test_slice+test_chunk]):
        for j, info in enumerate(test_info[n:n+batch_size]):
            path, label = info.split()
            data_value[j, :3, :, :] = resize(imread(prefix + path), (3, 224, 224))
            label_value[j, 0] = int(label)
        data.set_value(data_value)
        net.test()

        acc += compute_accuracy(output, label_value)
        net.clear_values()
    acc /= (num_test / batch_size)
    acc *= 100
    print("Epoch {} - Validation accuracy = {:.3f}%".format(epoch, acc))

