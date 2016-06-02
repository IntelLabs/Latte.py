import numpy as np
from latte import *
import time
from latte.solvers import sgd_update
from mpi4py import MPI
from latte.math import compute_softmax_loss, softmax_loss_backprop, compute_accuracy

def main():
    comm = MPI.COMM_WORLD
    batch_size = 64 // comm.size
    net = Net(batch_size)
    print("Benchmark Config")
    print("    batch_size = {}".format(batch_size))

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

    params = []
    for name in net.buffers.keys():
        if name.endswith("weights") and "grad_" not in name:
            ensemble_name = name[:-len("weights")]
            grad = net.buffers[ensemble_name + "grad_weights"]
            params.append((net.buffers[name],
                           grad, 
                           np.zeros_like(net.buffers[name]),
                           None))
        elif name.endswith("bias") and "grad_" not in name:
            ensemble_name = name[:-len("bias")]
            grad = net.buffers[ensemble_name + "grad_bias"]
            params.append((net.buffers[name],
                           grad, 
                           np.zeros_like(net.buffers[name]),
                           None))

    output = net.buffers[fc8.name + "value"].reshape(batch_size, 1000)
    prob = np.zeros_like(output)

    output_grad = net.buffers[fc8.name + "grad"].reshape(batch_size, 1000)

    # warmup
    print("Warming up...")
    for _ in range(3):
        net.forward()
        net.backward()

    t_total = 0.0
    num_trials = 10
    print("Running trials")
    for i in range(num_trials):
        t = time.time()
        net.forward()

        label_value = np.floor(np.random.rand(batch_size, 1) * 1000)
        loss = compute_softmax_loss(output, prob, label_value)
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

if __name__ == '__main__':
    main()
