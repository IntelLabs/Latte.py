import numpy as np
from latte import *
import time
from latte.solvers import sgd_update

def main():
    batch_size = 64
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

    # warmup
    print("Warming up...")
    for _ in range(3):
        net.forward()
        net.backward()

    forward_t_total = 0.0
    backward_t_total = 0.0
    num_trials = 10
    print("Running trials")
    for _ in range(num_trials):
        t = time.time()
        net.forward()
        forward_t_total += time.time() - t 
        t = time.time()
        net.backward()
        backward_t_total += time.time() - t 
        t = time.time()

    print("FP         : {0:.3f} ms".format(forward_t_total / num_trials * 1000))
    print("BP+WU      : {0:.3f} ms".format(backward_t_total / num_trials * 1000))
    print("Throughput : {0:.3f} img/s".format((batch_size * num_trials) / (forward_t_total + backward_t_total)))

if __name__ == '__main__':
    main()
