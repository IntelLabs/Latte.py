import unittest
import numpy as np
from latte import *
import time
def main():
    batch_size = 32
    net = Net(batch_size)
    channels, height, width = 256, 14, 14
    pad = 0
    ofm = 512
    data, data_value = MemoryDataLayer(net, (channels, height, width))
    conv1 = ConvLayer(net, data, num_filters=ofm, kernel=3, stride=1, pad=pad)

    data_value[:, :, :] = np.random.rand(batch_size, channels, height, width)

    net.compile()

    assert(len(net.forward_tasks) == 2)
    assert(len(net.backward_tasks) == 1)

    # warmup
    for _ in range(3):
        net.forward_tasks[1]()
        net.backward_tasks[0]()

    forward_t_total = 0.0
    backward_t_total = 0.0
    num_trials = 2
    for _ in range(num_trials):
        t = time.time()
        net.forward_tasks[1]()
        forward_t_total += time.time() - t 
        t = time.time()
        net.backward_tasks[0]()
        backward_t_total += time.time() - t 


    _, ofm, oh, ow = net.buffers[conv1.name + "value"].shape
    gflops = (batch_size * channels * ofm * oh * ow * (2 * 3 * 3)) * num_trials * 1e-9
    print("GFLOPS/s : fp = {}, bp = {}".format(gflops / forward_t_total, (gflops * 2) / backward_t_total))

if __name__ == '__main__':
    main()
