import numpy as np
from latte import *
import time
from tqdm import tqdm 

def main():
    batch_size = 64
    net = Net(batch_size)
    print("Benchmark Config")
    print("    batch_size = {}".format(batch_size))

    data, data_value = MemoryDataLayer(net, (8, 224, 224))

    conv1, _ = ConvLayer(       net, data, num_filters=64, kernel=3, stride=1, pad=1)
    relu1    = ReLULayer(       net, conv1)
    pool1    = MaxPoolingLayer( net, relu1, kernel=2, stride=2, pad=0)

    conv2, _ = ConvLayer(       net, pool1, num_filters=128, kernel=3, stride=1, pad=1)
    relu2    = ReLULayer(       net, conv2)
    pool2    = MaxPoolingLayer( net, relu2, kernel=2, stride=2, pad=0)

    conv3_1, _ = ConvLayer(      net, pool2,   num_filters=256, kernel=3, stride=1, pad=1)
    relu3_1    = ReLULayer(      net, conv3_1)
    conv3_2, _ = ConvLayer(      net, relu3_1, num_filters=256, kernel=3, stride=1, pad=1)
    relu3_2    = ReLULayer(      net, conv3_2)
    pool3      = MaxPoolingLayer(net, relu3_2, kernel=2, stride=2, pad=0)

    conv4_1, _ = ConvLayer(      net, pool3,   num_filters=512, kernel=3, stride=1, pad=1)
    relu4_1    = ReLULayer(      net, conv4_1)
    conv4_2, _ = ConvLayer(      net, relu4_1, num_filters=512, kernel=3, stride=1, pad=1)
    relu4_2    = ReLULayer(      net, conv4_2)
    pool4      = MaxPoolingLayer(net, relu4_2, kernel=2, stride=2, pad=0)

    conv5_1, _ = ConvLayer(      net, pool4,   num_filters=512, kernel=3, stride=1, pad=1)
    relu5_1    = ReLULayer(      net, conv5_1)
    conv5_2, _ = ConvLayer(      net, relu5_1, num_filters=512, kernel=3, stride=1, pad=1)
    relu5_2    = ReLULayer(      net, conv5_2)
    pool5      = MaxPoolingLayer(net, relu5_2, kernel=2, stride=2, pad=0)

    fc6 = FullyConnectedLayer(net, pool5, 4096)
    fc7 = FullyConnectedLayer(net, fc6, 4096)
    fc8 = FullyConnectedLayer(net, fc7, 1000)

    data_value[:, :, :, :] = np.random.rand(batch_size, 8, 224, 224)

    print("Compiling...")
    net.compile()

    # warmup
#     print("Warming up...")
#     for _ in range(3):
#         net.forward()
#         net.backward()
# 
#     forward_t_total = 0.0
#     backward_t_total = 0.0
#     num_trials = 1
#     print("Running trials")
#     for _ in tqdm(range(num_trials), ncols=100):
#         t = time.time()
#         net.forward()
#         forward_t_total += time.time() - t 
#         t = time.time()
#         net.backward()
#         backward_t_total += time.time() - t 
# 
#     print("FP    : {0:.3f} ms".format(forward_t_total / num_trials * 1000))
#     print("BP+WU : {0:.3f} ms".format(backward_t_total / num_trials * 1000))
# 
if __name__ == '__main__':
    main()
