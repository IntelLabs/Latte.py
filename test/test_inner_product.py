import unittest
import numpy as np
from latte import *

class FCTest(unittest.TestCase):
    def test_forward(self):
        net = Net(8)
        data, data_value = MemoryDataLayer(net, (24, 24))
        fc = FullyConnectedLayer(net, data, 20)

        data_value[:, :, :] = np.random.rand(8, 24, 24)

        net.compile()
        net.forward()

        weights = net.buffers[fc.name + "weights"]
        bias    = net.buffers[fc.name + "bias"].reshape((20, ))
        _input  = net.buffers[data.name + "value"]
        np.testing.assert_array_almost_equal(_input, data_value)
        actual  = net.buffers[fc.name + "value"]
        expected = np.dot(_input.reshape((8, 24 * 24)), weights.transpose())
        for n in range(8):
            expected[n, :] += bias

        try:
            np.testing.assert_array_almost_equal(actual, expected)
        except AssertionError:
            self.fail("Arrays not equal")


if __name__ == '__main__':
    unittest.main()
