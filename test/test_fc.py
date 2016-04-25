import unittest
import numpy as np
from latte import *

class FCTest(unittest.TestCase):
    def _check_equal(self, actual, expected):
        try:
            np.testing.assert_array_almost_equal(actual, expected)
        except AssertionError:
            self.fail("Arrays not equal")

    def test_forward_backward(self):
        net = Net(8)
        data, data_value = MemoryDataLayer(net, (24, 24))
        fc1 = FullyConnectedLayer(net, data, 20)
        fc2 = FullyConnectedLayer(net, fc1, 20)

        data_value[:, :, :] = np.random.rand(8, 24, 24)

        net.compile()
        net.forward()

        weights = net.buffers[fc1.name + "weights"]
        bias    = net.buffers[fc1.name + "bias"].reshape((20, ))
        _input  = net.buffers[data.name + "value"]
        np.testing.assert_array_almost_equal(_input, data_value)
        actual  = net.buffers[fc1.name + "value"]
        expected = np.dot(data_value.reshape((8, 24 * 24)), weights.transpose())
        for n in range(8):
            expected[n, :] += bias

        self._check_equal(actual, expected)

        top_grad = net.buffers[fc2.name + "grad"]
        np.copyto(top_grad, np.random.rand(*top_grad.shape))

        net.backward()
        weights = net.buffers[fc2.name + "weights"]

        bot_grad = net.buffers[fc1.name + "grad"]
        expected_bot_grad = np.dot(top_grad, weights)
        self._check_equal(bot_grad, expected_bot_grad)

        weights_grad = net.buffers[fc2.name + "grad_weights"]
        expected_weights_grad = np.dot(top_grad.transpose(), actual)
        self._check_equal(weights_grad, expected_weights_grad)

        bias_grad = net.buffers[fc2.name + "grad_bias"]
        expected_bias_grad = np.sum(top_grad, 0).reshape(20, 1)
        self._check_equal(bias_grad, expected_bias_grad)

if __name__ == '__main__':
    unittest.main()
