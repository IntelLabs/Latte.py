import unittest
import numpy as np
from latte import *
import latte.util as util

class FCTest(unittest.TestCase):
    def _check_equal(self, actual, expected, decimal=6):
        try:
            np.testing.assert_array_almost_equal(actual, expected, decimal)
        except AssertionError:
            self.fail("Arrays not equal")

    def test_forward_backward(self):
        net = Net(8)
        data, data_value = MemoryDataLayer(net, (24, 24))
        fc1 = FullyConnectedLayer(net, data, 24)
        fc2 = FullyConnectedLayer(net, fc1, 24)

        data_value[:, :, :] = np.random.rand(8, 24, 24)

        net.compile()
        net.forward()

        weights = net.buffers[fc1.name + "weights"]
        weights_converted = util.convert_3d_2d(weights)
        # bias    = net.buffers[fc1.name + "bias"].reshape((24, ))
        _input  = net.buffers[data.name + "value"]
        np.testing.assert_array_almost_equal(_input, data_value)
        actual  = net.buffers[fc1.name + "value"]
        expected = np.dot(data_value.reshape((8, 24 * 24)), weights_converted.transpose())
        # for n in range(8):
        #     expected[n, :] += bias

        self._check_equal(actual, expected, 5)

        top_grad = net.buffers[fc2.name + "grad"]
        np.copyto(top_grad, np.random.rand(*top_grad.shape))

        net.backward()
        weights = net.buffers[fc2.name + "weights"]
        weights_converted = util.convert_3d_2d(weights)

        bot_grad = net.buffers[fc1.name + "grad"]
        expected_bot_grad = np.dot(top_grad, weights_converted)
        self._check_equal(bot_grad, expected_bot_grad)

        weights_grad = net.buffers[fc2.name + "grad_weights"]
        expected_weights_grad = np.dot(top_grad.transpose(), actual)
        weights_grad_converted = util.convert_3d_2d(weights_grad)
        self._check_equal(weights_grad_converted, expected_weights_grad)

        # bias_grad = net.buffers[fc2.name + "grad_bias"]
        # expected_bias_grad = np.sum(top_grad, 0).reshape(24, 1)
        # self._check_equal(bias_grad, expected_bias_grad)

if __name__ == '__main__':
    unittest.main()
