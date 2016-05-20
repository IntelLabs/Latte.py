import pytest
import numpy as np
from latte import *
import latte.util as util

def check_equal(actual, expected, atol=1e-6):
    assert np.allclose(actual, expected, atol=atol)

def reference_softmax_forward(_input, label):
    output = np.zeros_like(_input)
    loss = 0.0
    for n in range(_input.shape[0]):
        x = _input[n]
        e_x = np.exp(x - np.max(x))
        output[n] = e_x / e_x.sum()
        loss -= np.log(max(output[n, int(label[n, 0])], np.finfo(np.float32).min))
    return output, loss

def reference_softmax_backward(prob, label):
    bot_grad = np.zeros_like(prob)
    np.copyto(bot_grad, prob)
    for n in range(prob.shape[0]):
        bot_grad[n, int(label[n, 0])] -= 1
    bot_grad /= np.prod(bot_grad.shape)
    return bot_grad


def test_forward_backward():
    net = Net(8)
    net.force_backward = True
    data = MemoryDataLayer(net, (1000, ))
    fc1, fc1bias = FullyConnectedLayer(net, data, 1000, )
    label = MemoryDataLayer(net, (1, ))
    softmax = SoftmaxLossLayer(net, fc1bias, label)
    
    data_value = np.random.rand(8, 1000)
    data.set_value(data_value)
    
    label_value = np.floor(np.random.rand(8, 1) * 1000)
    label.set_value(label_value)

    net.compile()
    net.forward()
    bottom = net.buffers[fc1bias.name + "value"].reshape(8, 1000)

    expected, loss = reference_softmax_forward(bottom, label_value)
    assert np.allclose(softmax.prob, expected)
    assert np.allclose(net.loss, loss)
    net.backward()

    expected_grad = reference_softmax_backward(expected, label_value)
    bot_grad = net.buffers[fc1bias.name + "grad"].reshape(8, 1000)
    assert np.allclose(bot_grad, expected_grad)
