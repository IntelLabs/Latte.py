import numpy as np

class Neuron:
    batch_fields = ["value", "grad"]

    def __init__(self):
        self.value = 0.0
        self.grad = 0.0

    def forward(self):
        raise NotImplementedError

    def backward(self):
        raise NotImplementedError

class DataNeuron(Neuron):
    def forward(self):
        pass

    def backward(self):
        pass

class WeightedNeuron(Neuron):
    def __init__(self, weights, grad_weights):
        super().__init__()
        self.inputs = []
        self.grad_inputs = []

        self.weights = weights
        self.grad_weights = grad_weights


class BiasNeuron(Neuron):
    def __init__(self, bias, grad_bias):
        self.inputs = []
        self.grad_inputs = []

        self.bias = bias
        self.grad_bias = grad_bias

    def forward(self):
        self.value = self.input + self.bias[0]

    def backward(self):
        self.grad_bias[0] += self.grad

