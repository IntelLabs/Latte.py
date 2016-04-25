import numpy as np

class Neuron:
    def __init__(self):
        self.value = 0.0
        self.grad = 0.0

    def forward(self):
        raise NotImplementedError

    def backward(self):
        raise NotImplementedError

class WeightedNeuron(Neuron):
    def __init__(self, weights, bias):
        super().__init__()
        self.inputs = []
        self.grad_inputs = []

        self.weights = weights
        self.grad_weights = np.zeros_like(weights)

        self.bias = bias
        self.grad_bias = np.zeros_like(bias)

    def forward(self):
        for i, input_idx in enumerate_mapping(self.inputs):
            self.value += self.inputs[input_idx] * self.weights[i]
        self.value += self.bias[0]

    def backward(self):
        for i, input_idx in enumerate_mapping(self.inputs):
            self.grad_inputs[input_idx] += self.grad * self.weights[i]
        for i, input_idx in enumerate_mapping(self.inputs):
            self.grad_weights[i] += self.grad * self.inputs[input_idx]
        self.grad_bias[0] += self.grad

class DataNeuron(Neuron):
    def forward(self):
        pass

    def backward(self):
        pass
