import numpy as np

class Neuron:
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
