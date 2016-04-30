import numpy as np
from .neuron import DataNeuron

ENSEMBLE_COUNTER = 0
class Ensemble:
    def __init__(self, neurons):
        self.neurons = neurons
        global ENSEMBLE_COUNTER 
        ENSEMBLE_COUNTER += 1
        self.name = "ensemble{}".format(ENSEMBLE_COUNTER)

    @property
    def shape(self):
        return self.neurons.shape

    @property
    def ndim(self):
        return self.neurons.ndim

    def __iter__(self):
        return np.ndenumerate(self.neurons)

    def __len__(self):
        return np.prod(self.neurons.shape)

class DataEnsemble(Ensemble):
    def __init__(self, value):
        neurons = np.empty(value.shape[1:], dtype='object')
        for i, _ in np.ndenumerate(neurons):
            neurons[i] = DataNeuron()
        super().__init__(neurons)
        self.value = value

    def forward(self, value):
        shape = self.value.shape
        # for n in range(shape[0]):
        #     for ifm in range(shape[1] // 8):
        #         for y in range(shape[2]):
        #             for x in range(shape[3]):
        #                 for v in range(8):
        #                     value.flat[(((n * shape[1] // 8 + ifm) * shape[2] + y) * shape[3] + x) * 8 + v] = self.value[n, ifm * 8 + v, y, x]
        np.copyto(value, self.value)

