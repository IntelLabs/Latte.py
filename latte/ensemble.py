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
    def batch_fields(self):
        return self.neurons.flat[0].batch_fields

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
        if True and self.value.ndim == 4:
            shape = self.value.shape
            value_reshaped = value.reshape(shape[0], shape[1] // 8, shape[2], shape[3], 8)
            for n in range(shape[0]):
                for ifm in range(shape[1] // 8):
                    for y in range(shape[2]):
                        for x in range(shape[3]):
                            for v in range(8):
                                value_reshaped[n, ifm, y, x, v] = self.value[n, ifm * 8 + v, y, x]
        else:
            np.copyto(value, self.value)

class ActivationEnsemble(Ensemble):
    pass
