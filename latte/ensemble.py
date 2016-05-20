import numpy as np
from .neuron import DataNeuron
from ctree.templates.nodes import FileTemplate
import ctree.c.nodes as C
import ctree
import ctypes
import os

ENSEMBLE_COUNTER = 0
class Ensemble:
    def __init__(self, neurons):
        self.neurons = neurons
        global ENSEMBLE_COUNTER 
        ENSEMBLE_COUNTER += 1
        self.name = "ensemble{}".format(ENSEMBLE_COUNTER)
        self.pad = tuple(0 for _ in neurons.shape)

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

    def set_padding(self, *padding):
        self.pad = padding

reorder_storage_file = FileTemplate(os.path.dirname(os.path.abspath(__file__)) + "/templates/reorder_storage.c")

c_file = C.CFile("reorder_storage", [reorder_storage_file])
module = ctree.nodes.Project([c_file]).codegen()

class DataEnsemble(Ensemble):
    def __init__(self, batch_size, shape):
        self.value = np.zeros((batch_size, *shape), np.float32)
        neurons = np.empty(shape, dtype='object')
        for i, _ in np.ndenumerate(neurons):
            neurons[i] = DataNeuron()
        self.reorder_4d_5d = module.get_callable("reorder_4d_5d", 
            ctypes.CFUNCTYPE(None, np.ctypeslib.ndpointer(np.float32, self.value.ndim, self.value.shape), 
                np.ctypeslib.ndpointer(np.float32, self.value.ndim, self.value.shape),
                ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int))
        super().__init__(neurons)

    def forward(self, value):
        if self.value.ndim == 4:
            shape = self.value.shape
            self.reorder_4d_5d(self.value, value, *shape)
        elif self.value.ndim == 2:
            np.copyto(value, self.value)
        else:
            raise NotImplementedError()

    def set_padding(self, *padding):
        super().set_padding(*padding)
        pad = ((0, 0), ) + tuple((p, p) for p in padding)
        self.value = np.lib.pad(self.value, pad, 'constant')
        self.reorder_4d_5d = module.get_callable("reorder_4d_5d", 
            ctypes.CFUNCTYPE(None, np.ctypeslib.ndpointer(np.float32, self.value.ndim, self.value.shape), 
                np.ctypeslib.ndpointer(np.float32, self.value.ndim, self.value.shape),
                ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int))

    def set_value(self, value):
        idx = [slice(None)]
        idx += [slice(p, d + p) for p, d in zip(self.pad, self.shape)]
        np.copyto(self.value[idx], value)

class ActivationEnsemble(Ensemble):
    def __init__(self, neurons, source):
        super().__init__(neurons)
        self.source = source

    def set_padding(self, *padding):
        self.pad = padding
        self.source.set_padding(*padding)

class LossEnsemble(Ensemble):
    pass
