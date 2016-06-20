import numpy as np
from .neuron import DataNeuron
from ctree.templates.nodes import FileTemplate
import ctree.c.nodes as C
import ctree
import ctypes
import os
import latte.util as util
import latte.core

ENSEMBLE_COUNTER = 0
class Ensemble:
    def __init__(self, neurons):
        self.neurons = neurons
        global ENSEMBLE_COUNTER 
        ENSEMBLE_COUNTER += 1
        self.name = "ensemble{}".format(ENSEMBLE_COUNTER)
        self.pad = tuple(0 for _ in neurons.shape)
        self.parent_group = None
        self.buffer_tiled_dims = {}
        self._tiling_info = {}
        self._transpose_info = {}
        self._vectorize_info = {}
        self._unroll_info = {}

    @property
    def tiling_info(self):
        return self._tiling_info

    @property
    def transpose_info(self):
        return self._transpose_info

    @property
    def vectorize_info(self):
        return self._vectorize_info

    @property
    def unroll_info(self):
        return self._unroll_info

    def tile(self, field, dim, factor):
        if field not in self.tiling_info:
            self.tiling_info[field] = []
        if (dim, factor) not in self.tiling_info[field]:
            self.tiling_info[field].append((dim, factor))

    def transpose(self, field, dim1, dim2):
        if field not in self.transpose_info:
            self.transpose_info[field] = []
        self.transpose_info[field].append((dim1, dim2))

    def vectorize(self, direction, loop_var, factor):
        self._vectorize_info[direction] = (loop_var, factor)

    def unroll(self, direction, loop_var, factor):
        self._unroll_info[direction] = (loop_var, factor)

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

    def is_tiled_field(self, field):
        return self.name + field in self.buffer_tiled_dims

    def get_tiled_dims(self, field):
        return self.buffer_tiled_dims[self.name + field]

    def set_buffer(self, field, buffer):
        def get():
            if field in self.tiling_info:
                untiled = buffer
                if "grad_" in field and field != "grad_inputs":
                    untiled = untiled[0]
                    # untiled = np.sum(untiled, axis=0)
                shape = untiled.shape
                tiled_shape = list(shape)
                if not isinstance(self, ActivationEnsemble) or field not in ["value", "grad"]:
                    for dim, factor in self.tiling_info[field]:
                        if field in self.batch_fields:
                            dim += 1
                        tiled_shape[dim] //= factor
                        tiled_shape.append(factor)
                untiled = untiled.reshape(tiled_shape)
                for dim, _ in reversed(self.tiling_info[field]):
                    if field in self.batch_fields:
                        dim += 1
                    untiled = util.untile(untiled, dim)
                to_return = untiled
            else:
                to_return = buffer
                if "grad_" in field and field != "grad_inputs":
                    to_return = to_return[0]
            if field in ["value", "grad"] and any(p > 0 for p in self.pad):
                _slice = [slice(None)]
                for p in self.pad:
                    if p > 0:
                        _slice.append(slice(p, -p))
                    else:
                        _slice.append(slice(None))
                to_return = to_return[tuple(_slice)]
            return to_return

        setattr(self, "get_" + field, get)
        def get_view():
            return buffer
        setattr(self, "get_" + field + "_view", get_view)

        def set(value):
            dest = buffer
            if field in ["value", "grad"] and any(p > 0 for p in self.pad):
                _slice = [slice(None)]
                for p in self.pad:
                    if p > 0:
                        _slice.append(slice(p, -p))
                    else:
                        _slice.append(slice(None))
                # dest = dest[tuple(_slice)]
            else:
                _slice = tuple(slice(None) for _ in dest.shape)
            if field in self.tiling_info:
                tiled = value
                tiled_shape = list(dest.shape)
                if not isinstance(self, ActivationEnsemble) or field not in ["value", "grad"]:
                    for dim, factor in self.tiling_info[field]:
                        if field in self.batch_fields:
                            dim += 1
                        tiled_shape[dim] //= factor
                        tiled_shape.append(factor)
                dest = dest.reshape(tiled_shape)
                for dim, _ in self.tiling_info[field]:
                    if field in self.batch_fields:
                        dim += 1
                    tiled = util.tile(tiled, dim)
                dest[_slice] = tiled
            else:
                dest[_slice] = value
        setattr(self, "set_" + field, set)
        if self.parent_group is not None:
            setattr(self.parent_group, "get_" + field, get)
            setattr(self.parent_group, "get_" + field + "_view", get_view)
            setattr(self.parent_group, "set_" + field, set)

reorder_storage_file = FileTemplate(os.path.dirname(os.path.abspath(__file__)) + "/templates/reorder_storage.c")

c_file = C.CFile("reorder_storage", [reorder_storage_file])
module = util.mpi_compile(ctree.nodes.Project([c_file]))

class DataEnsemble(Ensemble):
    def __init__(self, batch_size, shape):
        self.value = np.zeros((batch_size, ) + shape, np.float32)
        neurons = np.empty(shape, dtype='object')
        for i, _ in np.ndenumerate(neurons):
            neurons[i] = DataNeuron()
        self.reorder_4d_5d = module.get_callable("reorder_4d_5d", 
            ctypes.CFUNCTYPE(None, np.ctypeslib.ndpointer(np.float32, self.value.ndim, self.value.shape), 
                np.ctypeslib.ndpointer(np.float32, self.value.ndim, self.value.shape),
                ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int))
        super().__init__(neurons)

    def forward(self, value):
        pass
        # if self.value.ndim == 4:
        #     shape = self.value.shape
        #     self.reorder_4d_5d(self.value, value, *shape)
        # elif self.value.ndim == 2:
        #     np.copyto(value, self.value)
        # else:
        #     raise NotImplementedError()

    def set_padding(self, *padding):
        super().set_padding(*padding)
        pad = ((0, 0), ) + tuple((p, p) for p in padding)
        self.value = np.lib.pad(self.value, pad, 'constant')
        self.reorder_4d_5d = module.get_callable("reorder_4d_5d", 
            ctypes.CFUNCTYPE(None, np.ctypeslib.ndpointer(np.float32, self.value.ndim, self.value.shape), 
                np.ctypeslib.ndpointer(np.float32, self.value.ndim, self.value.shape),
                ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int))

    def update_value(self, value):
        idx = [slice(None)]
        idx += [slice(p, d + p) for p, d in zip(self.pad, self.shape)]
        np.copyto(self.value[idx], value)

class ActivationEnsemble(Ensemble):
    def __init__(self, neurons, source):
        super().__init__(neurons)
        if isinstance(source, EnsembleGroup):
            source = source.ensembles[-1]
        self.source = source

    def set_padding(self, *padding):
        self.pad = padding
        self.source.set_padding(*padding)

    def is_tiled_field(self, field):
        return self.source.is_tiled_field(field)

    @property
    def transpose_info(self):
        return self.source.transpose_info

    @property
    def vectorize_info(self):
        return self._vectorize_info

    @property
    def unroll_info(self):
        return self._unroll_info

    @property
    def tiling_info(self):
        source_tiling_info = self.source.tiling_info
        if "value" in source_tiling_info:
            self._tiling_info["inputs"] = source_tiling_info["value"]
            self._tiling_info["value"] = source_tiling_info["value"]
        if "grad" in source_tiling_info:
            self._tiling_info["grad_inputs"] = source_tiling_info["grad"]
            self._tiling_info["grad"] = source_tiling_info["grad"]
        return self._tiling_info

class LossEnsemble(Ensemble):
    pass

class AccuracyEnsemble(Ensemble):
    pass

class EnsembleGroup:
    def __init__(self, *ensembles):
        self.ensembles = ensembles
        for ensemble in ensembles:
            ensemble.parent_group = self

    def __len__(self):
        return len(self.ensembles[-1])

    @property
    def shape(self):
        return self.ensembles[-1].shape

    @property
    def ndim(self):
        return self.ensembles[-1].ndim

    def set_padding(self, *args):
        self.ensembles[-1].set_padding(*args)

    def tile(self, field, dim, factor):
        self.ensembles[-1].tile(field, dim, factor)

    @property
    def tiling_info(self):
        return self.ensembles[-1].tiling_info
