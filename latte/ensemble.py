'''
Copyright (c) 2015, Intel Corporation

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright notice,
      this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of Intel Corporation nor the names of its contributors
      may be used to endorse or promote products derived from this software
      without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
'''
import numpy as np
from .neuron import DataNeuron
from ctree.templates.nodes import FileTemplate
import ctree.c.nodes as C
import ctree
import ctypes
import os
import latte.util as util
import latte.core

if "OPENCL" in latte.config.parallel_strategy:
    import pycl as cl

ENSEMBLE_COUNTER = 0
class Ensemble:
    def __init__(self, neurons):
        self.neurons = neurons
        global ENSEMBLE_COUNTER 
        ENSEMBLE_COUNTER += 1
        self.name = "ensemble{}".format(ENSEMBLE_COUNTER)
        self.pad = tuple((0, 0) for _ in neurons.shape)
        self.filter_pad = tuple((0, 0) for _ in neurons.shape)
        self.stride = 0
        self.parent_group = None
        self.buffer_tiled_dims = {}
        self._tiling_info = {}
        self._transpose_info = {}
        self._vectorize_info = {}
        self._unroll_no_jam_info =  {"forward": [], "backward": [], "update_internal": []}
        self._unroll_info = {"forward": {}, "backward": {}, "update_internal": {}}
        self._unroll_and_jam_info = {"forward": {}, "backward": {}, "update_internal": {}}
        self._private_info = set()
        self._parallel_info = {"forward": [], "backward": [], "update_internal": []}
        self._prefetch_info = {"forward": {}, "backward": {}, "update_internal": {}}
        self.loops_to_swap = {'forward': [], 'backward': [], "update_internal": []}
        self.simd_info = {'forward': [], 'backward': [], "update_internal": []}
        self._scalar_expand_info = {}
        self._if_convert_info = {}

        self.scalar_fields = ["value", "grad"]
        self.use_libxsmm_lib = 0

    @property
    def private_info(self):
        return self._private_info

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
    def prefetch_info(self):
        return self._prefetch_info

    @property
    def scalar_expand_info(self):
        return self._scalar_expand_info

    @property
    def if_convert_info(self):
        return self._if_convert_info


    @property
    def unroll_info(self):
        return self._unroll_info

    @property
    def unroll_and_jam_info(self):
        return self._unroll_and_jam_info
    @property
    def unroll_no_jam_info(self):
        return self._unroll_no_jam_info

    @property
    def parallel_info(self):
        return self._parallel_info

    def simd(self, phase, loop_var):
        self.simd_info[phase].append(loop_var)

    def privatize(self, buffer):
        self.private_info.add(buffer)

    def tile(self, field, dim, factor):
        if field not in self.tiling_info:
            self.tiling_info[field] = []
        if (dim, factor) not in self.tiling_info[field]:
            self.tiling_info[field].append((dim, factor))

    def transpose(self, field, dim1, dim2):
        if field not in self.transpose_info:
            self.transpose_info[field] = []
        self.transpose_info[field].append((dim1, dim2))

    def vectorize(self, phase, loop_var, factor):
        self._vectorize_info[phase] = (loop_var, factor)

    def unroll(self, phase, loop_var, factor,unroll_type=0):
        dict = self._unroll_info[phase]
        dict[loop_var] =  (factor, unroll_type)
 
    def unroll_and_jam(self, phase, loop_var, factor,unroll_type=0):
        dict = self._unroll_and_jam_info[phase]
        dict[loop_var] =  (factor, unroll_type)
    def unroll_no_jam(self, phase, loop_var, factor,unroll_type=0):
          self._unroll_no_jam_info[phase].append([loop_var, factor, unroll_type])


    def scalar_expand(self, phase, scalar_vars):
        self._scalar_expand_info[phase] =  scalar_vars

    def if_convert(self, phase):
        self._if_convert_info[phase] =  True



    def parallelize(self, phase, loop_var):
        self._parallel_info[phase].append(loop_var)

    def prefetch(self, phase, prefetch_dict_list):
        self._prefetch_info[phase]=prefetch_dict_list

    def reset_prefetch(self, phase):
        self._prefetch_info[phase]={}

    def swap_loops(self, phase, loop_vars):
        assert isinstance(loop_vars, tuple) and len(loop_vars) == 2
        self.loops_to_swap[phase].append(loop_vars)

    def use_libxsmm(self, use_libxsmm):
        self.use_libxsmm_lib = use_libxsmm

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

    def set_filter_padding(self, *padding):
        self.filter_pad = padding

    def is_tiled_field(self, field):
        return self.name + field in self.buffer_tiled_dims

    def get_tiled_dims(self, field):
        return self.buffer_tiled_dims[self.name + field]

    def set_buffer(self, field, buffer, cl_buffer=None):
        def get():
            if cl_buffer is not None:
                _, evt = cl.buffer_to_ndarray(latte.config.cl_queue, cl_buffer, out=buffer)
                evt.wait()
            if field in self.tiling_info:
                untiled = buffer
                if field in self.private_info:
                    untiled = untiled[0]
                shape = untiled.shape
                tiled_shape = list(shape)
                if not isinstance(self, ActivationEnsemble) or field not in ["value", "grad"]:
                    for dim, factor in self.tiling_info[field]:
                        if field in self.batch_fields:
                            dim += 1
                        tiled_shape[dim] //= factor
                        tiled_shape.append(factor)
                #print(tiled_shape)
                untiled = untiled.reshape(tiled_shape)
                for dim, _ in reversed(self.tiling_info[field]):
                    if field in self.batch_fields:
                        dim += 1
                    untiled = util.untile(untiled, dim)
                to_return = untiled
            else:
                to_return = buffer
                if "grad_" in field and "grad_inputs" not in field:
                    to_return = to_return[0]
            if field in ["value", "grad"] and any(p != (0, 0) for p in self.pad):
                _slice = [slice(None)]
                for p in self.pad:
                    if p != (0, 0):
                        _slice.append(slice(p[0], -p[1]))
                    else:
                        _slice.append(slice(None))
                to_return = to_return[tuple(_slice)]
            if field in ["value", "grad"] and any(p != (0, 0) for p in self.filter_pad):
                _slice = [slice(None)]
                for p in self.filter_pad:
                    if p != (0, 0):
                        _slice.append(slice(p[0], -p[1]))
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
            if field in ["value", "grad"] and any(p != (0, 0) for p in self.pad):
                _slice = [slice(None)]
                for i, p in enumerate(self.pad):
                    if p != (0, 0):
                        _slice.append(slice(p[0], -p[1]))
                    else:
                        _slice.append(slice(None))
                # dest = dest[tuple(_slice)]
            else:
                _slice = [slice(None) for _ in dest.shape]

            if field in self.tiling_info:
                tiled = value
                # Handle padding of weights and bias if set using set_value
                if field in ["weights", "bias"]:
                    padding = list()
                    # generate padding tuple
                    for dim in tiled.shape:
                        padding += ((0,0),)
                    for dim, factor in self.tiling_info[field]:
                        if tiled.shape[dim] % factor != 0:
                            padding[dim] = (0, factor - tiled.shape[dim] % factor)
                    tiled = np.lib.pad(tiled, padding, 'constant')
                for dim, _ in self.tiling_info[field]:
                    if field in self.batch_fields:
                        dim += 1
                    if not isinstance(self, ActivationEnsemble):
                        _slice.append(_slice[dim])
                        _slice[dim] = slice(None)
                    tiled = util.tile(tiled, dim)
                tiled_shape = list(dest.shape)
                if not isinstance(self, ActivationEnsemble) or field not in ["value", "grad"]:
                    for dim, factor in self.tiling_info[field]:
                        if field in self.batch_fields:
                            dim += 1
                        if tiled_shape[dim] < factor:
                            factor = tiled_shape[dim]
                        elif tiled_shape[dim] % factor != 0:
                            raise NotImplementedError()
                        tiled_shape[dim] //= factor
                        tiled_shape.append(factor)
                dest = dest.reshape(tiled_shape)
                dest[_slice] = tiled
            else:
                dest[_slice] = value
            if cl_buffer is not None:
                _, evt = cl.buffer_from_ndarray(latte.config.cl_queue, dest, buf=cl_buffer)
                evt.wait()
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
        pad = ((0, 0), ) + padding
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
    def unroll_and_jam_info(self):
        return self._unroll_and_jam_info

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

class ConcatEnsemble(Ensemble):
    def __init__(self, neurons):
        super().__init__(neurons)

class LRNEnsemble(Ensemble):
    def __init__(self, neurons):
        super().__init__(neurons)

   
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

    @property
    def pad(self):
        return self.ensembles[-1].pad

    def tile(self, field, dim, factor):
        self.ensembles[-1].tile(field, dim, factor)

    @property
    def tiling_info(self):
        return self.ensembles[-1].tiling_info

    @property
    def parallel_info(self):
        return self.ensembles[-1].parallel_info
