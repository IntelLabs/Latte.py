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
from ctree.templates.nodes import FileTemplate
import os
import ctree
import ctree.c.nodes as C
import ctypes
import latte
import latte.util as util

_file = FileTemplate(os.path.dirname(os.path.abspath(__file__)) + "/templates/sgd.c")

c_file = C.CFile("sgd", [_file])
module = util.mpi_compile(ctree.nodes.Project([c_file]))
_sgd_update = module.get_callable("sgd_update", 
    ctypes.CFUNCTYPE(None, ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_float, ctypes.c_float,
        ctypes.c_int, ctypes.c_int))


def sgd_update(param, grad, hist, lr, mom, batch_size):
    _sgd_update(param.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                grad.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                hist.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                ctypes.c_float(lr), ctypes.c_float(mom),
                # ctypes.c_int(param.size), ctypes.c_int(latte.core.num_threads))
                ctypes.c_int(param.size), ctypes.c_int(batch_size))
