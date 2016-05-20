from ctree.templates.nodes import FileTemplate
import os
import ctree
import ctree.c.nodes as C
import ctypes

_file = FileTemplate(os.path.dirname(os.path.abspath(__file__)) + "/templates/sgd.c")

c_file = C.CFile("sgd", [_file])
module = ctree.nodes.Project([c_file]).codegen()
_sgd_update = module.get_callable("sgd_update", 
    ctypes.CFUNCTYPE(None, ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_float, ctypes.c_float,
        ctypes.c_int))


def sgd_update(param, grad, hist, lr, mom):
    _sgd_update(param.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                grad.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                hist.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                ctypes.c_float(lr), ctypes.c_float(mom), ctypes.c_int(param.size))
