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
import inspect
import textwrap
import ast
import astor
import collections
import ctree.c.nodes as C
import numpy as np
import latte
from copy import deepcopy
import sys
import ctree
from ctree.templates.nodes import FileTemplate, StringTemplate
import ctypes
import os

MPI_ENABLED = True
try:
    from mpi4py import MPI
except ImportError:
    MPI_ENABLED = False

def mpi_compile(project):
    if not MPI_ENABLED:
        module = project.codegen()
    elif MPI.COMM_WORLD.Get_rank() == 0:
        module = project.codegen()
        MPI.COMM_WORLD.bcast(module, root=0)
    else:
        module = MPI.COMM_WORLD.bcast(None, root=0)
    return module

_file = FileTemplate(os.path.dirname(os.path.abspath(__file__)) + "/templates/utils.c")
 
c_file = C.CFile("util", [_file])
module = ctree.nodes.Project([c_file]).codegen()
get_cpu_freq = module.get_callable("get_cpu_freq", 
    ctypes.CFUNCTYPE(ctypes.c_double))

# 
# def aligned(shape, dtype, alignment=64, init=np.empty):
#     if isinstance(shape, list):
#         shape = tuple(shape)
#     pointer = aligned_malloc(np.prod(shape) * np.dtype(dtype).itemsize)
#     typ = np.ctypeslib.ndpointer(dtype=dtype, ndim=len(shape), shape=shape)
#     arr = np.ctypeslib.as_array(typ(pointer), shape)
#     if init == np.empty:
#         return arr
#     elif init == np.zeros:
#         arr.fill(0.0)
#         return arr
#     else:
#         raise NotImplementedError()



def insert_malloc(body, shape, name, dtype, _global=False):
    shape_str = "".join("[{}]".format(d) for d in shape[1:])
    size=1;
    for d in shape:
        size *= d  

 
    body.insert(0, StringTemplate(
        "$global$type (* $arg_name0)$shape = ($global$type (*)$cast) $arg_name1;",
        {
            "arg_name0":C.SymbolRef(name),
            "arg_name1": C.FunctionCall(C.SymbolRef('_mm_malloc'), [C.Mul(C.Constant(size), C.FunctionCall(C.SymbolRef('sizeof'), [ctree.types.codegen_type(ctree.types.get_c_type_from_numpy_dtype(dtype)())])), C.Constant(64)]),
            "shape": C.SymbolRef(shape_str),
            "cast": C.SymbolRef(shape_str),
            "type": C.SymbolRef(ctree.types.codegen_type(ctree.types.get_c_type_from_numpy_dtype(dtype)())),
            "global": C.SymbolRef("__global " if _global else "")
        }))



def insert_free(name):
 
    return(StringTemplate(
        "$free;", 
        {
            "free": C.FunctionCall(C.SymbolRef('_mm_free'), [C.SymbolRef(name)]),
        }))

def insert_cast(body, shape, name, dtype, _global=False):
    shape_str = "".join("[{}]".format(d) for d in shape)

    body.insert(0, StringTemplate(
        "$global$type (* $arg_name)$shape = ($global$type (*)$cast) _$arg_name;",
        {
            "arg_name": C.SymbolRef(name), 
            "shape": C.SymbolRef(shape_str),
            "cast": C.SymbolRef(shape_str),
            "type": C.SymbolRef(ctree.types.codegen_type(ctree.types.get_c_type_from_numpy_dtype(dtype)())),
            "global": C.SymbolRef("__global " if _global else "")
        }))

def aligned(shape, dtype, alignment=64, init=np.empty):
    size = np.prod(shape)
    nbytes = size * np.dtype(dtype).itemsize
    buf = init(nbytes + alignment, dtype=np.uint8)
    start_index = -buf.ctypes.data % alignment
    arr = buf[start_index:start_index + nbytes].view(dtype).reshape(shape)
    assert arr.ctypes.data % alignment == 0
    return arr

def empty(shape, dtype):
    return aligned(shape, dtype)

def zeros(shape, dtype):
    return aligned(shape, dtype, init=np.zeros)

def get_dependent_statements(statements, stmt):
    deps = collect_loads(stmt)
    dep_statements = []
    for statement in statements:
        for dep in deps:
            if dep in collect_stores(statement):
                dep_statements.append(statement)
                deps = deps.union(collect_loads(statement))
    return dep_statements

def gen_for(loopvar, start, end, body, pragma=""):
    return C.For(
        C.Assign(C.SymbolRef(loopvar, ctypes.c_int()), C.Constant(start)),
        C.Lt(C.SymbolRef(loopvar), C.Constant(end)),
        C.PostInc(C.SymbolRef(loopvar)),
        body,
        pragma
    )

def gen_index_expr(target, idxs):
    node = C.ArrayRef(target, idxs[0])
    for idx in idxs[1:]:
        node = C.ArrayRef(node, idx)
    return node

def gen_flat_index(idxs, shape):
    flat_idx = idxs[0]
    for i in range(len(idxs[1:])):
        flat_idx = ast.BinOp(
                ast.BinOp(
                    flat_idx,
                    ast.Mult(),
                    ast.Num(shape[i + 1])
                ),
                ast.Add(),
                idxs[i + 1]
            )
    return flat_idx

def print_ast(ast):
    print(astor.to_source(ast))

def find_loop(tree, loop_var):
    class Visitor(ast.NodeVisitor):
        def __init__(self):
            self.loop = None
        
        def visit_For(self, node):
            if node.init.left.name == loop_var:
                self.loop = node
            else:
                [self.visit(s) for s in node.body]
    visitor = Visitor()
    visitor.visit(tree)
    assert visitor.loop is not None, "Did not find loop with variable {}".format(loop_var)
    return visitor.loop


def get_ast(obj):
    indented_program_txt = inspect.getsource(obj)
    program_txt = textwrap.dedent(indented_program_txt)
    return ast.parse(program_txt)

def gen_loop_nest(body, loop_vars, loop_ranges):
    for var, _range in zip(loop_vars, loop_ranges):
        if sys.version_info >= (3, 5):
            body = [ast.For(ast.Name(var, ast.Store()),
                    ast.Call(ast.Name("range", ast.Load()), [ast.Num(_range)], []), body, [])]
        else:
            body = [ast.For(ast.Name(var, ast.Store()),
                    ast.Call(ast.Name("range", ast.Load()), [ast.Num(_range)], [], None, None), body, [])]
        # body = [ast.For(ast.Name(var, ast.Store()),
        #         ast.Call(ast.Name("range", ast.Load()), [ast.Num(_range[0]), ast.Num(_range[1])], []), body, [])]
    return body[0]


class InlineVariable(ast.NodeTransformer):
    def __init__(self, variable, value):
        self.variable = variable
        self.value = value

    def visit_Name(self, node):
        if node.id == self.variable:
            return self.value
        return node

def inline_variable(variable, value, ast):
    
    return InlineVariable(variable, value).visit(ast)


class AppendAllNames(ast.NodeTransformer):
    def __init__(self, suffix):
        self.suffix = suffix

    def visit_Name(self, node):
        node.id += self.suffix
        return node

class TileArrayRefs(ast.NodeTransformer):
    def __init__(self, idx):
        super().__init__()
        self.idx = idx
        self.tiled_buffers = {}

    def visit_BinaryOp(self, node):
        if isinstance(node.op, C.Op.ArrayRef) and \
            contains_name(node, self.idx):
            idx = 0
            curr_node = node
            while not contains_name(curr_node.right, self.idx):
                idx += 1
                curr_node = curr_node.left
            idx_expr = deepcopy(curr_node.right)
            while not isinstance(curr_node, ast.Name):
                curr_node = curr_node.left
            self.tiled_buffers[curr_node.id] = idx

            idx_expr = AppendAllNames("_inner").visit(idx_expr)
            return C.ArrayRef(node, idx_expr)
        node.left = self.visit(node.left)
        node.right = self.visit(node.right)
        return node

def tile_array_refs(idx, ast):
    visitor = TileArrayRefs(idx)
    ast = visitor.visit(ast)
    return ast, visitor.tiled_buffers

class ContainsName(ast.NodeVisitor):
    def __init__(self, sym):
        self.result = False
        self.sym = sym

    def visit_Name(self, node):
        if node.id == self.sym:
            self.result = True

def contains_name(ast, id):
    checker = ContainsName(id)
    checker.visit(ast)
    return checker.result

class ContainsSymbol(ast.NodeVisitor):
    def __init__(self, sym):
        self.result = False
        self.sym = sym

    def visit_SymbolRef(self, node):
        if node.name == self.sym:
            self.result = True

def contains_symbol(ast, symbol):
    checker = ContainsSymbol(symbol)
    checker.visit(ast)
    return checker.result

class SymbolCounter(ast.NodeVisitor):
    def __init__(self, sym):
        self.count = 0
        self.sym = sym

    def visit_SymbolRef(self, node):
        if node.name == self.sym:
            self.count += 1

def count_symbol_instances(ast, symbol):
    checker = SymbolCounter(symbol)
    checker.visit(ast)
    return checker.count




class ReplaceName(ast.NodeTransformer):
    def __init__(self, old, new):
        self.old = old
        self.new = new

    def visit_Name(self, node):
        if node.id == self.old.id:
            return self.new
        return node

def replace_name(old, new, ast):
    return ReplaceName(old, new).visit(ast)

class ReplaceSymbol(ast.NodeTransformer):
    def __init__(self, old, new):
        self.old = old
        self.new = new

    def visit(self, node):
        return super().visit(node)

    def visit_SymbolRef(self, node):
        if node.name == self.old:
            if isinstance(self.new, C.SymbolRef):
                self.new.type = node.type
            return self.new
        return node

    def visit_FunctionCall(self, node):
    
        new_args = [deepcopy(self.visit(a)) for a in node.args]
    
        return C.FunctionCall(node.func, new_args)



def replace_symbol(old, new, ast):
    return ReplaceSymbol(old, new).visit(ast)

__counter = 0
def generate_unique_function_name():
    global __counter
    __counter += 1
    return "generated_function_{}".format(__counter)


def has_nested_for(body):
    return len(body) == 1 and isinstance(body[0], (ast.For, latte.transformers.neuron.RangeDim))

import ctypes
from ctypes import c_int, c_float, byref, cdll

# mkl = cdll.LoadLibrary("libmkl_rt.so")
# 
# MKL_NOTRANS = 111
# MKL_TRANS = 112
# MKL_ORDER = 101  # Row major
# def sgemm(trans_A, trans_B, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc):
#     if trans_A:
#         trans_A = MKL_TRANS
#     else:
#         trans_A = MKL_NOTRANS
#     if trans_B:
#         trans_B = MKL_TRANS
#     else:
#         trans_B = MKL_NOTRANS
#     c_float_p = ctypes.POINTER(ctypes.c_float)
#     mkl.cblas_sgemm(c_int(MKL_ORDER), c_int(trans_A), c_int(trans_B), c_int(m), c_int(n), c_int(k), 
#             c_float(alpha), A.ctypes.data_as(c_float_p), c_int(lda),
#             B.ctypes.data_as(c_float_p), c_int(ldb), c_float(beta),
#             C.ctypes.data_as(c_float_p), c_int(ldc))

def prod(elts):
    result = 1
    for _range in elts:
        result *= (_range.stop - _range.start)
    return result

def flatten(x):
    if isinstance(x, collections.Iterable):
        return [a for i in x for a in flatten(i)]
    else:
        return [x]

def extend_or_append(_list, value):
    if isinstance(value, list):
        _list.extend(value)
    else:
        _list.append(value)


class LoadCollecter(ast.NodeVisitor):
    def __init__(self):
        self.seen = set()

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Load):
            self.seen.add(node.id)

def collect_loads(ast):
    visitor = LoadCollecter()
    visitor.visit(ast)
    return visitor.seen

class StoreCollecter(ast.NodeVisitor):
    def __init__(self):
        self.seen = set()

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Store):
            self.seen.add(node.id)

def collect_stores(ast):
    visitor = StoreCollecter()
    visitor.visit(ast)
    return visitor.seen

def untile(buffer, dim):
    shape = buffer.shape
    untiled_shape = list(shape[:-1])
    untiled_shape[dim] *= shape[-1]
    untiled = np.zeros(untiled_shape, dtype=buffer.dtype)
    for index in np.ndindex(shape):
        untiled_index = list(index[:-1])
        untiled_index[dim] = untiled_index[dim] * shape[-1] + index[-1]
        untiled[tuple(untiled_index)] = buffer[index]
    return untiled

def tile(buffer, dim):
    tiled_shape = list(buffer.shape)
    factor = latte.config.SIMDWIDTH
    if tiled_shape[dim] < factor:
        factor = tiled_shape[dim]
    elif tiled_shape[dim] % latte.config.SIMDWIDTH != 0:
        filters_pad = factor - (tiled_shape[dim] % factor)
        buffer = np.lib.pad(buffer, ((0,0), (0, filters_pad), (0,0), (0,0)), 'constant')
   
    shape = buffer.shape
    tiled_shape = list(buffer.shape)
 
    tiled_shape[dim] //= factor
    tiled_shape.append(factor)
    tiled = np.zeros(tiled_shape, dtype=buffer.dtype)
    for untiled_index in np.ndindex(shape):
        tiled_index = list(untiled_index)
        dim_index = tiled_index[dim]
        tiled_index[dim] = dim_index // factor
        tiled_index.append(dim_index % factor)
        tiled[tuple(tiled_index)] = buffer[untiled_index]
    return tiled

def convert_6d_4d(arr):
    shape = arr.shape
    converted_shape = (shape[0] * shape[5], shape[1] * shape[4], shape[2], shape[3])
    arr_converted = np.zeros(converted_shape)
    for ofm in range(shape[0]):
        for ifm in range(shape[1]):
            for y in range(shape[2]):
                for x in range(shape[3]):
                    for v2 in range(shape[4]):
                        for v in range(shape[5]):
                            arr_converted[ofm * shape[5] + v, ifm * shape[4] + v2, y, x] = \
                                arr[ofm, ifm, y, x, v, v2]
    return arr_converted

def convert_3d_2d(arr):
    shape = arr.shape
    converted_shape = (shape[0] * shape[2], shape[1])
    arr_converted = np.zeros(converted_shape)
    for x in range(shape[0]):
        for y in range(shape[1]):
            for v in range(shape[2]):
                arr_converted[x * shape[2] + v, y] = \
                    arr[x, y, v]
    return arr_converted

def convert_6d_4d_tr(arr):
    shape = arr.shape
    arr_converted = np.zeros_like(arr)
    arr_reshaped = arr.reshape(shape[0] // 8, shape[1] // 8, shape[2], shape[3], 8, 8)
    for ofm in range(shape[0] // 8):
        for ifm in range(shape[1] // 8):
            for y in range(shape[2]):
                for x in range(shape[3]):
                    for v2 in range(8):
                        for v in range(8):
                            arr_converted[ofm * 8 + v, ifm * 8 + v2, y, x] = \
                                arr_reshaped[ofm, ifm, y, x, v, v2]
    return arr_converted
    
def convert_5d_4d(arr):
    shape = arr.shape
    arr_converted = np.zeros((shape[0], shape[1] * shape[4], shape[2], shape[3]))
    for n in range(shape[0]):
        for ifm in range(shape[1]):
            for y in range(shape[2]):
                for x in range(shape[3]):
                    for v in range(shape[4]):
                        arr_converted[n, ifm * shape[4] + v, y, x] = arr[n, ifm, y, x, v]
    return arr_converted
    
def convert_6d_5d(arr):
    shape = arr.shape
    arr_converted = np.zeros((shape[0], shape[1] * shape[5],) + shape[2:-1])
    for n in range(shape[0]):
        for ifm in range(shape[1]):
            for y in range(shape[2]):
                for x in range(shape[3]):
                    for z in range(shape[4]):
                        for v in range(shape[5]):
                            arr_converted[n, ifm * shape[5] + v, y, x, z] = arr[n, ifm, y, x, z, v]
    return arr_converted

def convert_4d_2d(arr):
    shape = arr.shape
    arr_converted = np.zeros((shape[0] * shape[2], shape[1] * shape[3]))
    for n in range(shape[0]):
        for i in range(shape[1]):
            for v in range(shape[2]):
                for v2 in range(shape[3]):
                    arr_converted[n * shape[2] + v, i * shape[3] + v2] = arr[n, i, v, v2]
    return arr_converted

def interleave_lists(list1, list2):
    return [val for pair in zip(list1, list2) for val in pair]

class ClampInputIndex(ast.NodeTransformer):
    def __init__(self, loop_var, gen_clamp):
        self.loop_var = loop_var
        self.gen_clamp = gen_clamp

    def visit_BinaryOp(self, node):
        if isinstance(node.op, C.Op.ArrayRef):
            curr_node = node
            while not isinstance(curr_node, C.SymbolRef):
                curr_node = curr_node.left
            if curr_node.name.endswith("inputs"):
                curr_node = node
                while 'right' in curr_node._fields and not contains_symbol(curr_node.right, self.loop_var):
                    curr_node = curr_node.left
                if 'right' in curr_node._fields:
                    curr_node.right = self.gen_clamp(curr_node.right)
                    return node
        node.left = self.visit(node.left)
        node.right = self.visit(node.right)
        return node

# class Unpack(ast.NodeTransformer):
#     def __init__(self):
#         super().__init__()
#         self.counter = 0
#         self.curr_tmp_var = None
# 
#     def _gen_next_tmp_var(self):
#         self.counter += 1
#         self.curr_tmp_var = "v{}".format(self.counter)
#         return self.curr_tmp_var
# 
#     def visit_FunctionDef(self, node):
#         new_body = []
#         for statement in node.body:
#             result = self.visit(statement)
#             extend_or_append(new_body, result)
#         node.body = new_body
#         return node
# 
#     def visit_Assign(self, node):
#         block = []
#         if not isinstance(node.value, ast.Name):
#             result = self.visit(node.value)
#             extend_or_append(block, result[:-1])
#             node.value = result[-1]
#         if len(block) > 0:
#             return block + [node]
# 
#     def visit_BinOp(self, node):
#         block = []
#         if not isinstance(node.left, ast.Name):
#             var = self._gen_next_tmp_var()
#             result = self.visit(node.left)
#             node.left = ast.Name(var, ast.Load())
#             extend_or_append(block, result)
#             # block[-1] = ast.Assign([C.SymbolRef(var, ctree.c.HalideFunc())], block[-1])
#         if not isinstance(node.right, ast.Name):
#             var = self._gen_next_tmp_var()
#             result = self.visit(node.right)
#             extend_or_append(block, result)
#             node.right = ast.Name(var, ast.Load())
#             # block[-1] = ast.Assign([C.SymbolRef(var, ctree.c.HalideFunc())], block[-1])
#         if len(block) > 0:
#             return block + [node]
# 
#     def visit_Call(self, node):
#         block = []
#         new_args = []
#         for arg in node.args:
#             if not isinstance(arg, ast.Name):
#                 var = self._gen_next_tmp_var()
#                 result = self.visit(arg)
#                 extend_or_append(block, result)
#                 # block[-1] = ast.Assign([C.SymbolRef(var, ctree.c.HalideFunc())], block[-1])
#                 new_args.append(ast.Name(var, ast.Load()))
#         node.args = new_args
#         if len(block) > 0:
#             return block + [node]
