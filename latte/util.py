import inspect
import textwrap
import ast
import astor
import collections
import ctree.c.nodes as C
import numpy as np

def aligned(a, alignment=64):
    if (a.ctypes.data % alignment) == 0:
        return a

    extra = alignment // a.itemsize
    buf = np.empty(a.size + extra, dtype=a.dtype)
    ofs = (-buf.ctypes.data % alignment) // a.itemsize
    aa = buf[ofs:ofs+a.size].reshape(a.shape)
    np.copyto(aa, a)
    assert (aa.ctypes.data % alignment) == 0
    return aa

def empty(shape, dtype):
    return aligned(np.empty(shape, dtype))

def zeros(shape, dtype):
    return aligned(np.zeros(shape, dtype))

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

def get_ast(obj):
    indented_program_txt = inspect.getsource(obj)
    program_txt = textwrap.dedent(indented_program_txt)
    return ast.parse(program_txt)

def gen_loop_nest(body, loop_vars, loop_ranges):
    for var, _range in zip(loop_vars, loop_ranges):
        body = [ast.For(ast.Name(var, ast.Store()),
                ast.Call(ast.Name("range", ast.Load()), [ast.Num(_range)], []), body, [])]
    return body[0]


class InlineVariable(ast.NodeTransformer):
    def __init__(self, variable, value):
        self.variable = variable
        self.value = value

    def visit_Name(self, node):
        if node.id == self.variable:
            return ast.parse(str(self.value)).body[0].value
        return node

def inline_variable(variable, value, ast):
    return InlineVariable(variable, value).visit(ast)


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
            while not isinstance(curr_node.right, ast.Name) or \
                    curr_node.right.id != self.idx:
                idx += 1
                curr_node = curr_node.left
            while not isinstance(curr_node, ast.Name):
                curr_node = curr_node.left
            self.tiled_buffers[curr_node.id] = idx

            node = replace_name(ast.Name(self.idx, ast.Load()), ast.Name(self.idx+"_tile", ast.Load()), node)
            return C.ArrayRef(node, C.SymbolRef(self.idx))
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

class ReplaceName(ast.NodeTransformer):
    def __init__(self, old, new):
        self.old = old
        self.new = new

    def visit_Name(self, node):
        if node.id == self.old.id:
            node.id = self.new.id
        return node

def replace_name(old, new, ast):
    return ReplaceName(old, new).visit(ast)

__counter = 0
def generate_unique_function_name():
    global __counter
    __counter += 1
    return "generated_function_{}".format(__counter)


def has_nested_for(body):
    return len(body) == 1 and isinstance(body[0], ast.For)

import ctypes
from ctypes import c_int, c_float, byref, cdll

mkl = cdll.LoadLibrary("libmkl_rt.so")

MKL_NOTRANS = 111
MKL_TRANS = 112
MKL_ORDER = 101  # Row major
def sgemm(trans_A, trans_B, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc):
    if trans_A:
        trans_A = MKL_TRANS
    else:
        trans_A = MKL_NOTRANS
    if trans_B:
        trans_B = MKL_TRANS
    else:
        trans_B = MKL_NOTRANS
    c_float_p = ctypes.POINTER(ctypes.c_float)
    mkl.cblas_sgemm(c_int(MKL_ORDER), c_int(trans_A), c_int(trans_B), c_int(m), c_int(n), c_int(k), 
            c_float(alpha), A.ctypes.data_as(c_float_p), c_int(lda),
            B.ctypes.data_as(c_float_p), c_int(ldb), c_float(beta),
            C.ctypes.data_as(c_float_p), c_int(ldc))

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
