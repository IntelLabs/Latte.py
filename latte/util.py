import inspect
import textwrap
import ast
import astor
import collections

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
