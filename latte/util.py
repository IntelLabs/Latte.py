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

class PatternMatchGemm(ast.NodeTransformer):
    def visit_For(self, node):
        first = node
        i = first.target.id
        i_len = first.iter.args[0]
        if not has_nested_for(node.body):
            return node
        second = node.body[0]
        j = second.target.id
        j_len = second.iter.args[0]
        if not has_nested_for(second.body):
            return node
        third = second.body[0]
        k = third.target.id
        k_len = third.iter.args[0]
        if isinstance(third.body[0], ast.AugAssign) and \
                isinstance(third.body[0].op, ast.Add) and \
                isinstance(third.body[0].value, ast.BinOp) and \
                isinstance(third.body[0].value.op, ast.Mult):
            A = third.body[0].value.left
            B = third.body[0].value.right
            C = third.body[0].target
            A_idx = [idx.id for idx in A.slice.value.elts]
            B_idx = [idx.id for idx in B.slice.value.elts]
            C_idx = [idx.id for idx in C.slice.value.elts]
            if C_idx == [i, j]:
                ldc = j_len
                if A_idx == [k, i]:
                    trans_A = ast.Name("True", ast.Load())
                    lda = i_len
                elif A_idx == [i, k]:
                    trans_A = ast.Name("False", ast.Load())
                    lda = k_len
                else:
                    raise NotImplementedError()
                if B_idx == [j, k]:
                    trans_B = ast.Name("True", ast.Load())
                    ldb = k_len
                elif B_idx == [k, j]:
                    trans_B = ast.Name("False", ast.Load())
                    ldb = j_len
                else:
                    raise NotImplementedError()
                gemm_call = ast.Call(ast.Name("sgemm", ast.Load()),
                        [trans_A, trans_B, i_len, j_len, k_len, ast.Num(1.0), A.value,
                            lda, B.value, ldb, ast.Num(1.0), C.value, ldc], [])
                return ast.Expr(gemm_call)
            elif C_idx == [i, k]:
                ldc = k_len
                if A_idx == [j, i]:
                    trans_A = ast.Name("True", ast.Load())
                    lda = i_len
                elif A_idx == [i, j]:
                    trans_A = ast.Name("False", ast.Load())
                    lda = j_len
                else:
                    raise NotImplementedError()
                if B_idx == [k, j]:
                    trans_B = ast.Name("True", ast.Load())
                    ldb = j_len
                elif B_idx == [j, k]:
                    trans_B = ast.Name("False", ast.Load())
                    ldb = k_len
                else:
                    raise NotImplementedError()
                gemm_call = ast.Call(ast.Name("sgemm", ast.Load()),
                        [trans_A, trans_B, i_len, k_len, j_len, ast.Num(1.0), A.value,
                            lda, B.value, ldb, ast.Num(1.0), C.value, ldc], [])
                return ast.Expr(gemm_call)
            elif C_idx == [j, k]:
                ldc = k_len
                if A_idx == [i, j]:
                    trans_A = ast.Name("True", ast.Load())
                    lda = j_len
                elif A_idx == [j, i]:
                    trans_A = ast.Name("False", ast.Load())
                    lda = i_len
                else:
                    raise NotImplementedError()
                if B_idx == [k, i]:
                    trans_B = ast.Name("True", ast.Load())
                    ldb = i_len
                elif B_idx == [i, k]:
                    trans_B = ast.Name("False", ast.Load())
                    ldb = k_len
                else:
                    raise NotImplementedError()
                gemm_call = ast.Call(ast.Name("sgemm", ast.Load()),
                        [trans_A, trans_B, j_len, k_len, i_len, ast.Num(1.0), A.value,
                            lda, B.value, ldb, ast.Num(1.0), C.value, ldc], [])
                return ast.Expr(gemm_call)
            else:
                raise NotImplementedError(C_idx, [i, j, k])
            # raise NotImplementedError(astor.to_source(node))
        return node

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
