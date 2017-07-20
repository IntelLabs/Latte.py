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
import ast
import latte.util as util
import latte

class PatternMatchGemm(ast.NodeTransformer):
    def visit_For(self, node):
        first = node
        i = first.target.id
        i_len = first.iter.args[0]
        if not util.has_nested_for(node.body):
            return node
        second = node.body[0]
        j = second.target.id
        j_len = second.iter.args[0]
        if not util.has_nested_for(second.body):
            return node
        third = second.body[0]
        if isinstance(third, latte.transformers.neuron.RangeDim):
            mapping_func = util.get_ast(third.mapping).body[0]
            ndim = len(mapping_func.args.args)
            dim = third.child_for.iter.args[1].n
            k_len = ast.Num(len(third.mapping(*[1 for _ in range(ndim)])[dim]))
            third = third.child_for
        else:
            k_len = third.iter.args[0]
        k = third.target.id
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
                A.value.id = "_" + A.value.id
                B.value.id = "_" + B.value.id
                C.value.id = "_" + C.value.id
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
                A.value.id = "_" + A.value.id
                B.value.id = "_" + B.value.id
                C.value.id = "_" + C.value.id
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
                A.value.id = "_" + A.value.id
                B.value.id = "_" + B.value.id
                C.value.id = "_" + C.value.id
                gemm_call = ast.Call(ast.Name("sgemm", ast.Load()),
                        [trans_A, trans_B, j_len, k_len, i_len, ast.Num(1.0), A.value,
                            lda, B.value, ldb, ast.Num(1.0), C.value, ldc], [])
                return ast.Expr(gemm_call)
            else:
                raise NotImplementedError(C_idx, [i, j, k])
            # raise NotImplementedError(astor.to_source(node))
        return node

def pattern_match_gemm(ast):
    return PatternMatchGemm().visit(ast)
