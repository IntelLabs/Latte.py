import ast
import latte.util as util

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

def pattern_match_gemm(ast):
    return PatternMatchGemm().visit(ast)
