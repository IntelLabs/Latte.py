import ast
import ctree.c.nodes as C

class ConvertSGEMMCalls(ast.NodeTransformer):
    """
    Converts sgemm(...) calls into a valid cblas_sgemm call for MKL by
    prepending 101 for Row-Major and converting trans_A, trans_B in to the
    proper constant.
    """
    def visit_Call(self, node):
        if node.func.id == "sgemm":
            node.func.id = "cblas_sgemm"
            for i in range(2):
                node.args[i] = C.Constant(112) if node.args[i].id == "True" else C.Constant(111) 
            node.args.insert(0, C.Constant(101))
        return node

def convert_sgemm_calls(ast):
    return ConvertSGEMMCalls().visit(ast)
