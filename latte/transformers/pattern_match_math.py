import ast
import ctypes
import ctree.c.nodes as C

class PatternMatchMath(ast.NodeTransformer):
    def visit_FunctionCall(self, node):
        if isinstance(node.func, C.SymbolRef):
            if node.func.name == "rand":
                return C.Div(node, C.Cast(ctypes.c_float(), C.SymbolRef("RAND_MAX")))
        return node
