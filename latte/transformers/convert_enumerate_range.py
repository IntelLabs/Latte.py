import ast
import ctree.c.nodes as C
import latte.util as util

class ConvertEnumerateRange(ast.NodeTransformer):
    """
    converts for ... in enumerate(range(...)) into a valid C for loop
    """
    def visit(self, node):
        node = super().visit(node)
        if hasattr(node, 'body'):
            node.body = util.flatten(node.body)
        return node

    def visit_For(self, node):
        node.body = util.flatten([self.visit(s) for s in node.body])
        if isinstance(node.iter, ast.Call) and node.iter.func.id == "enumerate":
            assert node.iter.args[0].func.id == "range"
            range_args = node.iter.args[0].args
            if len(range_args) == 1:
                init = [C.Assign(
                    C.SymbolRef(node.target.elts[0].id),
                    C.Constant(0)),
                    C.Assign(
                        C.SymbolRef(node.target.elts[1].id),
                        C.Constant(0))]
                end = node.iter.args[0].args[0]
            elif len(range_args) == 2:
                start = node.iter.args[0].args[0] 
                if isinstance(start, ast.Name):
                    start = C.SymbolRef(start.id)
                elif isinstance(start, ast.Num):
                    start = C.Constant(start.n)
                else:
                    raise NotImplementedError
                init = [C.Assign(
                    C.SymbolRef(node.target.elts[0].id),
                    C.Constant(0)),
                    C.Assign(
                        C.SymbolRef(node.target.elts[1].id),
                        C.Constant(start))]
                end = node.iter.args[0].args[1]
            else:
                raise NotImplementedError

            if isinstance(end, ast.Name):
                end = C.SymbolRef(end.id)
            elif isinstance(end, ast.Num):
                end = C.Constant(end.n)
            else:
                raise NotImplementedError
            pre_stmts = [C.SymbolRef(var.id, ctypes.c_int()) for var in node.target.elts]
            return pre_stmts + [C.For(
                    init,
                    C.Lt(C.SymbolRef(node.target.elts[1].id), end),
                    [C.PostInc(C.SymbolRef(var.id)) for var in node.target.elts],
                    node.body
                )]
        return node

def convert_enumerate_ranges(ast):
    return ConvertEnumerateRange().visit(ast)
