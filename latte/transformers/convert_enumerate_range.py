import ast
import ctree.c.nodes as C
import latte.util as util
import ctypes
import latte.core

class ConvertEnumerateRange(ast.NodeTransformer):
    """
    converts for ... in enumerate(range(...)) into a valid C for loop
    """
    def __init__(self):
        super().__init__()
        self.blocked_loops = []
        self.tiled_buffers = {}

    def visit(self, node):
        node = super().visit(node)
        if hasattr(node, 'body'):
            node.body = util.flatten(node.body)
        return node

    def visit_For(self, node):
        if isinstance(node.iter, ast.Call) and node.iter.func.id == "range" and \
            node.target.id == "_neuron_index_1":
            new_body = []
            for statement in node.body:
                result = self.visit(statement)
                if len(self.blocked_loops) > 0:
                    if len(self.blocked_loops) > 1:
                        raise NotImplementedError()
                    new_body.append(self.blocked_loops[0])
                    self.blocked_loops[0].body = [result]
                    self.blocked_loops = []
                else:
                    new_body.append(result)
            node.body = new_body
            return node
        node.body = util.flatten([self.visit(s) for s in node.body])

        if isinstance(node.iter, ast.Call) and node.iter.func.id == "enumerate":
            assert node.iter.args[0].func.id == "range"
            range_args = node.iter.args[0].args
            if len(range_args) == 1:
                old = node.target.elts[0]
                new = node.target.elts[1]
                node.body = [util.replace_name(old, new, s) for s in node.body]
                end = node.iter.args[0].args[0]
                if isinstance(end, ast.Num):
                    self.blocked_loops.append(
                        C.For(
                            C.Assign(C.SymbolRef(node.target.elts[1].id + "_tile", ctypes.c_int()), C.Constant(0)),
                            C.Lt(C.SymbolRef(node.target.elts[1].id + "_tile"), C.Constant(end.n // latte.core.TILE_SIZE)),
                            # C.AddAssign(C.SymbolRef(node.target.elts[1].id + "_tile"), C.SymbolRef("TILE_SIZE")),
                            C.PostInc(C.SymbolRef(node.target.elts[1].id + "_tile")),
                            [])
                    )
                    if end.n % latte.core.TILE_SIZE == 0:
                        # end = C.Add(C.SymbolRef(node.target.elts[1].id + "_tile"), 
                        #             C.SymbolRef("TILE_SIZE"))
                        end = C.SymbolRef("TILE_SIZE")
                    else:
                        raise NotImplementedError()
                    # init = C.Assign(
                    #     C.SymbolRef(node.target.elts[1].id, ctypes.c_int()),
                    #     C.SymbolRef(node.target.elts[1].id + "_tile"))
                    init = C.Assign(
                        C.SymbolRef(node.target.elts[1].id, ctypes.c_int()),
                        C.Constant(0))
                    new_body = []
                    for statement in node.body:
                        result, tiled_buffers = util.tile_array_refs(node.target.elts[1].id, statement)
                        new_body.append(result)
                        self.tiled_buffers = dict(self.tiled_buffers, **tiled_buffers)
                    node.body = new_body
                elif isinstance(end, Ast.Name):
                    end = C.SymbolRef(end.id)
                    init = C.Assign(
                        C.SymbolRef(node.target.elts[1].id, ctypes.c_int()),
                        C.Constant(0))
                else:
                    raise NotImplementedError()
                return C.For(
                        init,
                        C.Lt(C.SymbolRef(node.target.elts[1].id), end),
                        C.PostInc(C.SymbolRef(new.id)),
                        node.body
                    )
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
    visitor = ConvertEnumerateRange()
    ast = visitor.visit(ast)
    return ast, visitor.tiled_buffers
