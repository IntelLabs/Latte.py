import ast

class FlattenSubscripts(ast.NodeTransformer):
    """
    Flattens indexing expressions buffer[x, y, z] into a linear index like
    buffer[(z * y_len + y) * x_len + x]
    """
    def __init__(self, buffers):
        self.buffers = buffers

    def visit_Subscript(self, node):
        assert node.value.id in self.buffers
        shape = self.buffers[node.value.id].shape
        if isinstance(node.slice.value, ast.Tuple):
            idxs = node.slice.value.elts
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
            node.slice.value = flat_idx
        return node

def flatten_subscripts(ast, buffers):
    return FlattenSubscripts(buffers).visit(ast)
