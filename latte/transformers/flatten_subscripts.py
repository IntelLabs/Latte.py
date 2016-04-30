import ast
import latte.util as util

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
            node.slice.value = util.gen_flat_index(idxs, shape)
        return node

def flatten_subscripts(ast, buffers):
    return FlattenSubscripts(buffers).visit(ast)
