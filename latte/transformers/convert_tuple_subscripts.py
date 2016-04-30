import ast
import ctree.c.nodes as C
import latte.util as util

class ConvertTupleSubscripts(ast.NodeTransformer):
    """
    Converts indexing expressions buffer[x, y, z] into buffer[x][y][z]
    """
    def visit_Subscript(self, node):
        if isinstance(node.slice.value, ast.Tuple):
            idxs = node.slice.value.elts
            new_node = C.ArrayRef(node.value, idxs[0])
            for idx in idxs[1:]:
                new_node = C.ArrayRef(new_node, idx)
            return new_node
        return node

def convert_tuple_subscripts(ast):
    return ConvertTupleSubscripts().visit(ast)
