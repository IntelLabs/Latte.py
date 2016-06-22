import ast
import ctypes
import ctree
import copy
import ctree.np
import ctree.c.nodes as C

class BasicTypeInference(ast.NodeTransformer):
    def __init__(self):
        self.seen = {}

    def visit_SymbolRef(self, node):
        if node.type is not None:
            self.seen[node.name] = node.type
        return node

    def visit(self, node):
        if hasattr(node, 'body'):
            curr = copy.deepcopy(self.seen)
        node = super().visit(node)
        if hasattr(node, 'body'):
            self.seen = curr
        return node

    def _get_type(self, node):
        if isinstance(node, C.SymbolRef):
            if node.name == "INFINITY":
                return ctypes.c_float()
            elif node.name in self.seen:
                return self.seen[node.name]
        elif isinstance(node, C.UnaryOp):
            return self._get_type(node.arg)
        elif isinstance(node, C.Constant):
            if isinstance(node.value, int):
                # promote all longs to int
                return ctypes.c_int()
            return ctree.types.get_ctype(node.value)
        elif isinstance(node, C.BinaryOp):
            if isinstance(node.op, C.Op.ArrayRef):
                while not isinstance(node, C.SymbolRef):
                    node = node.left
                pointer_type = self._get_type(node)
                return ctree.types.get_c_type_from_numpy_dtype(pointer_type._dtype_)()
            else:
                left = self._get_type(node.left)
                right = self._get_type(node.right)
                return ctree.types.get_common_ctype([left, right])
        elif isinstance(node, C.FunctionCall):
            if node.func.name in ["MAX", "MIN", "max", "min", "floor"]:
                return ctree.types.get_common_ctype([self._get_type(a) for a in node.args])
        raise NotImplementedError(ast.dump(node))
    
    def visit_FunctionCall(self, node):
        if node.func.name in ["max", "min"] and isinstance(self._get_type(node), ctypes.c_float):
            # convert to fmax/fmin
            node.func.name = "f" + node.func.name
        else:
            node.args = [self.visit(a) for a in node.args]
        return node

    def visit_BinaryOp(self, node):
        node.left = self.visit(node.left)
        if isinstance(node.op, C.Op.Assign) and isinstance(node.left, C.SymbolRef) and node.left.name not in self.seen:
            node.left.type = self._get_type(node.right)
            self.seen[node.left.name] = node.left.type
        node.right = self.visit(node.right)
        return node

def type_infer(ast):
    try:
        return BasicTypeInference().visit(ast)
    except NotImplementedError as e:
        print("AST that caused exception during type inference")
        print(ast)
        raise e
