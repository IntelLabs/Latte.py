import ast
import ctree.c.nodes as C


class SimpleConstantPropogation(ast.NodeTransformer):
    def __init__(self):
        self.seen = {}
    
    def visit(self, node):
        node = super().visit(node)
        if hasattr(node, 'body'):
            node.body = [s for s in node.body if s is not None]
        return node

    def _get_value(self, node):
        if isinstance(node, C.Constant):
            return node.value
        elif isinstance(node, C.SymbolRef) and node.name in self.seen:
            return self.seen[node.name]
        return None

    def visit_For(self, node):
        # Skip loopvar inits
        node.body = [self.visit(s) for s in node.body]
        return node

    def visit_SymbolRef(self, node):
        if node.name in self.seen:
            return C.Constant(self.seen[node.name])
        return node

    def visit_BinaryOp(self, node):
        node.left = self.visit(node.left)
        node.right = self.visit(node.right)
        if isinstance(node.op, C.Op.Assign) and isinstance(node.left, C.SymbolRef):
            value = self._get_value(node.right)
            if value is not None:
                self.seen[node.left.name] = value
                return None
        elif isinstance(node.op, C.Op.Div):
            left = self._get_value(node.left)
            right = self._get_value(node.right)
            if left is not None and right is not None:
                if isinstance(left, int) and isinstance(right, int):
                    return C.Constant(left // right)
                else:
                    return C.Constant(left / right)
        elif isinstance(node.op, C.Op.Mul):
            left = self._get_value(node.left)
            right = self._get_value(node.right)
            if left is not None and right is not None:
                return C.Constant(left * right)
            elif left == 0 or right == 0:
                return C.Constant(0)
        elif isinstance(node.op, C.Op.Mod):
            left = self._get_value(node.left)
            right = self._get_value(node.right)
            if left is not None and right is not None:
                return C.Constant(left % right)
        elif isinstance(node.op, C.Op.Add):
            left = self._get_value(node.left)
            right = self._get_value(node.right)
            if left is not None and right is not None:
                return C.Constant(left + right)
            elif left == 0:
                return node.right
            elif right == 0:
                return node.left
        elif isinstance(node.op, C.Op.Sub):
            left = self._get_value(node.left)
            right = self._get_value(node.right)
            if left is not None and right is not None:
                return C.Constant(left - right)
        return node

def propogate_constants(ast):
    return SimpleConstantPropogation().visit(ast)
