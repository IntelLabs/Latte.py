import latte.util as util
import ast
import ctree.c.nodes as C
from copy import deepcopy

class UnrollStatements(ast.NodeTransformer):
    def __init__(self, target_var, factor):
        super().__init__()
        self.target_var = target_var
        self.factor = factor
        self.unrolled_vars = set()

    def visit(self, node):
        """
        Support replacing nodes with a list of nodes by flattening `body`
        fields.
        """
        node = super().visit(node)
        if hasattr(node, 'body'):
            node.body = util.flatten(node.body)
        return node

    def visit_BinaryOp(self, node):
        if isinstance(node.op, C.Op.Assign):
            check = [util.contains_symbol(node.right, var) for var in [*self.unrolled_vars] + [self.target_var]]
            if any(check):
                body = []
                self.unrolled_vars.add(node.left.name)
                for i in range(self.factor):
                    stmt = deepcopy(node)
                    for var in self.unrolled_vars:
                        stmt = util.replace_symbol(var, C.SymbolRef(var + str(i)), stmt)
                    body.append(util.replace_symbol(self.target_var, C.Add(C.SymbolRef(self.target_var), C.Constant(i)), stmt))
                return body
        return node

    def visit_AugAssign(self, node):
        check = [util.contains_symbol(node.value, var) for var in [*self.unrolled_vars] + [self.target_var]]
        if any(check):
            body = []
            if isinstance(node.target, C.SymbolRef):
                self.unrolled_vars.add(self._get_name(node.target.name))
                for i in range(self.factor):
                    stmt = deepcopy(node)
                    for var in self.unrolled_vars:
                        stmt = util.replace_symbol(var, C.SymbolRef(var + str(i)), stmt)
                    body.append(util.replace_symbol(self.target_var, C.Add(C.SymbolRef(self.target_var), C.Constant(i)), stmt))
                return body
            elif isinstance(node.target, C.BinaryOp) and isinstance(node.target.op, C.Op.ArrayRef):
                assert False
                for i in range(self.factor):
                    stmt = deepcopy(node)
                    for var in self.unrolled_vars:
                        stmt = util.replace_symbol(var, C.SymbolRef(var + str(i)), stmt)
                    body.append(util.replace_symbol(self.target_var, C.Add(C.SymbolRef(self.target_var), C.Constant(i)), stmt))
                return body
            else:
                raise NotImplementedError()
        return node

    def visit_FunctionCall(self, node):
        check = [util.contains_symbol(node, var) for var in [*self.unrolled_vars] + [self.target_var]]
        if "store" in node.func.name and "_mm" in node.func.name and any(check):
            body = []
            for i in range(self.factor):
                stmt = deepcopy(node)
                stmt = util.replace_symbol(self.target_var, C.Add(C.SymbolRef(self.target_var), C.Constant(i)), stmt)
                for var in self.unrolled_vars:
                    stmt = util.replace_symbol(var, C.SymbolRef(var + str(i)), stmt)
                body.append(stmt)
            return body
        return node

class LoopUnroller(ast.NodeTransformer):
    def __init__(self, target_var, factor):
        super().__init__()
        self.target_var = target_var
        self.factor = factor

    def visit_For(self, node):
        node.body = [self.visit(s) for s in node.body]
        if node.init.left.name == self.target_var:
            node.incr = C.AddAssign(C.SymbolRef(self.target_var), C.Constant(self.factor))
            visitor = UnrollStatements(self.target_var, self.factor)
            node.body = util.flatten([visitor.visit(s) for s in node.body])
        return node


def unroll_inner_neuron_loop(ast, target_var, factor):
    return LoopUnroller(target_var, factor).visit(ast)
