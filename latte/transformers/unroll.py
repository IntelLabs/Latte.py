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
        #     new_body = []
        #     for s in node.body:
        #         s = self.visit(s)
        #         if len(new_body) > 1 and isinstance(new_body[-1], list) and isinstance(s, list):
        #             print("-----------")
        #             print([str(x) for x in s])
        #             print(isinstance(new_body[-1][0], C.BinaryOp))
        #             print([isinstance(x, C.BinaryOp) and \
        #                      isinstance(x.op, C.Op.Assign) and \
        #                      x.left.codegen() == new_body[-1][0].left.codegen() 
        #                      for x in new_body[-1]])
        #             print(not all([isinstance(x, C.BinaryOp) and \
        #                          isinstance(x.op, C.Op.Assign) and \
        #                          x.left.codegen() == s[0].left.codegen() 
        #                          for x in s]))
        #             print("-----------")
        #         if len(new_body) > 1 and isinstance(new_body[-1], list) and \
        #                 isinstance(s, list) and \
        #                 isinstance(new_body[-1][0], C.BinaryOp) and \
        #                 all([isinstance(x, C.BinaryOp) and \
        #                      isinstance(x.op, C.Op.Assign) and \
        #                      x.left.codegen() == new_body[-1][0].left.codegen() 
        #                      for x in new_body[-1]]):
        #             new_body[-1] = util.interleave_lists(new_body[-1], s)
        #         else:
        #             new_body.append(s)
        #     node.body = util.flatten(new_body)
        #     # node.body = util.flatten(node.body)
        # else: 
        #     node = super().visit(node)
        return node

    def visit_BinaryOp(self, node):
        if isinstance(node.op, C.Op.Assign):
            check = [util.contains_symbol(node.right, var) for var in self.unrolled_vars + [self.target_var]]
            if any(check):
                body = []
                if node.left.type is not None:
                    self.unrolled_vars.add(node.left.name)
                for i in range(self.factor):
                    stmt = deepcopy(node)
                    for var in self.unrolled_vars:
                        stmt = util.replace_symbol(var, C.SymbolRef(var + "_" + str(i)), stmt)
                    body.append(util.replace_symbol(self.target_var, C.Add(C.SymbolRef(self.target_var), C.Constant(i)), stmt))
                return body
        return node

    def visit_AugAssign(self, node):
        check = [util.contains_symbol(node.value, var) for var in self.unrolled_vars + [self.target_var]]
        if any(check):
            body = []
            if isinstance(node.target, C.SymbolRef):
                self.unrolled_vars.add(self._get_name(node.target.name))
                for i in range(self.factor):
                    stmt = deepcopy(node)
                    for var in self.unrolled_vars:
                        stmt = util.replace_symbol(var, C.SymbolRef(var + "_" + str(i)), stmt)
                    body.append(util.replace_symbol(self.target_var, C.Add(C.SymbolRef(self.target_var), C.Constant(i)), stmt))
                return body
            elif isinstance(node.target, C.BinaryOp) and isinstance(node.target.op, C.Op.ArrayRef):
                assert False
                for i in range(self.factor):
                    stmt = deepcopy(node)
                    for var in self.unrolled_vars:
                        stmt = util.replace_symbol(var, C.SymbolRef(var + "_" + str(i)), stmt)
                    body.append(util.replace_symbol(self.target_var, C.Add(C.SymbolRef(self.target_var), C.Constant(i)), stmt))
                return body
            else:
                raise NotImplementedError()
        return node

    def visit_FunctionCall(self, node):
        check = [util.contains_symbol(node, var) for var in self.unrolled_vars + [self.target_var]]
        if "store" in node.func.name and "_mm" in node.func.name and any(check):
            body = []
            for i in range(self.factor):
                stmt = deepcopy(node)
                for var in self.unrolled_vars:
                    stmt = util.replace_symbol(var, C.SymbolRef(var + "_" + str(i)), stmt)
                stmt = util.replace_symbol(self.target_var, C.Add(C.SymbolRef(self.target_var), C.Constant(i)), stmt)
                body.append(stmt)
            return body
        return node

class LoopUnroller(ast.NodeTransformer):
    def __init__(self, target_var, factor):
        super().__init__()
        self.target_var = target_var
        self.factor = factor
    if False:
        def visit(self, node):
            """
            Support replacing nodes with a list of nodes by flattening `body`
            fields.
            """
            node = super().visit(node)
            if hasattr(node, 'body'):
                node.body = util.flatten(node.body)
            return node

        def visit_For(self, node):
            node.body = [self.visit(s) for s in node.body]
            if node.init.left.name == self.target_var:
                node.incr = C.AddAssign(C.SymbolRef(self.target_var), C.Constant(self.factor))
                visitor = UnrollStatements(self.target_var, self.factor)
                node.body = util.flatten([visitor.visit(s) for s in node.body])
                if node.test.right.value == self.factor:
                    return [util.replace_symbol(node.init.left.name, C.Constant(0), s) for s in node.body]
            return node
    elif False:
        def visit_For(self, node):
            node.body = [self.visit(s) for s in node.body]
            if node.init.left.name == self.target_var:
                # node.pragma = "unroll_and_jam({})".format(self.factor)
                node.pragma = "unroll"
            return node
    else:
        def visit_For(self, node):
            node.body = [self.visit(s) for s in node.body]
            if node.init.left.name == self.target_var:
                node.incr = C.AddAssign(C.SymbolRef(self.target_var), C.Constant(self.factor))
                visitor = UnrollStatements(self.target_var, self.factor)
                node.body = util.flatten([visitor.visit(s) for s in node.body])
            return node


def unroll_inner_neuron_loop(ast, target_var, factor):
    return LoopUnroller(target_var, factor).visit(ast)


class ConstantLoopUnroller(ast.NodeTransformer):
    def visit_For(self, node):
        node.body = util.flatten([self.visit(s) for s in node.body])
        if node.pragma is not None and node.pragma == "unroll":
            length = node.test.right.value
            loop_var = node.init.left.name
            to_return = []
            for i in range(length):
                to_return.append(C.Block([util.replace_symbol(loop_var, C.Constant(i), deepcopy(s)) for s in node.body]))
            return to_return
        return node

def unroll_constant_loops(ast):
    return ConstantLoopUnroller().visit(ast)
