import latte.util as util
import ast
import ctree.c.nodes as C
from copy import deepcopy

class UnrollStatements(ast.NodeTransformer):
    def __init__(self, target_var, factor, unroll_type):
        super().__init__()
        self.target_var = target_var
        self.factor = factor
        self.unrolled_vars = set()
        self.unroll_type = unroll_type

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
            check = [util.contains_symbol(node.right, var) for var in list(self.unrolled_vars)+ [self.target_var]]
            if any(check):
                body = []
                if hasattr(node.left, 'type') and node.left.type is not None:
                    self.unrolled_vars.add(node.left.name)
                for i in range(self.factor):
                    stmt = deepcopy(node)
                    for var in self.unrolled_vars:
                        stmt = util.replace_symbol(var, C.SymbolRef(var + "_" + str(i)), stmt)
                    if self.unroll_type == 0:
                        body.append(util.replace_symbol(self.target_var, C.Add(C.SymbolRef(self.target_var), C.Constant(i)), stmt))
                    elif self.unroll_type == 1 :
                        body.append(util.replace_symbol(self.target_var, C.Add(C.Mul(C.Constant(self.factor),C.SymbolRef(self.target_var)), C.Constant(i)), stmt))
                    else:
                       assert(false)
                return body
        return node

    def visit_AugAssign(self, node):
        check = [util.contains_symbol(node.value, var) for var in list(self.unrolled_vars) + [self.target_var]]
        if any(check):
            body = []
            if isinstance(node.target, C.SymbolRef):
                self.unrolled_vars.add(self._get_name(node.target.name))
                for i in range(self.factor):
                    stmt = deepcopy(node)
                    for var in self.unrolled_vars:
                        stmt = util.replace_symbol(var, C.SymbolRef(var + "_" + str(i)), stmt)
                    #body.append(util.replace_symbol(self.target_var, C.Add(C.SymbolRef(self.target_var), C.Constant(i)), stmt))
                    if self.unroll_type == 0:
                        body.append(util.replace_symbol(self.target_var, C.Add(C.SymbolRef(self.target_var), C.Constant(i)), stmt))
                    elif self.unroll_type == 1 :
                        body.append(util.replace_symbol(self.target_var, C.Add(C.Mul(C.Constant(self.factor),C.SymbolRef(self.target_var)), C.Constant(i)), stmt))
                    else:
                       assert(false)


                return body
            elif isinstance(node.target, C.BinaryOp) and isinstance(node.target.op, C.Op.ArrayRef):
                assert False
                for i in range(self.factor):
                    stmt = deepcopy(node)
                    for var in self.unrolled_vars:
                        stmt = util.replace_symbol(var, C.SymbolRef(var + "_" + str(i)), stmt)
                    #body.append(util.replace_symbol(self.target_var, C.Add(C.SymbolRef(self.target_var), C.Constant(i)), stmt))

                    if self.unroll_type == 0:
                        body.append(util.replace_symbol(self.target_var, C.Add(C.SymbolRef(self.target_var), C.Constant(i)), stmt))
                    elif self.unroll_type == 1 :
                        body.append(util.replace_symbol(self.target_var, C.Add(C.Mul(C.Constant(self.factor),C.SymbolRef(self.target_var)), C.Constant(i)), stmt))
                    else:
                       assert(false)

                return body
            else:
                raise NotImplementedError()
        return node

    def visit_FunctionCall(self, node):
        check = [util.contains_symbol(node, var) for var in list(self.unrolled_vars) + [self.target_var]]
        if "store" in node.func.name and "_mm" in node.func.name and any(check):
            body = []
            for i in range(self.factor):
                stmt = deepcopy(node)
                for var in self.unrolled_vars:
                    stmt = util.replace_symbol(var, C.SymbolRef(var + "_" + str(i)), stmt)
                #stmt = util.replace_symbol(self.target_var, C.Add(C.SymbolRef(self.target_var), C.Constant(i)), stmt)
                #body.append(stmt)
                if self.unroll_type == 0:
                    body.append(util.replace_symbol(self.target_var, C.Add(C.SymbolRef(self.target_var), C.Constant(i)), stmt))
                elif self.unroll_type == 1 :
                    body.append(util.replace_symbol(self.target_var, C.Add(C.Mul(C.Constant(self.factor),C.SymbolRef(self.target_var)), C.Constant(i)), stmt))
                else:
                    assert(false)

            return body
        return node

class LoopUnroller(ast.NodeTransformer):
    def __init__(self, target_var, factor, unroll_type):
        super().__init__()
        self.target_var = target_var
        self.unroll_type = unroll_type
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
                if self.unroll_type == 0:
                    node.incr = C.AddAssign(C.SymbolRef(self.target_var), C.Constant(self.factor))
                    node.incr = C.AddAssign(C.SymbolRef(self.target_var), C.Constant(self.factor))
                elif self.unroll_type == 1:
                    assert(node.test.right.value%self.factor == 0)
                    node.test.right.value = node.test.right.value//self.factor
                else:
                    assert(0)   
                visitor = UnrollStatements(self.target_var, self.factor, self.unroll_type)
                node.body = util.flatten([visitor.visit(s) for s in node.body])
            return node


def unroll_loop(ast, target_var, factor, unroll_type=0):
    return LoopUnroller(target_var, factor, unroll_type).visit(ast)

class UnrollStatementsNoJam(ast.NodeTransformer):
    new_body={}
    def __init__(self, target_var, factor, unroll_type):
        super().__init__()
        self.target_var = target_var
        self.factor = factor
        self.unrolled_vars = set()
        self.unroll_type = unroll_type

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
            check = [util.contains_symbol(node.right, var) for var in list(self.unrolled_vars)+ [self.target_var]]
            if any(check):
                body = []
                if hasattr(node.left, 'type') and node.left.type is not None:
                    self.unrolled_vars.add(node.left.name)
                for i in range(self.factor):
                    stmt = deepcopy(node)
                    for var in self.unrolled_vars:
                        stmt = util.replace_symbol(var, C.SymbolRef(var + "_" + str(i)), stmt)
                    if self.unroll_type == 0:
                        UnrollStatementsNoJam.new_body[var].append(util.replace_symbol(self.target_var, C.Add(C.SymbolRef(self.target_var), C.Constant(i)), stmt))
                    elif self.unroll_type == 1 :
                        UnrollStatementsNoJam.new_body[var].append(util.replace_symbol(self.target_var, C.Add(C.Mul(C.Constant(self.factor),C.SymbolRef(self.target_var)), C.Constant(i)), stmt))
                    else:
                       assert(false)
                return body
        return node

    def visit_AugAssign(self, node):
        check = [util.contains_symbol(node.value, var) for var in list(self.unrolled_vars) + [self.target_var]]
        if any(check):
            body = []
            if isinstance(node.target, C.SymbolRef):
                self.unrolled_vars.add(self._get_name(node.target.name))
                for i in range(self.factor):
                    stmt = deepcopy(node)
                    for var in self.unrolled_vars:
                        stmt = util.replace_symbol(var, C.SymbolRef(var + "_" + str(i)), stmt)
                    #body.append(util.replace_symbol(self.target_var, C.Add(C.SymbolRef(self.target_var), C.Constant(i)), stmt))
                    if self.unroll_type == 0:
                        UnrollStatementsNoJam.new_body[var].append(util.replace_symbol(self.target_var, C.Add(C.SymbolRef(self.target_var), C.Constant(i)), stmt))
                    elif self.unroll_type == 1 :
                        UnrollStatementsNoJam.new_body[var]..append(util.replace_symbol(self.target_var, C.Add(C.Mul(C.Constant(self.factor),C.SymbolRef(self.target_var)), C.Constant(i)), stmt))
                    else:
                       assert(false)


                return body
            elif isinstance(node.target, C.BinaryOp) and isinstance(node.target.op, C.Op.ArrayRef):
                assert False
                for i in range(self.factor):
                    stmt = deepcopy(node)
                    for var in self.unrolled_vars:
                        stmt = util.replace_symbol(var, C.SymbolRef(var + "_" + str(i)), stmt)
                    #body.append(util.replace_symbol(self.target_var, C.Add(C.SymbolRef(self.target_var), C.Constant(i)), stmt))

                    if self.unroll_type == 0:
                        UnrollStatementsNoJam.new_body[var].append(util.replace_symbol(self.target_var, C.Add(C.SymbolRef(self.target_var), C.Constant(i)), stmt))
                    elif self.unroll_type == 1 :
                        UnrollStatementsNoJam.new_body[var].append(util.replace_symbol(self.target_var, C.Add(C.Mul(C.Constant(self.factor),C.SymbolRef(self.target_var)), C.Constant(i)), stmt))
                    else:
                       assert(false)

                return body
            else:
                raise NotImplementedError()
        return node

    def visit_FunctionCall(self, node):
        check = [util.contains_symbol(node, var) for var in list(self.unrolled_vars) + [self.target_var]]
        if "store" in node.func.name and "_mm" in node.func.name and any(check):
            body = []
            for i in range(self.factor):
                stmt = deepcopy(node)
                for var in self.unrolled_vars:
                    stmt = util.replace_symbol(var, C.SymbolRef(var + "_" + str(i)), stmt)
                #stmt = util.replace_symbol(self.target_var, C.Add(C.SymbolRef(self.target_var), C.Constant(i)), stmt)
                #body.append(stmt)
                if self.unroll_type == 0:
                    UnrollStatementsNoJam.new_body[var].append(util.replace_symbol(self.target_var, C.Add(C.SymbolRef(self.target_var), C.Constant(i)), stmt))
                elif self.unroll_type == 1 :
                    UnrollStatementsNoJam.new_body[var].append(util.replace_symbol(self.target_var, C.Add(C.Mul(C.Constant(self.factor),C.SymbolRef(self.target_var)), C.Constant(i)), stmt))
                else:
                    assert(false)

            return body
        return node

class LoopUnrollerNoJam(ast.NodeTransformer):
    def __init__(self, target_var, factor, unroll_type):
        super().__init__()
        self.target_var = target_var
        self.unroll_type = unroll_type
        self.factor = factor
    def visit_For(self, node):
        node.body = [self.visit(s) for s in node.body]
        if node.init.left.name == self.target_var:
            if self.unroll_type == 0:
                node.incr = C.AddAssign(C.SymbolRef(self.target_var), C.Constant(self.factor))
                node.incr = C.AddAssign(C.SymbolRef(self.target_var), C.Constant(self.factor))
            elif self.unroll_type == 1:
                assert(node.test.right.value%self.factor == 0)
                node.test.right.value = node.test.right.value//self.factor
            else:
                assert(0)   
            UnrollStatementsNoJam.new_body=[] 
            visitor = UnrollStatementsNoJam(self.target_var, self.factor, self.unroll_type)
            node.body = util.flatten([visitor.visit(s) for s in node.body])
            node.body = node.body + UnrollStatementsNoJam.new_body
        return node
def unroll_no_jam_loop(ast, target_var, factor, unroll_type=0):
    return LoopUnrollerNoJam(target_var, factor, unroll_type).visit(ast)
