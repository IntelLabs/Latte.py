
import latte.util as util
import ctypes
import ast
import ctree.c.nodes as C
from copy import deepcopy

class LoopSimplifier(ast.NodeTransformer):
    def __init__(self):
        super().__init__()

    def visit_For(self, node):
        node.body = util.flatten([self.visit(s) for s in node.body])
        #TODO: assumption is that every loop starts with zero, not negative
        init = -1
        incr = -1
        test = -1
        if isinstance(node.init, C.BinaryOp) and \
           isinstance(node.init.op, C.Op.Assign) and \
           isinstance(node.init.left, C.SymbolRef) and \
           isinstance(node.init.right, C.Constant):
           init = node.init.right.value

        if isinstance(node.test, C.BinaryOp) and \
           isinstance(node.test.op, C.Op.Lt) and \
           isinstance(node.test.left, C.SymbolRef) and \
           isinstance(node.test.right, C.Constant):
           test = node.test.right.value

        if isinstance(node.incr, C.AugAssign) and \
           isinstance(node.incr.op, C.Op.Add) and \
           isinstance(node.incr.target, C.SymbolRef) and \
           isinstance(node.incr.value, C.Constant):
           incr = node.incr.value.value

        if init != -1 and test != -1 and incr != -1 and (init+incr) >= test:
          return [util.replace_symbol(node.init.left.name, C.Constant(init), s) for s in node.body]

        return node

def simplify_loops(ast):
     return LoopSimplifier().visit(ast)

