
import latte.util as util
import ctypes
import ast
import ctree.c.nodes as C
from copy import deepcopy

du_map = {}
class ReplaceSymbolRef(ast.NodeTransformer):
    def __init__(self):
        super().__init__()

    def visit(self, node):
        return super().visit(node)

    def visit_SymbolRef(self, node):
        if node.name in du_map:
          return du_map[node.name]
        return node

class CopyPropagation(ast.NodeTransformer):
    def __init__(self):
        super().__init__()

    def visit_For(self, node):
        node.body = util.flatten([s for s in node.body])
        new_body = []
        for stmt in node.body:
          if isinstance(stmt, C.BinaryOp) and \
             isinstance(stmt.op, C.Op.Assign) and \
             isinstance(stmt.left, C.SymbolRef) and \
             (stmt.left.name.startswith("in_") or stmt.left.name.startswith("_input_")) and \
             not isinstance(stmt.right, C.FunctionCall):
               new_body.append(stmt)
               if isinstance(stmt.right, C.SymbolRef) and \
                 stmt.right.name in du_map:
                 du_map[stmt.left.name] = du_map[stmt.right.name]
               else:
                 du_map[stmt.left.name] = stmt.right

          elif isinstance(stmt, C.BinaryOp) and \
             isinstance(stmt.op, C.Op.Assign) and \
             isinstance(stmt.left, C.SymbolRef) and \
             isinstance(stmt.right, C.FunctionCall) and "_mm" in stmt.right.func.name \
             and ("_load_" in stmt.right.func.name or "_set1" in stmt.right.func.name or "_broadcast" in stmt.right.func.name):
               stmt = ReplaceSymbolRef().visit(stmt)
               new_body.append(stmt)
          elif isinstance(stmt, C.FunctionCall) and "_mm" in stmt.func.name and "_store" in stmt.func.name:
               stmt = ReplaceSymbolRef().visit(stmt)
               new_body.append(stmt)
          else:
               new_body.append(stmt)
        node.body = util.flatten([self.visit(s) for s in new_body])
        return node

def propagate_copies(ast):
     du_map = {}
     return CopyPropagation().visit(ast)

