import ast
import math
import ctree.c.nodes as C
import ctypes
import ctree.simd as simd
import ctree.simd.macros as simd_macros
from ctree.transformations import PyBasicConversions
import latte.util as util

SIMDWIDTH = 8

class RegisterPromoteValueRefs(ast.NodeTransformer):
    def __init__(self, ensemble, direction, batch_size, target_loop_var):
        self.ensemble = ensemble
        self.target = "value" if direction == "forward" else "grad"
        self.batch_size = batch_size
        self.seen = {}
        self._vars = []
        self.target_loop_var = target_loop_var

    def visit_BinaryOp(self, node):
        """
        Promote array reference ensemble_name$field[...] to register reference
        $field

        $field is either "value" or "grad" depending on the current direction
        """
        if isinstance(node.op, C.Op.ArrayRef):
            if node.codegen() in self.seen:
                return C.SymbolRef(self.seen[node.codegen()][0])
            curr_node = node
            while not isinstance(curr_node, C.SymbolRef):
                curr_node = curr_node.left
            if curr_node.name.endswith(self.target):
                var = self.target + "_" + str(len(self.seen))
                self.seen[node.codegen()] = (var, node)
                return C.SymbolRef(var)
        node.left = self.visit(node.left)
        node.right = self.visit(node.right)
        return node

    def visit_For(self, node):
        """
        Find the innermost loop to insert a load and store of the target register

        target is either "value" or "grad" depending on direction
        """
        node.body = [self.visit(s) for s in node.body]
        if node.init.left.name == self.target_loop_var:
            for var, seen in self.seen.values():
                node.body.insert(0,
                    C.Assign(
                        C.SymbolRef(var, ctypes.c_float()), 
                        seen
                    ))

            # we only store the value register as "grad" is only read by definition
            if self.target == "value":
                for var, seen in self.seen.values():
                    node.body.append( 
                        C.Assign(
                            seen,
                            C.SymbolRef(var)
                        ))
        return node

def register_promote_value_refs(ast, ensemble, direction, batch_size, target_loop_var):
    return RegisterPromoteValueRefs(ensemble, direction, batch_size, target_loop_var).visit(ast)
