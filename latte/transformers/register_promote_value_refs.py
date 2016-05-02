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
    def __init__(self, ensemble, direction, batch_size):
        self.ensemble = ensemble
        self.target = "value" if direction == "forward" else "backward"
        self.batch_size = batch_size

    def visit_Subscript(self, node):
        """
        Promote array reference ensemble_name$field[...] to register reference
        $field

        $field is either "value" or "grad" depending on the current direction
        """
        if node.value.id.endswith(self.target):
            return ast.Name(self.target, node.ctx)
        return node

    def _gen_nested_index_expr(self, target, idxs):
        """
        Generates a nested indexing expression target[i1][i2][i3] for a list of
        idxs = [i1, i2, i3]
        """
        idx_expr = C.ArrayRef(target, idxs[0])
        for idx in idxs[1:]:
            idx_expr = C.ArrayRef(idx_expr, idx)
        return idx_expr


    def visit_For(self, node):
        """
        Find the innermost loop to insert a load and store of the target register

        target is either "value" or "grad" depending on direction
        """
        node.body = [self.visit(s) for s in node.body]
        # innermost loop uses "_neuron_index_X" where X = ensemble.ndim
        if isinstance(node.target, ast.Name) and \
                node.target.id == "_neuron_index_{}".format(self.ensemble.ndim):

            idxs = [C.SymbolRef("_neuron_index_{}".format(i)) for i in
                    range(self.ensemble.ndim + 1)]

            idx_expr = self._gen_nested_index_expr(
                C.SymbolRef(self.ensemble.name+self.target), idxs)

            # Insert a vectorized load for the current value of target[idx...]
            node.body.insert(0, 
                C.Assign(
                    C.SymbolRef(self.target, simd.types.m256()), 
                    simd_macros.mm256_load_ps(idx_expr)
                ))

            # we only store the value register as "grad" is only read by definition
            if self.target == "value":
                node.body.append( 
                    simd_macros.mm256_store_ps(
                        idx_expr,
                        C.SymbolRef(self.target)
                    ))
        return node

def register_promote_value_refs(ast, ensemble, direction, batch_size):
    return RegisterPromoteValueRefs(ensemble, direction, batch_size).visit(ast)
