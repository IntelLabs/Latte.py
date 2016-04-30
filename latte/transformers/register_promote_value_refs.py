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
        self.direction = direction
        self.batch_size = batch_size

    def visit_Subscript(self, node):
        if self.direction == "forward":
            target = "value"
        elif self.direction == "backward":
            target = "grad"
        else:
            raise NotImplementedError
        if node.value.id.endswith(target):
            return ast.Name(target, node.ctx)
        return node

    def visit_For(self, node):
        node.body = [self.visit(s) for s in node.body]
        if isinstance(node.target, ast.Name) and node.target.id == "_neuron_index_{}".format(self.ensemble.ndim):
            if self.direction == "forward":
                to_load = "value"
            elif self.direction == "backward":
                to_load = "grad"
            else:
                raise NotImplementedError
            # Initialize value
            idxs = [C.SymbolRef("_neuron_index_{}".format(i)) for i in range(self.ensemble.ndim + 1)]
            idx_expr = C.ArrayRef(C.SymbolRef(self.ensemble.name+to_load), idxs[0])
            for idx in idxs[1:]:
                idx_expr = C.ArrayRef(idx_expr, idx)
            # flat_idx = util.gen_flat_index(index, (self.batch_size, ) + self.ensemble.shape)
            # flat_idx = util.gen_flat_index(index, (self.batch_size, math.ceil(self.ensemble.shape[0] / float(SIMDWIDTH))) + self.ensemble.shape[1:] + (SIMDWIDTH, ))
            # flat_idx = PyBasicConversions().visit(flat_idx)
            node.body.insert(0, 
                C.Assign(
                    # C.SymbolRef(to_load, ctypes.c_float()), idx_expr
                    C.SymbolRef(to_load, simd.types.m256()), 
                    simd_macros.mm256_load_ps( 
                        idx_expr)
                ))
            if to_load == "value":
                # Store value
                node.body.append( 
                    # C.Assign(idx_expr, C.SymbolRef(to_load), ))
                    simd_macros.mm256_store_ps(
                        idx_expr,
                        C.SymbolRef(to_load)
                    ))
        return node

def register_promote_value_refs(ast, ensemble, direction, batch_size):
    return RegisterPromoteValueRefs(ensemble, direction, batch_size).visit(ast)
