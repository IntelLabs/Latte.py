from .convert_sgemm import convert_sgemm_calls
from .flatten_subscripts import flatten_subscripts
from .convert_tuple_subscripts import convert_tuple_subscripts
from .simple_fusion import simple_fusion
from .pattern_match_gemm import pattern_match_gemm
from .convert_enumerate_range import convert_enumerate_ranges
from .register_promote_value_refs import register_promote_value_refs
from .vectorize_outer_loop import vectorize_outer_loop
from .neuron import NeuronTransformer
from .unroll import unroll_inner_neuron_loop, unroll_constant_loops
from .register_promote import register_promote_vector_loads_stores, lift_invariant_load_stores
from .vectorize import tile_outer_loop, get_loop_to_vectorize, vectorize_loop, fma_replace


import ast
import ctree.c.nodes as C
import latte.util as util
class LoadInterleaver(ast.NodeTransformer):
    def visit(self, node):
        node = super().visit(node)
        if hasattr(node, 'body'):
            new_body = []
            for stmt in reversed(node.body):
                if isinstance(stmt, C.BinaryOp) and isinstance(stmt.op, C.Op.Assign) and \
                        isinstance(stmt.right, C.FunctionCall) and stmt.right.func.name in ["_mm256_set1_ps"]:
                    value = stmt.left.name
                    for i in range(len(new_body)):
                        if util.contains_symbol(new_body[i], value):
                            new_body.insert(i, stmt)
                            break
                else:
                    new_body.insert(0, stmt)
            node.body = new_body
        return node

def interleave_loads(ast):
    return LoadInterleaver().visit(ast)

import ctree.simd as simd
class SingleUsePromotor(ast.NodeTransformer):
    def visit(self, node):
        node = super().visit(node)
        if hasattr(node, 'body'):
            new_body = []
            for i, stmt in enumerate(node.body):
                if isinstance(stmt, C.BinaryOp) and isinstance(stmt.op, C.Op.Assign) and \
                        isinstance(stmt.left.type, simd.types.m256):
                    value = stmt.left.name
                    counter = 0
                    for _stmt in node.body[i+1:]:
                        counter += util.count_symbol_instances(_stmt, value)
                    if counter > 1:
                        new_body.append(stmt)
                    else:
                        for i in range(i+1, len(node.body)):
                            node.body[i] = util.replace_symbol(value, stmt.right, node.body[i])
                else:
                    new_body.append(stmt)
            node.body = new_body
        return node

def promote_single_use_registers(ast):
    return SingleUsePromotor().visit(ast)


class TiledLoopInterchanger(ast.NodeTransformer):
    def __init__(self):
        self.tiled_loops = []

    def visit_For(self, node):
        node.body = [self.visit(s) for s in node.body]
        if node.init.left.name == "_neuron_index_0":
            for loop in self.tiled_loops:
                loop.body = node.body
                node.body = [loop]
            return node
        if node.init.left.name.endswith("_outer"):
            self.tiled_loops.append(node)
            return node.body[0]
        return node

def interchange_tiled_loops(ast):
    return TiledLoopInterchanger().visit(ast)

class InnerLoopInterchanger(ast.NodeTransformer):
    def visit_For(self, node):
        node.body = [self.visit(s) for s in node.body]
        if node.init.left.name.endswith("_inner"):
            if len(node.body) == 1 and isinstance(node.body[0], C.For):
                print(node)
                to_return = node.body[0]
                tmp = node.body[0].body
                node.body[0].body = [node]
                node.body = tmp
                print(to_return)
                while sum(isinstance(s, C.For) for s in node.body) == 1:
                    for index, s in enumerate(node.body):
                        if isinstance(s, C.For):
                            tmp_init = node.init
                            tmp_test = node.test
                            tmp_incr = node.incr
                            node.init = s.init
                            node.test = s.test
                            node.incr = s.incr
                            node.pragma = s.pragma
                            s.init = tmp_init
                            s.test = tmp_test
                            s.incr = tmp_incr
                            node = s
                    print(to_return)
                print(to_return)
                return to_return
        return node

def interchange_inner_loop(ast):
    return InnerLoopInterchanger().visit(ast)
