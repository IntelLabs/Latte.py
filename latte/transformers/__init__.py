from .convert_sgemm import convert_sgemm_calls
from .flatten_subscripts import flatten_subscripts
from .convert_tuple_subscripts import convert_tuple_subscripts
from .simple_fusion import simple_fusion
from .pattern_match_gemm import pattern_match_gemm
from .convert_enumerate_range import convert_enumerate_ranges
from .register_promote_value_refs import register_promote_value_refs
from .vectorize_outer_loop import vectorize_outer_loop
from .neuron import NeuronTransformer
import latte.transformers.unroll
from .pattern_match_math import PatternMatchMath
from .vectorize import fuse_multiply_adds, vectorize_loop

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
                        isinstance(stmt.right, C.FunctionCall) and stmt.right.func.name in ["_mm256_broadcast_ss"]:
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
            seen = set()
            for i, stmt in enumerate(node.body):
                if isinstance(stmt, C.BinaryOp) and isinstance(stmt.op, C.Op.Assign) and \
                        (isinstance(stmt.left.type, simd.types.m256) or stmt.left.name in seen):
                    seen.add(stmt.left.name)
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
        if node.init.left.name.endswith("_outer") and "1" not in node.init.left.name:
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

class InnerLoopPusher(ast.NodeTransformer):
    def visit_For(self, node):
        if "_inner" in node.init.left.name:
            curr_node = node
            outer_body = node.body[:-1]
            node.body = [node.body[-1]]
            while isinstance(curr_node.body[-1], C.For):
                tmp_init = curr_node.init
                tmp_test = curr_node.test
                tmp_incr = curr_node.incr
                tmp_pragma = curr_node.pragma
                curr_node.init = curr_node.body[-1].init
                curr_node.test = curr_node.body[-1].test
                curr_node.incr = curr_node.body[-1].incr
                curr_node.pragma = curr_node.body[-1].pragma
                curr_node.body[-1].init = tmp_init
                curr_node.body[-1].test = tmp_test
                curr_node.body[-1].incr = tmp_incr
                curr_node.body[-1].pragma = tmp_pragma
                curr_node = curr_node.body[-1]
            curr_node.body = outer_body + curr_node.body
            curr_node.pragma = "unroll"
            return node

        node.body = [self.visit(s) for s in node.body]
        return node



def push_inner_loop_down(ast):
    return InnerLoopPusher().visit(ast)


class PragmaSIMDInserter(ast.NodeTransformer):
    def __init__(self, loop_var):
        self.loop_var = loop_var

    def visit_For(self, node):
        node.body = [self.visit(s) for s in node.body]
        if node.init.left.name == self.loop_var:
            node.pragma = "simd"
        return node

def insert_pragma_simd(ast, loop_var):
    return PragmaSIMDInserter(loop_var).visit(ast)

def move_inner_index(tree):
    class Transformer(ast.NodeTransformer):
        def __init__(self):
            self.loop_vars = ["omp_get_thread_num"]

        def visit_For(self, node):
            self.loop_vars.append(node.init.left.name)
            node.body = [self.visit(s) for s in node.body]
            return node

        def visit_BinaryOp(self, node):
            node.left = self.visit(node.left)
            node.right = self.visit(node.right)
            if isinstance(node.op, C.Op.ArrayRef) and isinstance(node.left, C.BinaryOp):
                for curr_index, var in enumerate(self.loop_vars):
                    if util.contains_symbol(node.right, var):
                        break
                for left_index, var in enumerate(self.loop_vars):
                    if util.contains_symbol(node.left.right, var):
                        break
                if curr_index < left_index:
                    node.left.right, node.right = node.right, node.left.right
                    node.left = self.visit(node.left)
                    return node
            return node
    return Transformer().visit(tree)

def lift_loads(tree):
    class Transformer(ast.NodeTransformer):
        def visit_For(self, node):
            node.body = [self.visit(s) for s in node.body]
            pre_stmts = []
            loads = []
            rest = []
            for stmt in node.body:
                if not hasattr(stmt, 'body'):
                    if util.contains_symbol(stmt, "_mm256_load_ps"):
                        loads.append(stmt)
                    elif isinstance(stmt, C.BinaryOp) and isinstance(stmt.op, C.Op.Assign) and isinstance(stmt.left, C.SymbolRef) and stmt.left.type is not None:
                        pre_stmts.append(stmt)
                    else:
                        rest.append(stmt)
                else:
                    rest.append(stmt)
            node.body = pre_stmts + loads + rest
            return node
    return Transformer().visit(tree)

def remove_repeated_declarations(tree):
    class Transformer(ast.NodeTransformer):
        def visit(self, node):
            node = super().visit(node)
            if hasattr(node, 'body'):
                seen = set()
                for stmt in node.body:
                    if isinstance(stmt, C.BinaryOp) and isinstance(stmt.op, C.Op.Assign) and \
                            isinstance(stmt.left, C.SymbolRef) and stmt.left.type is not None:
                        if stmt.left.name in seen:
                            stmt.left.type = None
                        else:
                            seen.add(stmt.left.name)
            return node
    return Transformer().visit(tree)

def promote_in_place_load_stores(tree, in_place_buffers):
    class Transformer(ast.NodeTransformer):
        def _get_array(self, array_ref):
            node = array_ref
            if isinstance(node, C.UnaryOp):
                node = node.arg
            while not isinstance(node, C.SymbolRef):
                node = node.left
            return node.name

        def _is_inplace_store(self, stmt):
            return isinstance(stmt, C.FunctionCall) and stmt.func.name == "_mm256_store_ps" and \
                    self._get_array(stmt.args[0]) in in_place_buffers

        def _is_inplace_load(self, stmt, target):
            return isinstance(stmt, C.BinaryOp) and isinstance(stmt.op, C.Op.Assign) and \
                    isinstance(stmt.right, C.FunctionCall) and stmt.right.func.name == "_mm256_load_ps" and \
                    self._get_array(stmt.right.args[0]) in in_place_buffers[target]

        def visit(self, node):
            node = super().visit(node)
            if hasattr(node, 'body'):
                new_body = []
                for i, stmt1 in enumerate(node.body):
                    if self._is_inplace_store(stmt1):
                        target = self._get_array(stmt1.args[0])
                        add = True
                        for stmt2 in node.body[i+1:]:
                            if self._is_inplace_load(stmt2, target):
                                stmt2.right = stmt1.args[1]
                                add = False
                                break
                        if add:
                            new_body.append(stmt1)
                    else:
                        new_body.append(stmt1)
                node.body = new_body
            return node
    return Transformer().visit(tree)
