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

import ctypes
import ctree
import copy
import ctree.np
class BasicTypeInference(ast.NodeTransformer):
    def __init__(self):
        self.seen = {}

    def visit_SymbolRef(self, node):
        if node.type is not None:
            self.seen[node.name] = node.type
        return node

    def visit(self, node):
        if hasattr(node, 'body'):
            curr = copy.deepcopy(self.seen)
        node = super().visit(node)
        if hasattr(node, 'body'):
            self.seen = curr
        return node

    def _get_type(self, node):
        if isinstance(node, C.SymbolRef):
            if node.name == "INFINITY":
                return ctypes.c_float()
            elif node.name in self.seen:
                return self.seen[node.name]
        elif isinstance(node, C.UnaryOp):
            return self._get_type(node.arg)
        elif isinstance(node, C.Constant):
            if isinstance(node.value, int):
                # promote all longs to int
                return ctypes.c_int()
            return ctree.types.get_ctype(node.value)
        elif isinstance(node, C.BinaryOp):
            if isinstance(node.op, C.Op.ArrayRef):
                while not isinstance(node, C.SymbolRef):
                    node = node.left
                pointer_type = self._get_type(node)
                return ctree.types.get_c_type_from_numpy_dtype(pointer_type._dtype_)()
            else:
                left = self._get_type(node.left)
                right = self._get_type(node.right)
                return ctree.types.get_common_ctype([left, right])
        elif isinstance(node, C.FunctionCall):
            if node.func.name in ["MAX", "MIN", "max", "min", "floor"]:
                return ctree.types.get_common_ctype([self._get_type(a) for a in node.args])
        raise NotImplementedError(ast.dump(node))
    
    def visit_FunctionCall(self, node):
        if node.func.name in ["max", "min"] and isinstance(self._get_type(node), ctypes.c_float):
            # convert to fmax/fmin
            node.func.name = "f" + node.func.name
        else:
            node.args = [self.visit(a) for a in node.args]
        return node

    def visit_BinaryOp(self, node):
        node.left = self.visit(node.left)
        if isinstance(node.op, C.Op.Assign) and isinstance(node.left, C.SymbolRef) and node.left.name not in self.seen:
            node.left.type = self._get_type(node.right)
            self.seen[node.left.name] = node.left.type
        node.right = self.visit(node.right)
        return node


class SimpleConstantPropogation(ast.NodeTransformer):
    def __init__(self):
        self.seen = {}
    
    def visit(self, node):
        node = super().visit(node)
        if hasattr(node, 'body'):
            node.body = [s for s in node.body if s is not None]
        return node

    def _get_value(self, node):
        if isinstance(node, C.Constant):
            return node.value
        elif isinstance(node, C.SymbolRef) and node.name in self.seen:
            return self.seen[node.name]
        return None

    def visit_For(self, node):
        # Skip loopvar inits
        node.body = [self.visit(s) for s in node.body]
        return node

    def visit_SymbolRef(self, node):
        if node.name in self.seen:
            return C.Constant(self.seen[node.name])
        return node

    def visit_BinaryOp(self, node):
        node.left = self.visit(node.left)
        node.right = self.visit(node.right)
        if isinstance(node.op, C.Op.Assign) and isinstance(node.left, C.SymbolRef):
            value = self._get_value(node.right)
            if value is not None:
                self.seen[node.left.name] = value
                return None
        elif isinstance(node.op, C.Op.Div):
            left = self._get_value(node.left)
            right = self._get_value(node.right)
            if left is not None and right is not None:
                if isinstance(left, int) and isinstance(right, int):
                    return C.Constant(left // right)
                else:
                    return C.Constant(left / right)
        elif isinstance(node.op, C.Op.Mul):
            left = self._get_value(node.left)
            right = self._get_value(node.right)
            if left is not None and right is not None:
                return C.Constant(left * right)
            elif left == 0 or right == 0:
                return C.Constant(0)
        elif isinstance(node.op, C.Op.Mod):
            left = self._get_value(node.left)
            right = self._get_value(node.right)
            if left is not None and right is not None:
                return C.Constant(left % right)
        elif isinstance(node.op, C.Op.Add):
            left = self._get_value(node.left)
            right = self._get_value(node.right)
            if left is not None and right is not None:
                return C.Constant(left + right)
            elif left == 0:
                return node.right
            elif right == 0:
                return node.left
        elif isinstance(node.op, C.Op.Sub):
            left = self._get_value(node.left)
            right = self._get_value(node.right)
            if left is not None and right is not None:
                return C.Constant(left - right)
        return node

class InnerLoopPusher(ast.NodeTransformer):
    def visit_For(self, node):
        if node.init.left.name == "_neuron_index_1_inner":
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
            # curr_node.pragma = "unroll"
            return node

        node.body = [self.visit(s) for s in node.body]
        return node



def push_inner_loop_down(ast):
    return InnerLoopPusher().visit(ast)


class PragmaSIMDInserter(ast.NodeTransformer):
    def visit_For(self, node):
        node.body = [self.visit(s) for s in node.body]
        if node.init.left.name.endswith("_inner"):
            node.pragma = "simd"
        return node

def insert_pragma_simd(ast):
    return PragmaSIMDInserter().visit(ast)

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
