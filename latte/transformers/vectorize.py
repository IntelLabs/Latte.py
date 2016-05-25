import ast
import ctree.c.nodes as C
import ctree.simd.macros as simd_macros
import latte
import latte.util as util
import ctree.simd as simd
import ctypes
from copy import deepcopy
from ctree.templates.nodes import StringTemplate

class OuterLoopTiler(ast.NodeTransformer):
    def __init__(self, ndim):
        self.ndim = ndim
        self.tiled_buffers = {}

    def visit_For(self, node):
        # if node.init.left.name == "_neuron_index_1":
        #     node.test.right.value = node.test.right.value // latte.core.SIMDWIDTH

        node.body = [self.visit(s) for s in node.body]

        # if node.init.left.name == "_neuron_index_{}".format(self.ndim):
        #     node.body = [C.For(
        #         C.Assign(C.SymbolRef("_neuron_index_1_inner", ctypes.c_int()), C.Constant(0)),
        #         C.Lt(C.SymbolRef("_neuron_index_1_inner"), C.Constant(latte.core.SIMDWIDTH)),
        #         C.PostInc(C.SymbolRef("_neuron_index_1_inner")),
        #         node.body
        #     )]
        tile_size = 7
        if node.init.left.name == "_neuron_index_{}".format(self.ndim):
            # node.body = [C.For(
            #     C.Assign(C.SymbolRef("_neuron_index_1_inner", ctypes.c_int()), C.Constant(0)),
            #     C.Lt(C.SymbolRef("_neuron_index_1_inner"), C.Constant(latte.core.SIMDWIDTH)),
            #     C.PostInc(C.SymbolRef("_neuron_index_1_inner")),
            #     node.body
            # )]
            for i in reversed(range(self.ndim, self.ndim + 1)):
                loopvar = "_neuron_index_{}".format(i)
                node.body = [C.For(
                    C.Assign(C.SymbolRef(loopvar, ctypes.c_int()), C.Mul(C.SymbolRef(loopvar + "_outer"), C.Constant(tile_size))),
                    C.Lt(C.SymbolRef(loopvar), C.Mul(C.Add(C.SymbolRef(loopvar + "_outer"), C.Constant(1)), C.Constant(tile_size))),
                    C.PostInc(C.SymbolRef(loopvar)),
                    node.body
                )]
        if node.init.left.name in ["_neuron_index_{}".format(i) for i in range(self.ndim, self.ndim + 1)]:
            node.init.left.name += "_outer"
            node.incr.target.name += "_outer"
            node.test.left.name += "_outer"
            node.test.right.value //= tile_size

        return node

    # def visit_BinaryOp(self, node):
    #     if isinstance(node.op, C.Op.ArrayRef) and util.contains_symbol(node, "_neuron_index_1"):
    #         idx = 0
    #         curr_node = node
    #         while not isinstance(curr_node.right, C.SymbolRef) or \
    #                 curr_node.right.name != "_neuron_index_1":
    #             idx += 1
    #             curr_node = curr_node.left
    #         while not isinstance(curr_node, C.SymbolRef):
    #             curr_node = curr_node.left
    #         self.tiled_buffers[curr_node.name] = idx

    #         return C.ArrayRef(node, C.SymbolRef("_neuron_index_1_inner"))
    #     node.left = self.visit(node.left)
    #     node.right = self.visit(node.right)
    #     return node

def tile_outer_loop(ast, ndim):
    transformer = OuterLoopTiler(ndim)
    ast = transformer.visit(ast)
    return ast, transformer.tiled_buffers

class LoopToVectorizeFinder(ast.NodeVisitor):
    def __init__(self):
        self.stores = []
        self.dependencies = set()

    def visit_AugAssign(self, node):
        if isinstance(node.target, C.BinaryOp) and isinstance(node.target.op, C.Op.ArrayRef) and \
                not any(util.contains_symbol(node.target, dep) for dep in self.dependencies):
            self.stores.append(node.target)

    def visit_If(self, node):
        if node.elze is None:
            self.stores.append(False)

    def visit_BinaryOp(self, node):
        if isinstance(node.op, C.Op.Assign) and isinstance(node.left, C.BinaryOp) and isinstance(node.left.op, C.Op.ArrayRef):
            self.stores.append(node.left)
        elif isinstance(node.op, C.Op.Assign) and isinstance(node.left, C.SymbolRef) and isinstance(node.right, C.BinaryOp) and isinstance(node.right.op, C.Op.ArrayRef):
            self.dependencies.add(node.left.name)

def get_loop_to_vectorize(tree):
    visitor = LoopToVectorizeFinder()
    visitor.visit(tree)
    stores = visitor.stores
    if len(stores) == 0:
        return None
    elif any(x == False for x in stores):
        return None
    elif len(stores) == 1:
        if isinstance(stores[0].right, C.SymbolRef):
            return stores[0].right.name
    elif all(x.right.name == stores[0].right.name for x in stores):
        return stores[0].right.name
    return None


class RemoveIndexExprs(ast.NodeTransformer):
    def __init__(self, var):
        self.var = var

    def visit_BinaryOp(self, node):
        if isinstance(node.op, C.Op.Assign) and isinstance(node.left, C.SymbolRef) and \
                util.contains_symbol(node.right, self.var):
            return None
        return node

class Vectorizer(ast.NodeTransformer):
    def __init__(self, loop_var):
        self.loop_var = loop_var
        self.transposed_buffers = {}

    def visit(self, node):
        node = super().visit(node)
        if hasattr(node, "body"):
            node.body = util.flatten(node.body)
        return node

    def visit_For(self, node):
        node.body = [self.visit(s) for s in node.body]
        if node.init.left.name == self.loop_var:
            body = node.body
            body = [RemoveIndexExprs(self.loop_var).visit(s) for s in node.body]
            body = [s for s in body if s is not None]
            # if len(self.transposed_buffers) > 0:
            #     for buffer_name in self.transposed_buffers:
            #         body.insert(0, 
            #             StringTemplate(
    # """
    # transpose<SIMDWIDTH,SIMDWIDTH>({buffer_name}, {buffer_name}_transposed);
    # """.format(buffer_name=buffer_name)
            #         ))
            return body
        return node

    def visit_AugAssign(self, node):
        node.value = self.visit(node.value)
        if util.contains_symbol(node.target, self.loop_var):
            if node.target.right.name != self.loop_var:
                target = self.visit(deepcopy(node.target))
                curr_node = node.target
                idx = 1
                while curr_node.left.right.name != self.loop_var:
                    curr_node = curr_node.left
                    idx += 1
                curr_node.left = curr_node.left.left
                while not isinstance(curr_node, C.SymbolRef):
                    curr_node = curr_node.left
                if curr_node.name in self.transposed_buffers and self.transposed_buffers[curr_node.name] != idx:
                    raise NotImplementedError()
                self.transposed_buffers[curr_node.name] = idx
                curr_node.name += "_transposed"
                return simd_macros.mm256_store_ps(
                        node.target,
                        C.BinaryOp(target, node.op, node.value))
            else:
                return simd_macros.mm256_store_ps(
                        node.target.left,
                        C.BinaryOp(self.visit(node.target), node.op, node.value))
        # elif isinstance(node.op, C.Op.Add) and isinstance(node.value, C.BinaryOp) and \
        #         isinstance(node.value.op, C.Op.Mul):
        #     # if not isinstance(node.target, C.SymbolRef):
        #     #     node.value = C.FunctionCall(C.SymbolRef("vsum"), [node.value])
        #     #     return node
        #     # else:
        #         return C.Assign(node.target, C.FunctionCall(C.SymbolRef("_mm256_fmadd_ps"), [node.value.left, node.value.right, node.target]))
        elif isinstance(node.op, C.Op.Add) and isinstance(node.value, C.FunctionCall):
            # TODO: Verfiy it's a vector intrinsic
            return C.Assign(node.target, C.FunctionCall(C.SymbolRef("_mm256_add_ps"), [node.value, node.target]))
        elif isinstance(node.target, C.BinaryOp) and isinstance(node.target.op, C.Op.ArrayRef):
            raise NotImplementedError(node)
        node.target = self.visit(node.target)
        return node

    def visit_BinaryOp(self, node):
        if isinstance(node.op, C.Op.ArrayRef):
            if util.contains_symbol(node, self.loop_var):
                if node.right.name != self.loop_var:
                    curr_node = node
                    idx = 1
                    while curr_node.left.right.name != self.loop_var:
                        curr_node = curr_node.left
                        idx += 1
                    curr_node.left = curr_node.left.left
                    while not isinstance(curr_node, C.SymbolRef):
                        curr_node = curr_node.left
                    if curr_node.name in self.transposed_buffers and self.transposed_buffers[curr_node.name] != idx:
                        raise NotImplementedError()
                    self.transposed_buffers[curr_node.name] = idx
                    curr_node.name += "_transposed"
                    return simd_macros.mm256_load_ps(node)
                else:
                    return simd_macros.mm256_load_ps(node.left)
            else:
                return C.FunctionCall(C.SymbolRef("_mm256_broadcast_ss"), [C.Ref(node)])
        elif isinstance(node.op, C.Op.Assign):
            node.right = self.visit(node.right)
            if isinstance(node.right, C.FunctionCall) and \
                    node.right.func.name in ["_mm256_load_ps", "_mm256_broadcast_ss"] and \
                    node.left.type is not None:
                node.left.type = simd.types.m256()
                return node
            elif isinstance(node.left, C.BinaryOp) and util.contains_symbol(node.left, self.loop_var):
                if node.left.right.name != self.loop_var:
                    curr_node = node
                    idx = 1
                    while curr_node.left.right.name != self.loop_var:
                        curr_node = curr_node.left
                        idx += 1
                    curr_node.left = curr_node.left.left
                    while not isinstance(curr_node, C.SymbolRef):
                        curr_node = curr_node.left
                    if curr_node.name in self.transposed_buffers and self.transposed_buffers[curr_node.name] != idx:
                        raise NotImplementedError()
                    self.transposed_buffers[curr_node.name] = idx
                    curr_node.name += "_transposed"
                    return simd_macros.mm256_store_ps(node.left, node.right)
                else:
                    return simd_macros.mm256_store_ps(node.left.left, node.right)
            node.left = self.visit(node.left)
            return node
        node.left = self.visit(node.left)
        node.right = self.visit(node.right)
        return node

    def visit_FunctionCall(self, node):
        node.args = [self.visit(a) for a in node.args]
        if node.func.name in ["fmax", "fmin"]:
            node.func.name = "_mm256_" + node.func.name[1:] + "_ps"
            args = []
            for arg in node.args:
                if isinstance(arg, C.Constant):
                    # args.append(C.SymbolRef("{" + ",".join(str(arg.value) for _ in range(latte.core.SIMDWIDTH)) + "}"))
                    args.append(C.FunctionCall(C.SymbolRef("_mm256_setzero_ps"), []))
                else:
                    args.append(arg)
            node.args = args
        return node

def vectorize_loop(ast, loopvar):
    transformer = Vectorizer(loopvar)
    ast = transformer.visit(ast)
    return ast, transformer.transposed_buffers


class FMAReplacer(ast.NodeTransformer):
    def visit_BinaryOp(self, node):
        if isinstance(node.op, C.Op.Add) and isinstance(node.right, C.BinaryOp) and \
            isinstance(node.right.op, C.Op.Mul):
                # FIXME: Check all are vector types
            return C.FunctionCall(C.SymbolRef("_mm256_fmadd_ps"), [node.right.left, node.right.right, node.left])
        node.left = self.visit(node.left)
        node.right = self.visit(node.right)
        return node


def fma_replace(ast):
    return FMAReplacer().visit(ast)
