import ast
import ctree
import ctree.c.nodes as C
import ctree.simd.macros as simd_macros
import latte
import latte.util as util
import ctree.simd as simd
import ctypes
from copy import deepcopy
from ctree.templates.nodes import StringTemplate

class RemoveIndexExprs(ast.NodeTransformer):
    def __init__(self, var):
        self.var = var

    def visit_SymbolRef(self, node):
        if node.name == self.var:
            return C.Constant(0)
        return node

class Vectorizer(ast.NodeTransformer):
    def __init__(self, loop_var):
        self.loop_var = loop_var
        self.transposed_buffers = {}
        self.symbol_table = {}

    def visit(self, node):
        node = super().visit(node)
        if hasattr(node, "body"):
            node.body = util.flatten(node.body)
        return node

    def visit_For(self, node):
        node.body = [self.visit(s) for s in node.body]
        if node.init.left.name == self.loop_var:
            assert node.test.right.value == latte.core.SIMDWIDTH
            index = C.Assign(
                    C.SymbolRef(node.init.left.name, ctypes.c_int()),
                    C.Constant(0)
                )
            return [index] + [RemoveIndexExprs(self.loop_var).visit(s) for s in node.body]
            if isinstance(node.incr, C.UnaryOp):
                node.incr = C.AddAssign(node.incr.arg, C.Constant(latte.core.SIMDWIDTH))
            else:
                node.incr.value = C.Constant(latte.core.SIMDWIDTH)
        return node

    def visit_AugAssign(self, node):
        node.value = self.visit(node.value)
        if util.contains_symbol(node.target, self.loop_var):
            if not util.contains_symbol(node.target.right, self.loop_var):
                target = self.visit(deepcopy(node.target))
                curr_node = node.target
                idx = 1
                while curr_node.left.right.name != self.loop_var:
                    curr_node = curr_node.left
                    idx += 1
                curr_node.left = curr_node.left.left
                node.target = C.ArrayRef(node.target, C.SymbolRef(self.loop_var))
                while not isinstance(curr_node, C.SymbolRef):
                    curr_node = curr_node.left
                if curr_node.name in self.transposed_buffers and self.transposed_buffers[curr_node.name] != idx:
                    raise NotImplementedError()
                self.transposed_buffers[curr_node.name] = idx
                curr_node.name += "_transposed"
                if isinstance(node.target.right, C.Constant) and node.target.value == 0.0:
                    return simd_macros.mm256_store_ps(
                            node.target.left,
                            C.BinaryOp(target, node.op, node.value))
                else:
                    return simd_macros.mm256_store_ps(
                            C.Ref(node.target),
                            C.BinaryOp(target, node.op, node.value))
            else:
                if isinstance(node.target.right, C.Constant) and node.target.value == 0.0:
                    return simd_macros.mm256_store_ps(
                            node.target.left,
                            C.BinaryOp(self.visit(node.target), node.op, node.value))
                else:
                    return simd_macros.mm256_store_ps(
                            C.Ref(node.target),
                            C.BinaryOp(self.visit(node.target), node.op, node.value))
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
                if not util.contains_symbol(node.right, self.loop_var):
                    curr_node = node
                    idx = 1
                    while curr_node.left.right.name != self.loop_var:
                        curr_node = curr_node.left
                        idx += 1
                    curr_node.left = curr_node.left.left
                    node = C.ArrayRef(node, C.SymbolRef(self.loop_var))
                    while not isinstance(curr_node, C.SymbolRef):
                        curr_node = curr_node.left
                    if curr_node.name in self.transposed_buffers and self.transposed_buffers[curr_node.name] != idx:
                        raise NotImplementedError()
                    self.transposed_buffers[curr_node.name] = idx
                    curr_node.name += "_transposed"
                if isinstance(node.right, C.Constant) and node.target.value == 0.0:
                    return simd_macros.mm256_load_ps(node.left)
                else:
                    return simd_macros.mm256_load_ps(C.Ref(node))
            else:
                return C.FunctionCall(C.SymbolRef("_mm256_broadcast_ss"), [C.Ref(node)])
        elif isinstance(node.op, C.Op.Assign):
            node.right = self.visit(node.right)
            if isinstance(node.right, C.FunctionCall) and \
                    node.right.func.name in ["_mm256_load_ps", "_mm256_broadcast_ss"] and \
                    isinstance(node.left, C.SymbolRef) and node.left.type is not None:
                node.left.type = simd.types.m256()
                self.symbol_table[node.left.name] = node.left.type
                return node
            elif isinstance(node.left, C.BinaryOp) and util.contains_symbol(node.left, self.loop_var):
                if node.left.right.name != self.loop_var:
                    curr_node = node
                    idx = 1
                    while curr_node.left.right.name != self.loop_var:
                        curr_node = curr_node.left
                        idx += 1
                    curr_node.left = curr_node.left.left
                    node = C.ArrayRef(node, C.SymbolRef(self.loop_var))
                    while not isinstance(curr_node, C.SymbolRef):
                        curr_node = curr_node.left
                    if curr_node.name in self.transposed_buffers and self.transposed_buffers[curr_node.name] != idx:
                        raise NotImplementedError()
                    self.transposed_buffers[curr_node.name] = idx
                    curr_node.name += "_transposed"
                    # return simd_macros.mm256_store_ps(C.Ref(node.left), node.right)
                if isinstance(node.left.right, C.Constant) and node.target.value == 0.0:
                    return simd_macros.mm256_store_ps(node.left.left, node.right)
                else:
                    return simd_macros.mm256_store_ps(C.Ref(node.left), node.right)
            node.left = self.visit(node.left)
            return node
        elif isinstance(node.left, C.Constant) and self._is_vector_type(node.right):
            node.left = C.FunctionCall(C.SymbolRef("_mm256_set1_ps"), [C.Cast(ctypes.c_float(), node.left)])
        node.left = self.visit(node.left)
        node.right = self.visit(node.right)
        return node

    def _is_vector_type(self, node):
        return node.name in self.symbol_table and isinstance(self.symbol_table[node.name], simd.types.m256)

    def visit_FunctionCall(self, node):
        node.args = [self.visit(a) for a in node.args]
        if node.func.name in ["fmax", "fmin"]:
            node.func.name = "_mm256_" + node.func.name[1:] + "_ps"
            args = []
            for arg in node.args:
                if isinstance(arg, C.Constant) and arg.value == 0:
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

def fuse_multiply_adds(ast):
    return FMAReplacer().visit(ast)


class VectorLoadReplacer(ast.NodeTransformer):
    def __init__(self, load_stmt, new_stmt):
        self.load_stmt = load_stmt
        self.new_stmt = new_stmt

    def visit_FunctionCall(self, node):
        if node.codegen() == self.load_stmt:
            return self.new_stmt
        node.args = [self.visit(arg) for arg in node.args]
        return node


class VectorLoadCollector(ast.NodeVisitor):
    def __init__(self):
        super().__init__()
        self.loads = {}

    def visit(self, node):
        # Don't descend into nested expressions
        if hasattr(node, 'body'):
            return
        super().visit(node)

    def visit_FunctionCall(self, node):
        if "_mm" in node.func.name and ("_load_" in node.func.name or "_set1" in node.func.name):
            if node.codegen() not in self.loads:
                self.loads[node.args[0].codegen()] = [node.args[0], 0, node.func.name]
            self.loads[node.args[0].codegen()][1] += 1
        [self.visit(arg) for arg in node.args]

class VectorLoadStoresRegisterPromoter(ast.NodeTransformer):
    _tmp = -1
    def _gen_register(self):
        VectorLoadStoresRegisterPromoter._tmp += 1
        return "___x" + str(self._tmp)

    def visit(self, node):
        node = super().visit(node)
        if hasattr(node, 'body'):
            # [collector.visit(s) for s in node.body]
            new_body = []
            seen = {}
            stores = []
            collector = VectorLoadCollector()
            for s in node.body:
                collector.visit(s)
                for stmt in collector.loads.keys():
                    if stmt not in seen:
                        reg = self._gen_register()
                        load_node, number, func = collector.loads[stmt]
                        seen[stmt] = (reg, load_node, func)
                        new_body.append(C.Assign(C.SymbolRef(reg, ctree.simd.types.m256()),
                                                 C.FunctionCall(C.SymbolRef(func), [load_node])))
                if isinstance(s, C.FunctionCall) and "_mm" in s.func.name and "_store" in s.func.name:
                    if s.args[0].codegen() in seen:
                        stores.append((s.args[0], seen[s.args[0].codegen()][0]))
                        s = C.Assign(C.SymbolRef(seen[s.args[0].codegen()][0]), s.args[1])
                for stmt in seen.keys():
                    reg, load_node, func = seen[stmt]
                    replacer = VectorLoadReplacer(
                            C.FunctionCall(C.SymbolRef(func), [load_node]).codegen(), 
                            C.SymbolRef(reg))
                    s = replacer.visit(s)
                new_body.append(s)
            for target, value in stores:
                new_body.append(C.FunctionCall(C.SymbolRef("_mm256_store_ps"), [target, C.SymbolRef(value)]))
            node.body = util.flatten(new_body)
        return node

def register_promote_vector_loads_stores(ast):
    return VectorLoadStoresRegisterPromoter().visit(ast)
