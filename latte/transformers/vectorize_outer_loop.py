import ast
import ctree.simd.macros as simd_macros
import ctree.c.nodes as C
import latte.util as util

class VectorizeOuterLoop(ast.NodeTransformer):
    def __init__(self, vectorized_buffers, loop_var):
        self.vectorized_buffers = vectorized_buffers
        self.loop_var = loop_var

    def visit_For(self, node):
        if not isinstance(node.init, list) and node.init.left.name == self.loop_var:
            node.test.right = C.Div(node.test.right, C.SymbolRef("SIMDWIDTH"))
        node.body = [self.visit(s) for s in node.body]
        return node

    def visit_AugAssign(self, node):
        node.value = self.visit(node.value)
        if util.contains_symbol(node.target, self.loop_var):
            return simd_macros.mm256_store_ps(
                    node.target,
                    C.BinaryOp(self.visit(node.target), node.op, node.value))
        elif isinstance(node.op, C.Op.Add) and isinstance(node.value, C.BinaryOp) and \
                isinstance(node.value.op, C.Op.Mul):
            # if not isinstance(node.target, C.SymbolRef):
            #     node.value = C.FunctionCall(C.SymbolRef("vsum"), [node.value])
            #     return node
            # else:
                return C.Assign(node.target, C.FunctionCall(C.SymbolRef("_mm256_fmadd_ps"), [node.value.left, node.value.right, node.target]))
        elif isinstance(node.op, C.Op.Add) and isinstance(node.value, C.FunctionCall):
            # TODO: Verfiy it's a vector intrinsic
            return C.Assign(node.target, C.FunctionCall(C.SymbolRef("_mm256_add_ps"), [node.value, node.target]))
        elif isinstance(node.target, C.BinaryOp) and isinstance(node.target.op, C.Op.ArrayRef):
            raise NotImplementedError()
        node.target = self.visit(node.target)
        return node

    def visit_FunctionCall(self, node):
        if "_mm" in node.func.name:
            return node
        node.args = [self.visit(arg) for arg in node.args]
        return node

    def visit_BinaryOp(self, node):
        if isinstance(node.op, C.Op.ArrayRef):
            if util.contains_symbol(node, self.loop_var):
                idx = 0
                curr_node = node
                while not isinstance(curr_node.right, C.SymbolRef) or \
                        curr_node.right.name != self.loop_var:
                    idx += 1
                    curr_node = curr_node.left
                while not isinstance(curr_node, C.SymbolRef):
                    curr_node = curr_node.left
                self.vectorized_buffers[curr_node.name] = idx
                return simd_macros.mm256_load_ps(
                        node)
            else:
                return simd_macros.mm256_set1_ps(node)
                
        node.left = self.visit(node.left)
        node.right = self.visit(node.right)
        return node

def vectorize_outer_loop(ast, loop_var):
    vectorized_buffers = {}
    ast = VectorizeOuterLoop(vectorized_buffers, loop_var).visit(ast)
    return ast, vectorized_buffers
