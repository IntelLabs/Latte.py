import ast
import ctree.simd.macros as simd_macros
import ctree.c.nodes as C
import latte.util as util

class VectorizeOuterLoop(ast.NodeTransformer):
    def __init__(self, vectorized_buffers):
        self.vectorized_buffers = vectorized_buffers

    def visit_For(self, node):
        if not isinstance(node.init, list) and node.init.left.name == "_neuron_index_1":
            node.test.right = C.Div(node.test.right, C.SymbolRef("SIMDWIDTH"))
        node.body = [self.visit(s) for s in node.body]
        return node

    def visit_AugAssign(self, node):
        node.value = self.visit(node.value)
        if util.contains_symbol(node.target, "_neuron_index_1"):
            return simd_macros.mm256_store_ps(
                    node.target,
                    C.BinaryOp(self.visit(node.target), node.op, node.value))
        elif isinstance(node.op, C.Op.Add) and isinstance(node.value, C.BinaryOp) and \
                isinstance(node.value.op, C.Op.Mul):
            return C.Assign(node.target, C.FunctionCall(C.SymbolRef("_mm256_fmadd_ps"), [node.value.left, node.value.right, node.target]))
        elif isinstance(node.op, C.Op.Add) and isinstance(node.value, C.FunctionCall):
            # TODO: Verfiy it's a vector intrinsic
            return C.Assign(node.target, C.FunctionCall(C.SymbolRef("_mm256_add_ps"), [node.value, node.target]))
        elif isinstance(node.target, C.BinaryOp) and isinstance(node.target.op, C.Op.ArrayRef):
            raise NotImplementedError()
        node.target = self.visit(node.target)
        return node


    def visit_BinaryOp(self, node):
        if isinstance(node.op, C.Op.ArrayRef):
            if util.contains_symbol(node, "_neuron_index_1"):
                idx = 0
                curr_node = node
                while not isinstance(curr_node.right, C.SymbolRef) or \
                        curr_node.right.name != "_neuron_index_1":
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

def vectorize_outer_loop(ast):
    vectorized_buffers = {}
    ast = VectorizeOuterLoop(vectorized_buffers).visit(ast)
    return ast, vectorized_buffers
