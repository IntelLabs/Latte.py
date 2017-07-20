'''
Copyright (c) 2015, Intel Corporation

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright notice,
      this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of Intel Corporation nor the names of its contributors
      may be used to endorse or promote products derived from this software
      without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
'''
import ast
import ctree.simd.macros as simd_macros
import ctree.c.nodes as C
import latte.util as util

class VectorizeOuterLoop(ast.NodeTransformer):
    def __init__(self, vectorized_buffers, loop_var, vectorize):
        self.vectorized_buffers = vectorized_buffers
        self.loop_var = loop_var
        self.vectorize = vectorize

    def visit_For(self, node):
        # if node.init.left.name == self.loop_var:
            # node.test.right = C.Div(node.test.right, C.SymbolRef("SIMDWIDTH"))
            # node.test.right.value = node.test.right.value // 8
        node.body = [self.visit(s) for s in node.body]
        return node

    def visit_AugAssign(self, node):
        node.value = self.visit(node.value)
        if not self.vectorize:
            node.target = self.visit(node.target)
            return node
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
                if self.vectorize:
                    return simd_macros.mm256_load_ps(
                            node)
                else:
                    return C.ArrayRef(node, C.SymbolRef("_neuron_index_1_inner"))
            else:
                if self.vectorize:
                    return simd_macros.mm256_set1_ps(node)
                else:
                    return node
                
        node.left = self.visit(node.left)
        node.right = self.visit(node.right)
        return node

def vectorize_outer_loop(ast, loop_var, vectorize):
    vectorized_buffers = {}
    ast = VectorizeOuterLoop(vectorized_buffers, loop_var, vectorize).visit(ast)
    return ast, vectorized_buffers
