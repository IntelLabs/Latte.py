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
import ctypes
import ctree
import copy
import ctree.np
import ctree.c.nodes as C

class BasicTypeInference(ast.NodeTransformer):
    def __init__(self):
        self.seen = {}

    def visit_SymbolRef(self, node):
        if node.type is not None:
            self.seen[node.name] = node.type
        return node

    def visit(self, node):
        if hasattr(node, 'body'):
            #raise NotImplementedError(ast.dump(node))
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
            if node.func.name in ["MAX", "MIN", "max", "min", "floor", "pow"]:
                return ctree.types.get_common_ctype([self._get_type(a) for a in node.args])
        elif isinstance(node, C.Cast):
            return node.type
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

def type_infer(ast):
    try:
        
        transformer = BasicTypeInference()
        ast = transformer.visit(ast)
        return (ast, transformer.seen)  
    except NotImplementedError as e:
        print("AST that caused exception during type inference")
        print(ast)
        raise e
