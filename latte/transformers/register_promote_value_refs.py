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
import math
import ctree.c.nodes as C
import ctypes
import ctree.simd as simd
import ctree.simd.macros as simd_macros
from ctree.transformations import PyBasicConversions
import latte.util as util

class RegisterPromoteValueRefs(ast.NodeTransformer):
    def __init__(self, ensemble, direction, batch_size, target_loop_var):
        self.ensemble = ensemble
        self.target = "value" if direction == "forward" else "grad"
        self.batch_size = batch_size
        self.seen = {}
        self._vars = []
        self.target_loop_var = target_loop_var

    def visit_BinaryOp(self, node):
        """
        Promote array reference ensemble_name$field[...] to register reference
        $field

        $field is either "value" or "grad" depending on the current direction
        """
        if isinstance(node.op, C.Op.ArrayRef):
            if node.codegen() in self.seen:
                return C.SymbolRef(self.seen[node.codegen()][0])
            curr_node = node
            while not isinstance(curr_node, C.SymbolRef):
                curr_node = curr_node.left
            if curr_node.name.endswith(self.target):
                var = self.target + "_" + str(len(self.seen))
                self.seen[node.codegen()] = (var, node)
                return C.SymbolRef(var)
        node.left = self.visit(node.left)
        node.right = self.visit(node.right)
        return node

    def visit_For(self, node):
        """
        Find the innermost loop to insert a load and store of the target register

        target is either "value" or "grad" depending on direction
        """
        node.body = [self.visit(s) for s in node.body]
        if node.init.left.name == self.target_loop_var:
            for var, seen in self.seen.values():
                node.body.insert(0,
                    C.Assign(
                        C.SymbolRef(var, ctypes.c_float()), 
                        seen
                    ))

            # we only store the value register as "grad" is only read by definition
            if self.target == "value":
                for var, seen in self.seen.values():
                    node.body.append( 
                        C.Assign(
                            seen,
                            C.SymbolRef(var)
                        ))
        return node

def register_promote_value_refs(ast, ensemble, direction, batch_size, target_loop_var):
    return RegisterPromoteValueRefs(ensemble, direction, batch_size, target_loop_var).visit(ast)
