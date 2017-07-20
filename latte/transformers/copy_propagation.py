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

import latte.util as util
import ctypes
import ast
import ctree.c.nodes as C
from copy import deepcopy

du_map = {}
class ReplaceSymbolRef(ast.NodeTransformer):
    def __init__(self):
        super().__init__()

    def visit(self, node):
        return super().visit(node)

    def visit_SymbolRef(self, node):
        if node.name in du_map:
          return du_map[node.name]
        return node

class CopyPropagation(ast.NodeTransformer):
    def __init__(self):
        super().__init__()

    def visit_For(self, node):
        node.body = util.flatten([s for s in node.body])
        new_body = []
        for stmt in node.body:
          if isinstance(stmt, C.BinaryOp) and \
             isinstance(stmt.op, C.Op.Assign) and \
             isinstance(stmt.left, C.SymbolRef) and \
             (stmt.left.name.startswith("in_") or stmt.left.name.startswith("_input_")) and \
             not isinstance(stmt.right, C.FunctionCall):
               new_body.append(stmt)
               if isinstance(stmt.right, C.SymbolRef) and \
                 stmt.right.name in du_map:
                 du_map[stmt.left.name] = du_map[stmt.right.name]
               else:
                 du_map[stmt.left.name] = stmt.right

          elif isinstance(stmt, C.BinaryOp) and \
             isinstance(stmt.op, C.Op.Assign) and \
             isinstance(stmt.left, C.SymbolRef) and \
             isinstance(stmt.right, C.FunctionCall) and "_mm" in stmt.right.func.name \
             and ("_load_" in stmt.right.func.name or "_set1" in stmt.right.func.name or "_broadcast" in stmt.right.func.name):
               stmt = ReplaceSymbolRef().visit(stmt)
               new_body.append(stmt)
          elif isinstance(stmt, C.FunctionCall) and "_mm" in stmt.func.name and "_store" in stmt.func.name:
               stmt = ReplaceSymbolRef().visit(stmt)
               new_body.append(stmt)
          else:
               new_body.append(stmt)
        node.body = util.flatten([self.visit(s) for s in new_body])
        return node

def propagate_copies(ast):
     du_map = {}
     return CopyPropagation().visit(ast)

