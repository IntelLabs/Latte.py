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

class LoopSimplifier(ast.NodeTransformer):
    def __init__(self):
        super().__init__()

    def visit_For(self, node):
        node.body = util.flatten([self.visit(s) for s in node.body])
        #TODO: assumption is that every loop starts with zero, not negative
        init = -1
        incr = -1
        test = -1
        if isinstance(node.init, C.BinaryOp) and \
           isinstance(node.init.op, C.Op.Assign) and \
           isinstance(node.init.left, C.SymbolRef) and \
           isinstance(node.init.right, C.Constant):
           init = node.init.right.value

        if isinstance(node.test, C.BinaryOp) and \
           isinstance(node.test.op, C.Op.Lt) and \
           isinstance(node.test.left, C.SymbolRef) and \
           isinstance(node.test.right, C.Constant):
           test = node.test.right.value

        if isinstance(node.incr, C.AugAssign) and \
           isinstance(node.incr.op, C.Op.Add) and \
           isinstance(node.incr.target, C.SymbolRef) and \
           isinstance(node.incr.value, C.Constant):
           incr = node.incr.value.value

        if init != -1 and test != -1 and incr != -1 and (init+incr) >= test:
          return [util.replace_symbol(node.init.left.name, C.Constant(init), s) for s in node.body]

        return node

def simplify_loops(ast):
     return LoopSimplifier().visit(ast)

