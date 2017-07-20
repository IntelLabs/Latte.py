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
import ctree.c.nodes as C
import astor
from copy import deepcopy

class SimpleFusion(ast.NodeTransformer):
    """
    Performs simple fusion of loops when to_source(node.iter) and to_source(node.target) are identical

    Does not perform dependence analysis
    """
    def __init__(self):
        self.seen = {}


    def visit_BinaryOp(self, node):
        node.left = self.visit(node.left)
        node.right = self.visit(node.right)
 
 
        if isinstance(node.op, C.Op.Assign) and isinstance(node.left, C.SymbolRef) and node.left.name not in self.seen and node.left.type is not None:
            self.seen[node.left.name] = node.left.type
        elif isinstance(node.op, C.Op.Assign) and isinstance(node.left, C.SymbolRef) and node.left.name in self.seen and node.left.type is not None:
            node.left.type = None
 
 
 
        node.right = self.visit(node.right)
        return node

    def visit(self, node):
     

        if hasattr(node, 'body'):
            curr = deepcopy(self.seen)
  
        node = super().visit(node)
        

        if hasattr(node, 'body'):
           #curr = deepcopy(self.seen)
           self.seen= {}


        if hasattr(node, 'body') and len(node.body) > 1:
            new_body = [node.body[0]]
            for statement in node.body[1:]:
                if isinstance(new_body[-1], ast.For) and \
                        isinstance(statement, ast.For) and \
                        astor.to_source(statement.iter) == astor.to_source(new_body[-1].iter) and \
                        astor.to_source(statement.target) == astor.to_source(new_body[-1].target):
                    new_body[-1].body.extend(statement.body)
                elif isinstance(new_body[-1], C.For) and \
                        isinstance(statement, C.For) and \
                        statement.init.codegen() == new_body[-1].init.codegen() and \
                        statement.incr.codegen() == new_body[-1].incr.codegen() and \
                        statement.test.codegen() == new_body[-1].test.codegen() and new_body[-1].parallel == statement.parallel:
                    # new_body[-1].body.extend(statement.body)
                    for stmt in statement.body:
                        add = True
                        #for seen in new_body[-1].body:
                        #    if stmt.codegen() == seen.codegen():
                        #        add = False
                        #        break
                        if add:
                            new_body[-1].body.append(stmt)
                else:
                    new_body.append(statement)
            node.body = [self.visit(s) for s in new_body]

        if hasattr(node, 'body'):
           #curr = deepcopy(self.seen) 
           self.seen= curr



        return node

    def visit_FunctionDecl(self, node):
        new_body = [node.defn[0]]
        for statement in node.defn[1:]:
            if isinstance(new_body[-1], ast.For) and \
                    isinstance(statement, ast.For) and \
                    astor.to_source(statement.iter) == astor.to_source(new_body[-1].iter) and \
                    astor.to_source(statement.target) == astor.to_source(new_body[-1].target):
                new_body[-1].body.extend(statement.body)
            elif isinstance(new_body[-1], C.For) and \
                    isinstance(statement, C.For) and \
                    statement.init.codegen() == new_body[-1].init.codegen() and \
                    statement.incr.codegen() == new_body[-1].incr.codegen() and \
                    statement.test.codegen() == new_body[-1].test.codegen():
                if hasattr(statement, 'pre_trans') and len(statement.pre_trans) > 0:
                    new_body.append(statement)
                elif new_body[-1].pragma is not None and "collapse" in new_body[-1].pragma:
                    if hasattr(new_body[-1], 'pre_trans'):
                        pre_trans = new_body[-1].pre_trans
                        new_body[-1].pre_trans = None
                    else:
                        pre_trans = None
                    if hasattr(new_body[-1], 'reduce_vars'):
                        reduce_vars = new_body[-1].reduce_vars
                        new_body[-1].reduce_vars = None
                    else:
                        reduce_vars = None
                    candidate_node = deepcopy(new_body[-1])
                    candidate_node.body.extend(statement.body)
                    candidate_node = self.visit(candidate_node)
                    if len(candidate_node.body) == 1:
                        new_body[-1].body.extend(statement.body)
                        if hasattr(statement, 'pre_trans'):
                            if pre_trans is not None: 
                                pre_trans.extend(statement.pre_trans)
                            else:
                                pre_trans = statement.pre_trans
                        if pre_trans is not None:
                            new_body[-1].pre_trans = pre_trans 
                        if hasattr(statement, 'reduce_vars'):
                            if reduce_vars is not None: 
                                reduce_vars.extend(statement.reduce_vars)
                            else:
                                reduce_vars = statement.reduce_vars
                        if reduce_vars is not None:
                            new_body[-1].reduce_vars = reduce_vars 
                    else:
                        if pre_trans is not None:
                            new_body[-1].pre_trans = pre_trans 
                        if reduce_vars is not None:
                            new_body[-1].reduce_vars = reduce_vars 
                        new_body.append(statement)
                else:
                    if hasattr(new_body[-1], 'pre_trans') and hasattr(statement, 'pre_trans'):
                        new_body[-1].pre_trans.extend(statement.pre_trans)
                    elif hasattr(statement, 'pre_trans'):
                        new_body[-1].pre_trans = statement.pre_trans
                    if hasattr(new_body[-1], 'reduce_vars') and hasattr(statement, 'reduce_vars'):
                        new_body[-1].reduce_vars.extend(statement.reduce_vars)
                    elif hasattr(statement, 'reduce_vars'):
                        new_body[-1].reduce_vars = statement.reduce_vars
                    new_body[-1].body.extend(statement.body)
            else:
                new_body.append(statement)
        node.defn = [self.visit(s) for s in new_body]
        return node

def simple_fusion(ast):
    return SimpleFusion().visit(ast)
