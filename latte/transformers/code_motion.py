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
import latte.util as util
import ctree.c.nodes as C


class SymbolChecker(ast.NodeVisitor):
    def __init__(self, sym, fuse_map):
        self.flag = True
        self.sym = sym
        self.fuse_map = fuse_map

    def visit_BinaryOp(self, node):  
        a = node
        while isinstance(a, C.BinaryOp):
            
            self.visit(a.right)   
            a = a.left 
        if isinstance(a, C.SymbolRef):
            if a.name in self.fuse_map:
                self.flag = False

    def visit_SymbolRef(self, node):
        #print(node.name)
        if node.name != self.sym :
            #print (self.sym)
            #print (node.name)
            self.flag = False



def only_contains_symbol(ast, symbol, fuse_map):
        checker = SymbolChecker(symbol, fuse_map)
        checker.visit(ast)
        return checker.flag


class hoist_intermediate_invariants(ast.NodeTransformer):
    def __init__(self, fuse_map):
       self.fuse_map = fuse_map    
 
    def visit(self, node):
        node = super().visit(node)
        if hasattr(node, 'body'):
            node.body = util.flatten(node.body)
        return node



    def visit_FunctionDecl(self, node):
      new_defn = util.flatten([self.visit(s) for s in node.defn])
      node.defn = new_defn
      return node

    def visit_For(self, node):
        node.body = util.flatten([self.visit(s) for s in node.body])
        #if node.init.left.name == "_neuron_index_0":
        # Don't lift out of outer most loop
        #    return node
        pre_stmts = []
        new_body = []
        loop_var = node.init.left.name

        for stmt in node.body:
            if isinstance(stmt, C.BinaryOp) and isinstance(stmt.op, C.Op.Assign) and \
              isinstance(stmt.right, C.FunctionCall) and "_load" in stmt.right.func.name:
              hoist = True
              for arg in stmt.right.args:
                  if not(only_contains_symbol(arg, node.init.left.name,self.fuse_map)):
                      hoist = False
              if hoist: 
                  pre_stmts.append(stmt)
              else:
                  new_body.append(stmt)
            else:
                new_body.append(stmt)

        node.body = pre_stmts + new_body

        return node 


class InvariantLoadStoreLifter(ast.NodeTransformer):
    def visit(self, node):
        node = super().visit(node)
        if hasattr(node, 'body'):
            node.body = util.flatten(node.body)
        return node

    

    def visit_For(self, node):
        node.body = util.flatten([self.visit(s) for s in node.body])
        if node.init.left.name == "_neuron_index_0":
          #Don't lift out of outer most loop
          return node
        pre_stmts = []
        new_body = []
        post_stmts = []
        loop_var = node.init.left.name
        deps = set()
        for stmt in node.body:
            # print(astor.dump_tree(stmt))
            if isinstance(stmt, C.FunctionCall) and "_mm" in stmt.func.name and \
                "_store" in stmt.func.name and \
                not util.contains_symbol(stmt, loop_var) and \
                not any(util.contains_symbol(stmt, dep) for dep in deps):
                    post_stmts.append(stmt)
            elif isinstance(stmt, C.BinaryOp) and isinstance(stmt.op, C.Op.Assign) and \
                    isinstance(stmt.right, C.FunctionCall) and "_load" in stmt.right.func.name and \
                    not util.contains_symbol(stmt, loop_var) and \
                    not any(util.contains_symbol(stmt, dep) for dep in deps):
                pre_stmts.append(stmt)
            elif isinstance(stmt, C.BinaryOp) and \
                 isinstance(stmt.op, C.Op.Assign) and \
                 isinstance(stmt.left, C.SymbolRef) and \
                 stmt.left.type is not None and \
                    not util.contains_symbol(stmt, loop_var) and \
                    not any(util.contains_symbol(stmt, dep) for dep in deps):
                pre_stmts.append(stmt)
            else:
                new_body.append(stmt)
                if isinstance(stmt, C.BinaryOp) and \
                   isinstance(stmt.op, C.Op.Assign) and \
                   isinstance(stmt.left, C.SymbolRef) and \
                   stmt.left.type is not None:
                    deps.add(stmt.left.name)
        node.body = new_body
        return pre_stmts + [node] + post_stmts


def lift_intermediate_loads(ast,fuse_map):
    return hoist_intermediate_invariants(fuse_map).visit(ast)


def lift_invariant_load_stores(ast):
    return InvariantLoadStoreLifter().visit(ast)
