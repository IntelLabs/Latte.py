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
#from copy import deepcopy
import ctypes

class Timer(ast.NodeTransformer):
    def visit_FunctionDecl(self, node):
        new_body = []
        count = 0
        for statement in node.defn:
                   
            if isinstance(statement, ast.For) or isinstance(statement, C.For):
                pre =  C.SubAssign(C.ArrayRef(C.SymbolRef('times'), C.Constant(count)),C.FunctionCall('omp_get_wtime', []))
                post =  C.AddAssign(C.ArrayRef(C.SymbolRef('times'), C.Constant(count)),C.FunctionCall('omp_get_wtime', []))
                new_body.append(pre)
                new_body.append(statement)
                new_body.append(post)
                count = count + 1
            else:
                new_body.append(statement)
        
        memset = C.Assign(C.SymbolRef('times'), C.FunctionCall(C.SymbolRef('calloc_doubles'),[C.Constant(count)]))
        new_body.insert(0,  memset)
        new_body.insert(0, C.Assign(C.SymbolRef("*times", ctypes.c_double()), C.Constant(0)))
        for i in range(0,count):
          print_stmt = C.FunctionCall(C.SymbolRef('printf'),[C.String("\ttimes[%d] = %g\\n"), C.Constant(i), C.ArrayRef(C.SymbolRef('times'), C.Constant(i))])
          new_body.append(print_stmt)
        node.defn = new_body     
        return node

def timer(ast):
    return Timer().visit(ast)
