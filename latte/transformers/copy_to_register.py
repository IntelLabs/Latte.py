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
import latte
import latte.util as util
import ctypes
import ast
import ctree.c.nodes as C
from copy import deepcopy
import ctree.simd as simd
du_map = []
#replace_map ={}
sym_map = {} 
def get_simd_type():
    return {
        "AVX": simd.types.m256,
        "AVX-2": simd.types.m256,
        "AVX-512": simd.types.m512,
    }[latte.config.vec_config]

def inReplaceMapSource(ast, replace_map): 


    if(isinstance(ast, C.UnaryOp)):
        if isinstance(ast.arg, C.BinaryOp):
          node = ast.arg
          
          assert(isinstance(node.op, C.Op.ArrayRef))
          while(isinstance(node, C.BinaryOp)):
              node = node.left

          assert(isinstance(node, C.SymbolRef))
        
          for i in replace_map:
            if str(replace_map[i]) == str( node.name):
               return True      
 
    return False



class ReplaceName(ast.NodeTransformer):
    def __init__(self, old, new):
        self.old = old
        self.new = new
 
    def visit_SymbolRef(self, node):
        if node.name == self.old.name:
            return self.new
        return node
 
def replace_name(old, new, ast):
    return ReplaceName(old, new).visit(ast)



 
def construct_arr_reference(sym, ref):
    
      assert isinstance(ref[0], C.UnaryOp)

      assert isinstance(ref[0].arg, C.BinaryOp)
      assert isinstance(ref[0].arg.op,C.Op.ArrayRef)
      node = ref[0].arg 
      while(isinstance(node, C.BinaryOp)):
              node = node.left
 
      assert(isinstance(node, C.SymbolRef))
      replace_name(node,C.SymbolRef(sym), ref[0])

      return ref[0].arg

def inReplaceMapSink(ast, replace_map):
      if(isinstance(ast, C.UnaryOp)):
        if isinstance(ast.arg, C.BinaryOp):
          node = ast.arg
 
          assert(isinstance(node.op, C.Op.ArrayRef))
          while(isinstance(node, C.BinaryOp)):
             node = node.left
 
          assert(isinstance(node, C.SymbolRef))
 
          assert(isinstance(node, C.SymbolRef))
          if node.name  in replace_map:
                 return True
      
 
      return False

def get_alias(ast, replace_map):

      assert isinstance(ast[0], C.UnaryOp)
 
      assert isinstance(ast[0].arg, C.BinaryOp)
      assert isinstance(ast[0].arg.op,C.Op.ArrayRef)
      node = ast[0].arg

      while(isinstance(node, C.BinaryOp)):
              node = node.left
 
      assert(isinstance(node, C.SymbolRef))
      

      if node.name in replace_map:
        
              return replace_map[node.name]

      return None

def extract_reference(ast):
   
   assert(len(ast) == 2)
   assert(isinstance(ast[0], C.UnaryOp))
   assert(isinstance(ast[0].arg, C.BinaryOp))
   node = ast[0].arg
   



   return(node, ast[1])  


def checkEquivalence(a, b):
    
    if not (len(a) == len(b)):
      return False

    for i in a:
      if i not in b:
         return False
      if not (a[i] == b[i]):
        return False  

    return True

def ref_equal_helper(a):

  coeff_map = {}

  if(isinstance(a, C.BinaryOp)):
      temp_map_1 = ref_equal_helper(a.left)
      temp_map_2 = ref_equal_helper(a.right)             
    
      if temp_map_1 is None or temp_map_2 is None:
          return None
      if isinstance(a.op, C.Op.Add):
          for i in temp_map_1:
            if i  in temp_map_2:
               coeff_map[i] = temp_map_1[i] + temp_map_2[i]
            else:
               coeff_map[i] = temp_map_1[i]
          for i in temp_map_2:
             #print(i) 
             if i  not in temp_map_1:
               #print("entered\n")  
               coeff_map[i] = temp_map_2[i]

            
          return coeff_map


      elif isinstance(a.op, C.Op.Mul):
         for i in temp_map_1:
            for j in temp_map_2:
                if i == 0 and j == 0:
                   coeff_map[i] = temp_map_1[i]*temp_map_2[j]  
                elif i ==0 and j != 0:
                   if j in coeff_map: 
                      coeff_map[j] = coeff_map[j] + temp_map_2[j]*temp_map_1[i]
                   else: 
                      coeff_map[j] = temp_map_2[j]*temp_map_1[i]  
                elif i != 0 and j == 0:            
                   if i in coeff_map:
                      coeff_map[i] = coeff_map[i] + temp_map_2[j]*temp_map_1[i]
                   else:
                      coeff_map[i] = temp_map_2[j]*temp_map_1[i]
                else:
                    if(''.join(sorted(str(i) + str(j))) in coeff_map):
                       coeff_map[''.join(sorted(str(i) + str(j)))] = coeff_map[''.join(sorted(str(i) + str(j)))]+ temp_map_1[i]*temp_map_2[j]
                    else:
                       coeff_map[''.join(sorted(str(i) + str(j)))] = temp_map_1[i]*temp_map_2[j]
 
         return coeff_map
      else:
          return None
  

  elif isinstance(a, C.SymbolRef):
      coeff_map[a.name] = 1
      return coeff_map
  elif isinstance(a, C.Constant):
      coeff_map[0] = a.value      
      return coeff_map
   
  else:
      return None



def ref_equal(a, b):
     
    node1 = a
    node2 = b
    assert(isinstance(node1.op,C.Op.ArrayRef))
    assert(isinstance(node2.op,C.Op.ArrayRef))
    while(isinstance(node1, C.BinaryOp)):
        if not isinstance(node2, C.BinaryOp):
            return False 
       
        one = ref_equal_helper(node1.right)
        two = ref_equal_helper(node2.right)       
    
 
        if (one is None or two is None):
            return False
        if (not checkEquivalence(one, two)):
            return False
        
        node1 = node1.left
        node2 = node2.left    
    return True

           
      
    


def store_in_du_map(sym_arr_ref):

     found = False
     
     for i in range(len(du_map)):
        if ref_equal(du_map[i][0], sym_arr_ref[0]):
            entry = (du_map[i][0], sym_arr_ref[1]) 
            index = i             
            found = True
            break
     

     if (found):
        du_map.pop(index)   
        du_map.append(entry)    
     else: 
        du_map.append((sym_arr_ref[0],sym_arr_ref[1] )) 
        
     return

def in_du_map(sym_arr_ref):

    for i in range(len(du_map)):
        if ref_equal(du_map[i][0], sym_arr_ref):
          #print(du_map[i][0])  
          #print(sym_arr_ref)
          return True

    return False


def get_register(sym_arr_ref):
    for i in range(len(du_map)):
        if ref_equal(du_map[i][0], sym_arr_ref):
            return C.SymbolRef(du_map[i][1])
            
 
    return None

class ReplaceSymbolRef(ast.NodeTransformer):
    def __init__(self):
        super().__init__()

    def visit(self, node):
        return super().visit(node)

    def visit_SymbolRef(self, node):
        if node.name in du_map:
          return du_map[node.name]
        return node

class RegisterCopy(ast.NodeTransformer):
    def __init__(self, map_):
        self.replace_map = map_
        self.seen = {}
        self._tmp = 0
    def _gen_register(self):
        self._tmp += 1
        return "___z" + str(self._tmp)

    def visit_SymbolRef(self, node):
        if node.type is not None:
            self.seen[str(node.name)] = node.type


        if node.name in sym_map:
            return sym_map[node.name]
        
        
        return node
 

    def visit(self, node):
        if hasattr(node, 'body'):
            #raise NotImplementedError(ast.dump(node))
            curr = deepcopy(self.seen)
        node = super().visit(node)
        if hasattr(node, 'body'):
            self.seen = curr
        return node



    def visit_For(self, node):
        node.body = util.flatten([s for s in node.body])
        new_body = []
        for stmt in node.body:
          if isinstance(stmt, C.FunctionCall) and "_mm" in stmt.func.name \
             and "_store" in stmt.func.name and inReplaceMapSource(stmt.args[0], self.replace_map):
                  
                  if isinstance(stmt.args[1], C.SymbolRef):
                    sym_arr_ref = extract_reference(stmt.args)  
                    store_in_du_map(sym_arr_ref)  
                    reg = stmt.args[1]
                    self.seen[reg.name] = None
                    new_body.append(stmt)

                  elif isinstance(stmt.args[1], C.FunctionCall) and "_mm" in stmt.func.name:
                      tmp = self._gen_register()
                      new_body.append(C.Assign(C.SymbolRef(tmp, get_simd_type()()), deepcopy(stmt.args[1])))
                      new_body.append(C.FunctionCall(C.SymbolRef(stmt.func.name),  [stmt.args[0],C.SymbolRef(tmp, None)]))
                      sym_arr_ref = extract_reference(C.FunctionCall(C.SymbolRef(stmt.func.name),  [stmt.args[0],C.SymbolRef(tmp, None)]).args)  
                      store_in_du_map(sym_arr_ref)
                  # if stmt.args[0].type:
                  #    self.seen[reg.name] = stmt.args[0].type     
                  #else:
                      self.seen[tmp] = None

          elif isinstance(stmt, C.BinaryOp) and \
             isinstance(stmt.op, C.Op.Assign) and \
             isinstance(stmt.left, C.SymbolRef) and \
             isinstance(stmt.right, C.FunctionCall) and "_mm" in stmt.right.func.name and "_load" in stmt.right.func.name and inReplaceMapSink(stmt.right.args[0], self.replace_map): 
                  #print(stmt.right.args[0])                         
                  source = get_alias(stmt.right.args, self.replace_map)
                  #print(source)      
                  if (source is not None):
                    sym_arr_ref = construct_arr_reference(source, deepcopy(stmt.right.args))
                    if in_du_map(sym_arr_ref):
                       reg = get_register(sym_arr_ref)
                       #print(reg.name)   
                       if str(reg.name) in self.seen: 
                          #print(reg.name)  
                          sym_map[stmt.left.name] = reg
                       else:
                          new_body.append(stmt) 
                    else:
                       new_body.append(stmt)    
                  else:
                      new_body.append(stmt)
                            
          else:
              new_body.append(stmt)  
        node.body = util.flatten([self.visit(s) for s in new_body])
        return node

def register_copy(ast, map_):
     #replace_map = map_   
     #print(map_)
     return RegisterCopy(map_).visit(ast)

