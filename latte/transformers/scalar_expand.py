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
import ctree
import ctree.c.nodes as C
import ctree.simd.macros as simd_macros
import latte
import latte.util as util
import ctree.simd as simd
import ctypes
from copy import deepcopy
from ctree.templates.nodes import StringTemplate

def broadcast_ss(arg, type_):
   

    if isinstance(type_, ctypes.c_int):
        if latte.config.vec_config == "AVX-512":
        # AVX-512 doesn't support broadcast, use set1_ps and remove Ref node
          if isinstance(arg, C.UnaryOp) and isinstance(arg.op, C.Op.Ref):
            arg = arg.arg
        return C.FunctionCall(C.SymbolRef({
        "AVX-2": "_mm256_broadcastd_epi32",
        "AVX-512": "_mm512_set1_epi32",
    }[latte.config.vec_config]), [arg])


    else:
      if latte.config.vec_config == "AVX-512":
        # AVX-512 doesn't support broadcast, use set1_ps and remove Ref node
        if isinstance(arg, C.UnaryOp) and isinstance(arg.op, C.Op.Ref):
            arg = arg.arg
      return C.FunctionCall(C.SymbolRef({
        "AVX": "_mm256_broadcast_ss",
        "AVX-2": "_mm256_broadcast_ss",
        "AVX-512": "_mm512_set1_ps",
    }[latte.config.vec_config]), [arg])

def get_simd_type(arg):
    
    if isinstance(arg, ctypes.c_int):
       return {
        "AVX-2": simd.types.m256i,
        "AVX-512": simd.types.m512i,
    }[latte.config.vec_config]
    else:
       return {
        "AVX": simd.types.m256,
        "AVX-2": simd.types.m256,
        "AVX-512": simd.types.m512,
    }[latte.config.vec_config]


def gen_selector_type():
 
      return {
        "AVX-2": simd.types.mask8,
        "AVX-512": simd.types.mask16,
    }[latte.config.vec_config]


def is_vector_type(var, type_map, symbol_map):

    if not isinstance(var, C.FunctionCall) and var.type is not None:
        if isinstance(var.type,simd.types.m256) or isinstance(var.type,simd.types.m256i)  or isinstance(var.type,simd.types.m512) or  isinstance(var.type,simd.types.m512i): 
            return True 
   
    elif isinstance(var, C.SymbolRef) and var.name in type_map:
        var_type = type_map[var.name]
          
        if isinstance(var_type,simd.types.m256) or isinstance(var_type,simd.types.m256i) or  isinstance(var_type,simd.types.m512) or isinstance(var_type,simd.types.m512i):
            return True
    elif isinstance(var, C.SymbolRef) and var.name in symbol_map:
        var_type = symbol_map[var.name]
        if isinstance(var_type,simd.types.m256) or isinstance(var_type,simd.types.m256i) or  isinstance(var_type,simd.types.m512) or  isinstance(var_type,simd.types.m512i):
            return True
  
    elif isinstance(var, C.FunctionCall) : 
        var_name = var.func.name
        if "_store" in var_name or "_load" in var_name or  "_broadcast" in var_name or "_set1" in var_name:
            return True  
    assert(False)
    return False


def get_type(src1,type_map, symbol_map):
   src1_type = None 
   if not isinstance(src1, C.FunctionCall) and src1.type is not None :    
     # if isinstance(src1.type,simd.types.m256) and isinstance(src2.type,simd.types.m256) 
     #   return C.Assign(dest, C.FunctionCall("_mm256_cmp_ps_mask", [src1, src2, C.SymbolRef("_MM_CMPINT_GT", None)]))
     # if isinstance(src1.type,simd.types.m512) and isinstance(src2.type,simd.types.m512)
     #   return C.FunctionCall("_mm512_cmp_ps_mask", [src1, src2, C.SymbolRef("_MM_CMPINT_GT", None)])
     src1_type = src1.type
   elif isinstance(src1, C.SymbolRef) and src1.name in type_map:   
       src1_type = type_map[src1.name]
   elif isinstance(src1, C.SymbolRef) and src1.name in symbol_map:
       src1_type = symbol_map[src1.name]

   elif isinstance(src1, C.FunctionCall):
      src1_name  = src1.func.name
      if "_store" in src1_name  or "_load" in src1_name  or "_broadcast" in src1_name or "_set1" in src1_name: 
         if "ps" in src1_name and "256" in src1_name:
            src1_type = simd.types.m256() 
         elif "epi32" in src1_name and "256" in src1_name:
            src1_type = simd.types.m256i()
         elif "ps" in src1_name and "512" in src1_name:
            src1_type = simd.types.m512()
         elif "epi32" in src1_name and "512" in src1_name:
            src1_type = simd.types.m512i()
         else:
            assert(False)
             
   return src1_type 

def gen_vector_cmp_instruction(dest, src1,src2, type_map, symbol_map):
  
   src1_type = get_type(src1,type_map, symbol_map)
   src2_type = get_type(src2,type_map, symbol_map)
   
   assert(src1_type is not None)
   assert(src2_type is not None)
   if isinstance(src1_type, simd.types.m256) and isinstance(src2_type, simd.types.m256):
      return C.Assign(dest, C.FunctionCall(C.SymbolRef("_mm256_cmp_ps_mask"), [src1, src2, C.SymbolRef("_MM_CMPINT_GT", None)]))
   elif isinstance(src1_type, simd.types.m512) and isinstance(src2_type, simd.types.m512):
      return C.Assign(dest, C.FunctionCall(C.SymbolRef("_mm512_cmp_ps_mask"), [src1, src2, C.SymbolRef("_MM_CMPINT_GT", None)]))
   elif isinstance(src1_type, simd.types.m256i) and isinstance(src2_type, simd.types.m256i):
      return C.Assign(dest, C.FunctionCall(C.SymbolRef("_mm256_cmp_epi32_mask"), [src1, src2, C.SymbolRef("_MM_CMPINT_GT", None)]))
   elif isinstance(src1_type, simd.types.m512i) and isinstance(src2_type, simd.types.m512i):
      return C.Assign(dest, C.FunctionCall(C.SymbolRef("_mm512_cmp_epi32_mask"), [src1, src2, C.SymbolRef("_MM_CMPINT_GT", None)]))
   else:
      assert(False)
 

def gen_mask_move_instruction(dest, src1, selector, src2, type_map, symbol_map):


   src1_type = get_type(src1, type_map, symbol_map)
   src2_type = get_type(src2, type_map, symbol_map)
 
   #assert(src1_type == src2_type)
   
   assert(src1_type is not None)
   assert(src2_type is not None)
   if isinstance(src1_type, simd.types.m256) and isinstance(src2_type, simd.types.m256):
      return C.Assign(dest, C.FunctionCall(C.SymbolRef("_mm256_mask_mov_ps"), [src1, selector,src2]))
   elif isinstance(src1_type, simd.types.m512) and isinstance(src2_type, simd.types.m512):
      return C.Assign(dest, C.FunctionCall(C.SymbolRef("_mm512_mask_mov_ps"), [src1, selector,src2]))
   elif isinstance(src1_type, simd.types.m256i) and isinstance(src2_type, simd.types.m256i):
      return C.Assign(dest, C.FunctionCall(C.SymbolRef("_mm256_mask_mov_epi32"), [src1, selector,src2]))
   elif isinstance(src1_type, simd.types.m512i) and isinstance(src2_type, simd.types.m512i):
      return C.Assign(dest, C.FunctionCall(C.SymbolRef("_mm512_mask_mov_epi32"), [src1, selector,src2]))
   else:
      assert(False)





class ReplaceSymbol(ast.NodeTransformer):
    def __init__(self, old, new):
      self.old = old
      self.new = new

    def visit(self, node):
        return super().visit(node)

    def visit_FunctionCall(self, node):
        if "_store" in node.func.name:
            node.args = [util.replace_symbol(self.old, self.new, arg) for arg in node.args]
        return node
 
    def visit_BinaryOp(self, node):
       
        if isinstance(node.op ,C.Op.Assign):        
            
            if not (isinstance(node.op ,C.Op.Assign) and isinstance(node.left, C.SymbolRef) and (node.left.type is not None) and node.left.name == self.old)\
              and not(isinstance(node.op ,C.Op.Assign) and isinstance(node.left, C.SymbolRef) and (node.left.type is not None) and node.left.name == self.new.name): 
              #if not isinstance(node.left, C.SymbolRef):
              #  util.print_ast(node.left)  
              
              node.left = util.replace_symbol(self.old, self.new, node.left)
              node.right = util.replace_symbol(self.old, self.new, node.right)
          
        else:
            node = util.replace_symbol(self.old, self.new, node)
        return node
 
def replace_symbol(old, new, ast):
    return ReplaceSymbol(old, new).visit(ast)

'''
class ScalarReplacer(ast.NodeTransformer):

   def __init__(self, defs_):
       self.defs = deepcopy(defs_)



    def visit(self, node):
        node = super().visit(node)
        if hasattr(node, 'body'):
            # [collector.visit(s) for s in node.body]
            newbody = []
            for s in node.body:
              if not(isinstance(s, C.BinaryOp) and isinstance(s.op, C.Op.Assign)\
                # Anand - needs more work 27th June 2017
                and isinstance(s.left, C.SymbolRef) and s.left.type is not None and s.left.name in self.variables \
                     and s.left.name in self.defs):
                     for i in se  


                newbody.append(s)
 
              else:
 
                  newbody.append(s)
            node.body = util.flatten(newbody)
        return node





   def visit_SymbolRef(self, node):

        if node.name in self.defs:
           return self.defs[node.name]
           
        return node 

'''


class IfConvert(ast.NodeTransformer):


      def __init__(self, type_map, symbol_map):
        self.var_types = deepcopy(type_map)
        self.symbol_table = deepcopy(symbol_map)

      _tmp = -1


      def _gen_register(self):
        IfConvert._tmp += 1
        return "___selector" + str(self._tmp)

      def visit_If(self, node):
       if node.then is not None and node.elze is None:
         if isinstance(node.cond, C.BinaryOp) and isinstance(node.cond.op, C.Op.Gt) and is_vector_type(node.cond.right, self.var_types,self.symbol_table) and is_vector_type(node.cond.left, self.var_types, self.symbol_table): 
            new_then = []
            selector = self._gen_register()  
            mask_assign =  gen_vector_cmp_instruction(C.SymbolRef(selector, gen_selector_type()()),node.cond.left, node.cond.right, self.var_types,self.symbol_table)
            new_then.append(mask_assign)
            for s in node.then:
               if isinstance(s, C.BinaryOp) and isinstance(s.op, C.Op.Assign): 
                  if isinstance(s.left, C.SymbolRef) and (isinstance(s.left.type, get_simd_type(ctypes.c_int())) or isinstance(s.left.type, get_simd_type(ctypes.c_float()))) and isinstance(s.right, C.SymbolRef):
                      if isinstance(s.left.type, get_simd_type(ctypes.c_int())):
                          s.right = broadcast_ss(C.SymbolRef(s.right.name, None), ctypes.c_int())
                          self.var_types[s.left.name] = get_simd_type(ctypes.c_int())()     
                      else:
                          s.right = broadcast_ss(C.SymbolRef(s.right.name, None), ctypes.c_float())
                          self.var_types[s.left.name] = get_simd_type(ctypes.c_float())()
                      
                  elif isinstance(s.left, C.SymbolRef) and s.left.name in self.symbol_table\
                         and (isinstance(self.symbol_table[s.left.name], get_simd_type(ctypes.c_int())) or isinstance(self.symbol_table[s.left.name], get_simd_type(ctypes.c_float()))) and isinstance(s.right, C.SymbolRef):
                      if isinstance(self.symbol_table[s.left.name], get_simd_type(ctypes.c_int())):
                          s.right = broadcast_ss(C.SymbolRef(s.right.name, None), ctypes.c_int())
                          self.var_types[s.left.name] = get_simd_type(ctypes.c_int())()
                      else:
                          s.right = broadcast_ss(C.SymbolRef(s.right.name, None), ctypes.c_float())
                          self.var_types[s.left.name] = get_simd_type(ctypes.c_float())()
                      

                  elif isinstance(s.left, C.SymbolRef) and s.left.name in self.var_types \
                         and (isinstance(self.var_types[s.left.name], get_simd_type(ctypes.c_int())) or isinstance(self.var_types[s.left.name], get_simd_type(ctypes.c_float()))) and isinstance(s.right, C.SymbolRef):
                      if isinstance(self.var_types[s.left.name], get_simd_type(ctypes.c_int())):
                          s.right = broadcast_ss(C.SymbolRef(s.right.name, None), ctypes.c_int())
                          self.var_types[s.left.name] = get_simd_type(ctypes.c_int())()
                      else:
                          s.right = broadcast_ss(C.SymbolRef(s.right.name, None), ctypes.c_float())
                          self.var_types[s.left.name] = get_simd_type(ctypes.c_float())()

               if is_vector_type(s.left, self.var_types,self.symbol_table) and is_vector_type(s.right, self.var_types, self.symbol_table):
                    s = gen_mask_move_instruction(s.left, s.left, C.SymbolRef(selector, None), s.right, self.var_types,self.symbol_table)
               else:
                   return node  
               new_then.append(s)
            return new_then
         else:
            return node  
       else:
           return node    
       
       '''  
       if node.cond is not None:
          node.cond = self.visit(node.cond)
       if node.elze is not None:
          node.elze =  [self.visit(s) for s in node.elze]
       return node
       ''' 




class ScalarExpand(ast.NodeTransformer):
    
    _tmp = -1
    def _gen_register(self):
        ScalarExpand._tmp += 1
        return "___y" + str(self._tmp)




    def __init__(self, vars_, type_table, symbol_table):
        self.variables = set()
        self.type_table = deepcopy(type_table)
        for i in vars_:
            self.variables.add(i)
        self.transposed_buffers = {}
        self.symbol_table = deepcopy(symbol_table)
        self.defs = {}


    def visit_If(self, node):
        if node.then is not None:
          node.then =  [self.visit(s) for s in node.then]
        if node.cond is not None:
          node.cond = self.visit(node.cond)
        if node.elze is not None:
          node.elze =  [self.visit(s) for s in node.elze]
        return node 
 
    def visit(self, node):
        node = super().visit(node)
        if hasattr(node, 'body'):
            # [collector.visit(s) for s in node.body]
            newbody = []
            for s in node.body:
              if isinstance(s, C.BinaryOp) and isinstance(s.op, C.Op.Assign) :
                # Anand - needs more work 27th June 2017
                if isinstance(s.left, C.SymbolRef) and (s.left.type is not None) and s.left.name in self.variables \
                     and s.left.name not in self.defs:
                      y = self._gen_register()
                        


                      new_stmt = C.Assign (C.SymbolRef(y, get_simd_type(s.left.type)()), broadcast_ss(C.SymbolRef(s.left.name, None), s.left.type)) 
                      newbody.append(s)
                      newbody.append(new_stmt)
                      self.defs[s.left.name] = C.SymbolRef(y, None)
                      self.symbol_table[y] = get_simd_type(s.left.type)() 
                else:
                    for i in self.defs:    
                      s =   replace_symbol(i, self.defs[i],s)
                    
                    if (isinstance(s.left.type, get_simd_type(ctypes.c_int())) or isinstance(s.left.type, get_simd_type(ctypes.c_float()))) and isinstance(s.right, C.SymbolRef):
                        s.right = broadcast_ss(C.SymbolRef(s.right.name, None), s.left.type)                             

                    elif isinstance(s.left, C.SymbolRef) and s.left.name in self.symbol_table and\
                         (isinstance(self.symbol_table[s.left.name], get_simd_type(ctypes.c_int())) or isinstance(self.symbol_table[s.left.name], get_simd_type(ctypes.c_float()))) and isinstance(s.right, C.SymbolRef):
                        s.right = broadcast_ss(C.SymbolRef(s.right.name, None), self.symbol_table[s.left.name])          
                    
                    newbody.append(s)
                     
              else:
                    
                   for i in self.defs:
                      s = replace_symbol(i, self.defs[i],s)

                   newbody.append(s)
            node.body = util.flatten(newbody)
        return node

def scalar_expand_vars(ast, variables, type_map, symbol_map):
    transformer = ScalarExpand(variables, type_map, symbol_map)
    try:
        ast = transformer.visit(ast)
        #ast = ScalarReplacer(transformer.defs).visit(ast)

    except Exception as e:
        print("ERROR: Failed to scalar expand\n")
        print("---------- BEGIN AST ----------")
        print(ast)
        print("---------- END AST   ----------")
        raise e
    return (ast, transformer.type_table, transformer.symbol_table)

def if_convert(ast, type_map, symbol_map): 
    transformer = IfConvert(type_map, symbol_map) 
    try:
        ast = transformer.visit(ast)
        #ast = ScalarReplacer(transformer.defs).visit(ast)
 
    except Exception as e:
        print("ERROR: Failed to scalar expand\n")
        print("---------- BEGIN AST ----------")
        print(ast)
        print("---------- END AST   ----------")
        raise e
    return ast

