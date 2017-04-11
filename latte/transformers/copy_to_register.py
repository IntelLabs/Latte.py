import latte.util as util
import ctypes
import ast
import ctree.c.nodes as C
from copy import deepcopy

du_map = []
#replace_map ={}
sym_map = {} 

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
        super().__init__()
        self.replace_map = map_
        self.seen = {}
    def visit_SymbolRef(self, node):
        

        if node.name in sym_map:
          return sym_map[node.name]
        
        
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





    def visit_For(self, node):
        node.body = util.flatten([s for s in node.body])
        new_body = []
        for stmt in node.body:
          if isinstance(stmt, C.FunctionCall) and "_mm" in stmt.func.name \
             and "_store" in stmt.func.name and inReplaceMapSource(stmt.args[0], self.replace_map):
                  sym_arr_ref = extract_reference(stmt.args)  
                  store_in_du_map(sym_arr_ref)  
                  new_body.append(stmt)      

          elif isinstance(stmt, C.BinaryOp) and \
             isinstance(stmt.op, C.Op.Assign) and \
             isinstance(stmt.left, C.SymbolRef) and \
             isinstance(stmt.right, C.FunctionCall) and "_mm" in stmt.right.func.name and "_load" in stmt.right.func.name and inReplaceMapSink(stmt.right.args[0], self.replace_map): 
                      
                  source = get_alias(stmt.right.args, self.replace_map)
                  if (source is not None):
                    sym_arr_ref = construct_arr_reference(source, deepcopy(stmt.right.args))
                    if in_du_map(sym_arr_ref):
                       reg = get_register(sym_arr_ref)
                       #print(reg)       
                       if reg.name in self.seen:
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
     return RegisterCopy(map_).visit(ast)

