
import latte.util as util
import ctypes
import ast
import ctree.c.nodes as C
from copy import deepcopy

prefetch_symbol_table={0:'__mm_prefetch_t0', 1:'__mm_prefetch_t1'}

def check_name(arg, prefetch_field):
  if isinstance(arg, C.UnaryOp) and isinstance(arg.op, C.Op.Ref) and isinstance(arg.arg, C.BinaryOp) and isinstance(arg.arg.op, C.Op.ArrayRef):
    curr_node = arg.arg
  elif isinstance(arg, C.BinaryOp) and isinstance(arg.op, C.Op.ArrayRef):
    curr_node = arg
  else:
    curr_node = None
  if curr_node:
    array_name=""
    while curr_node:
      if(isinstance(curr_node, C.SymbolRef)):
        array_name=curr_node.name
        break
      else :
        curr_node = curr_node.left
    if prefetch_field in array_name:
      return True
  return False

class SimplePrefetcher(ast.NodeTransformer):
    def __init__(self, prefetch_field, prefetch_type, enclosing_loop_var, dim, prefetch_count, prefetch_loop_var, prefetch_multiplier, prefetch_constant, cacheline_hint):
        super().__init__()
        self.prefetch_field = prefetch_field
        self.prefetch_type = prefetch_type
        self.enclosing_loop_var = enclosing_loop_var
        self.dim = dim
        self.prefetch_count = prefetch_count
        self.prefetch_loop_var = prefetch_loop_var
        self.prefetch_multiplier = prefetch_multiplier
        self.prefetch_constant = prefetch_constant
        self.cacheline_hint= cacheline_hint

    def rewrite_arg(self, arg):
        if isinstance(arg, C.UnaryOp) and isinstance(arg.op, C.Op.Ref) and isinstance(arg.arg, C.BinaryOp) and isinstance(arg.arg.op, C.Op.ArrayRef):
          curr_node = arg.arg
        elif isinstance(arg, C.BinaryOp) and isinstance(arg.op, C.Op.ArrayRef):
          curr_node = arg
        else:
          curr_node = None
        idx = self.dim
        while (idx+1 != 0):
          curr_node = curr_node.left
          idx+=1
        old_expr = curr_node.right
        new_expr = C.Add(old_expr, C.Constant(self.prefetch_constant))
        curr_node.right = new_expr
        if isinstance(arg, C.BinaryOp) and isinstance(arg.op, C.Op.ArrayRef):
          return C.Ref(arg)
        return arg

    def visit_For(self, node):
        node.body = util.flatten([self.visit(s) for s in node.body])
        if node.init.left.name == self.enclosing_loop_var:
            new_body = []
            prefetch_count = self.prefetch_count
            for stmt in node.body:
                new_body.append(stmt)
                if prefetch_count > 0 and isinstance(stmt, C.BinaryOp) and isinstance(stmt.op, C.Op.Assign) and \
                   isinstance(stmt.right, C.FunctionCall) and "_mm" in stmt.right.func.name \
                   and ("_load_" in stmt.right.func.name or "_set1" in stmt.right.func.name or "_broadcast" in stmt.right.func.name):
                  ast.dump(stmt.right.args[0])
                  if check_name(stmt.right.args[0], self.prefetch_field):
                    array_ref = deepcopy(stmt.right.args[0])
                    new_array_ref= self.rewrite_arg(array_ref)
                    prefetch_count -= 1
                    new_body.append(C.FunctionCall(C.SymbolRef(prefetch_symbol_table[self.cacheline_hint]), [new_array_ref]))
            node.body = new_body
        return node

def insert_simple_prefetches(ast, prefetch_field, prefetch_type, enclosing_loop_var, dim, prefetch_count, prefetch_loop_var, prefetch_multiplier,  prefetch_constant, cacheline_hint):
     return SimplePrefetcher(prefetch_field, prefetch_type, enclosing_loop_var, dim, prefetch_count, prefetch_loop_var, prefetch_multiplier, prefetch_constant,cacheline_hint).visit(ast)


class HoistPrefetch(ast.NodeTransformer):
    escape_body=[]
    def __init__(self, prefetch_dest_loop):
        super().__init__()
        self.prefetch_dest_loop = prefetch_dest_loop

    def visit_For(self, node):
        node.body = util.flatten([self.visit(s) for s in node.body])
        if node.init.left.name == self.prefetch_dest_loop:
          node.body = HoistPrefetch.escape_body + node.body
        return node

class InitPrefetcher(ast.NodeTransformer):
    init_body=[]
    def __init__(self, prefetch_init_loop):
        super().__init__()
        self.prefetch_init_loop = prefetch_init_loop

    def visit_For(self, node):
        node.body = util.flatten([self.visit(s) for s in node.body])
        if node.init.left.name == self.prefetch_init_loop:
          node.body = InitPrefetcher.init_body + node.body
        return node


class StridedPrefetcher(ast.NodeTransformer):
    def __init__(self, prefetch_field, prefetch_type, enclosing_loop_var, dim, prefetch_count, prefetch_offset, prefetch_dest_loop,  prefetch_init_loop, prefetch_loop_var, prefetch_multiplier, prefetch_constant, prefetch_num_zeroes, cacheline_hint):
        super().__init__()
        self.prefetch_field = prefetch_field
        self.prefetch_type = prefetch_type
        self.enclosing_loop_var = enclosing_loop_var
        self.dim = dim
        self.prefetch_count = prefetch_count
        self.prefetch_offset = prefetch_offset
        self.prefetch_dest_loop = prefetch_dest_loop
        self.prefetch_init_loop = prefetch_init_loop
        self.prefetch_loop_var = prefetch_loop_var
        self.prefetch_multiplier = prefetch_multiplier
        self.prefetch_constant = prefetch_constant
        self.cacheline_hint= cacheline_hint
        self.prefetch_num_zeroes = prefetch_num_zeroes

    def rewrite_arg(self, arg):
        if isinstance(arg, C.UnaryOp) and isinstance(arg.op, C.Op.Ref) and isinstance(arg.arg, C.BinaryOp) and isinstance(arg.arg.op, C.Op.ArrayRef):
          curr_node = arg.arg
        elif isinstance(arg, C.BinaryOp) and isinstance(arg.op, C.Op.ArrayRef):
          curr_node = arg
        else:
          curr_node = None
        idx = self.dim
        num_zeroes = self.prefetch_num_zeroes
        while (idx+1 != 0):
          if num_zeroes > 0:
            curr_node.right=C.Constant(0)
            num_zeroes--
          curr_node = curr_node.left
          idx+=1
        old_expr = curr_node.right
        #if isinstance(old_expr, C.BinaryOp) and isinstance(old_expr.op, C.Op.Add):
        #  old_expr = old_expr.left
        #new_expr = C.Add(old_expr, C.Mul(C.Add(C.SymbolRef(self.prefetch_loop_var), C.SymbolRef(self.prefetch_constant)), C.SymbolRef(self.prefetch_multiplier)))
        new_expr = C.Mul(C.Add(C.SymbolRef(self.prefetch_loop_var), C.SymbolRef(self.prefetch_constant)), C.SymbolRef(self.prefetch_multiplier))
        curr_node.right = new_expr
        if isinstance(arg, C.BinaryOp) and isinstance(arg.op, C.Op.ArrayRef):
            return C.Ref(arg)
        return arg

    def visit_For(self, node):
        node.body = util.flatten([self.visit(s) for s in node.body])
        if node.init.left.name == self.enclosing_loop_var:
            new_body = []
            added_code = False
            prefetch_count = self.prefetch_count
            for stmt in node.body:
                new_body.append(stmt)
                if prefetch_count > 0 and isinstance(stmt, C.BinaryOp) and isinstance(stmt.op, C.Op.Assign) and \
                   isinstance(stmt.right, C.FunctionCall) and "_mm" in stmt.right.func.name \
                   and ("_load_" in stmt.right.func.name or "_set1" in stmt.right.func.name or "_broadcast" in stmt.right.func.name):
                  ast.dump(stmt.right.args[0])
                  if check_name(stmt.right.args[0], self.prefetch_field):
                    array_ref = deepcopy(stmt.right.args[0])
                    new_array_ref= self.rewrite_arg(array_ref)
                    where_to_add = new_body
                    prefetch_count -= 1
                    if node.init.left.name != self.prefetch_dest_loop:
                      where_to_add = HoistPrefetch.escape_body
                    added_code = True
                    where_to_add.append(C.FunctionCall(C.SymbolRef(prefetch_symbol_table[self.cacheline_hint]), [C.Add(new_array_ref, C.SymbolRef("prefetch_offset_var"))]))
                    where_to_add.append(C.Assign(C.SymbolRef("prefetch_offset_var"), C.Add(C.SymbolRef("prefetch_offset_var"), C.Constant(self.prefetch_offset))))

            if added_code: 
              InitPrefetcher.init_body.append(C.Assign(C.SymbolRef("prefetch_offset_var", ctypes.c_int()), C.Constant(0)))
            node.body = new_body
        return node

def insert_strided_prefetches(ast,  prefetch_field, prefetch_type, enclosing_loop_var, dim, prefetch_count, prefetch_offset, prefetch_dest_loop,  prefetch_init_loop, prefetch_loop_var, prefetch_multiplier, prefetch_constant, prefetch_num_zeroes, cacheline_hint):
     HoistPrefetch.escape_body=[]
     InitPrefetcher.init_body=[]
     return InitPrefetcher(prefetch_init_loop).visit(HoistPrefetch(prefetch_dest_loop).visit(StridedPrefetcher(prefetch_field, prefetch_type, enclosing_loop_var, dim, prefetch_count, prefetch_offset, prefetch_dest_loop,  prefetch_init_loop, prefetch_loop_var, prefetch_multiplier, prefetch_constant, prefetch_num_zeroes, cacheline_hint).visit(ast)))

class SimpleHoistPrefetcher(ast.NodeTransformer):
    def __init__(self, prefetch_field, prefetch_type, enclosing_loop_var, dim, prefetch_count, prefetch_loop_var, prefetch_multiplier, prefetch_constant, prefetch_num_zeroes, cacheline_hint):
        super().__init__()
        self.prefetch_field = prefetch_field
        self.prefetch_type = prefetch_type
        self.enclosing_loop_var = enclosing_loop_var
        self.dim = dim
        self.prefetch_count = prefetch_count
        self.prefetch_dest_loop = prefetch_loop_var
        self.prefetch_multiplier = prefetch_multiplier
        self.prefetch_constant = prefetch_constant
        self.prefetch_num_zeroes = prefetch_num_zeroes
        self.cacheline_hint= cacheline_hint

    def rewrite_arg(self, arg):
        if isinstance(arg, C.UnaryOp) and isinstance(arg.op, C.Op.Ref) and isinstance(arg.arg, C.BinaryOp) and isinstance(arg.arg.op, C.Op.ArrayRef):
          curr_node = arg.arg
        elif isinstance(arg, C.BinaryOp) and isinstance(arg.op, C.Op.ArrayRef):
          curr_node = arg
        else:
          curr_node = None
        idx = self.dim
        num_zeroes = self.prefetch_num_zeroes
        while (idx+1 != 0):
          if num_zeroes > 0:
            curr_node.right=C.Constant(0)
            num_zeroes--
          curr_node = curr_node.left
          idx+=1
        old_expr = curr_node.right
        new_expr = C.Add(old_expr, C.Constant(self.prefetch_constant))
        curr_node.right = new_expr
        if isinstance(arg, C.BinaryOp) and isinstance(arg.op, C.Op.ArrayRef):
          return C.Ref(arg)
        return arg

    def visit_For(self, node):
        node.body = util.flatten([self.visit(s) for s in node.body])
        if node.init.left.name == self.enclosing_loop_var:
            new_body = []
            prefetch_count = self.prefetch_count
            for stmt in node.body:
                new_body.append(stmt)
                if prefetch_count > 0 and isinstance(stmt, C.BinaryOp) and isinstance(stmt.op, C.Op.Assign) and \
                   isinstance(stmt.right, C.FunctionCall) and "_mm" in stmt.right.func.name \
                   and ("_load_" in stmt.right.func.name or "_set1" in stmt.right.func.name or "_broadcast" in stmt.right.func.name):
                  ast.dump(stmt.right.args[0])
                  if check_name(stmt.right.args[0], self.prefetch_field):
                    array_ref = deepcopy(stmt.right.args[0])
                    new_array_ref= self.rewrite_arg(array_ref)
                    prefetch_count -= 1
                    HoistPrefetch.escape_body.append(C.FunctionCall(C.SymbolRef(prefetch_symbol_table[self.cacheline_hint]), [new_array_ref]))
            node.body = new_body
        return node

def insert_simple_hoist_prefetches(ast, prefetch_field, prefetch_type, enclosing_loop_var, dim, prefetch_count, prefetch_loop_var, prefetch_multiplier,  prefetch_constant, prefetch_num_zeroes, cacheline_hint):
     HoistPrefetch.escape_body = []
     return HoistPrefetch(prefetch_loop_var).visit(SimpleHoistPrefetcher(prefetch_field, prefetch_type, enclosing_loop_var, dim, prefetch_count, prefetch_loop_var, prefetch_multiplier, prefetch_constant, prefetch_num_zeroes,cacheline_hint).visit(ast))

