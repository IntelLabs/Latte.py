import latte.util as util
import ast
import ctree.c.nodes as C
from copy import deepcopy
import ctypes

class UnrollStatements(ast.NodeTransformer):
    def __init__(self, target_var, factor, unroll_type):
        super().__init__()
        self.target_var = target_var
        self.factor = factor
        self.unrolled_vars = set()
        self.unroll_type = unroll_type

    def visit(self, node):
        """
        Support replacing nodes with a list of nodes by flattening `body`
        fields.
        """
        node = super().visit(node)
        if hasattr(node, 'body'):
            node.body = util.flatten(node.body)
        return node

    def visit_BinaryOp(self, node):
        if isinstance(node.op, C.Op.Assign):
            check = [util.contains_symbol(node.right, var) for var in list(self.unrolled_vars)+ [self.target_var]]
            if any(check):
                body = []
                if hasattr(node.left, 'type') and node.left.type is not None:
                    self.unrolled_vars.add(node.left.name)
                for i in range(self.factor):
                    stmt = deepcopy(node)
                    for var in self.unrolled_vars:
                        stmt = util.replace_symbol(var, C.SymbolRef(var + "_" + str(i)), stmt)
                    if self.unroll_type == 0:
                        body.append(util.replace_symbol(self.target_var, C.Add(C.SymbolRef(self.target_var), C.Constant(i)), stmt))
                    elif self.unroll_type == 1 :
                        body.append(util.replace_symbol(self.target_var, C.Add(C.Mul(C.Constant(self.factor),C.SymbolRef(self.target_var)), C.Constant(i)), stmt))
                    else:
                       assert(false)
                return body
        return node

    def visit_If(self, node):
         check = [util.contains_symbol(node, var) for var in list(self.unrolled_vars) + [self.target_var]]
         
         if any(check):
            body= []
            for i in range(self.factor):
                stmt = deepcopy(node)
                for var in self.unrolled_vars:
                    stmt = util.replace_symbol(var, C.SymbolRef(var + "_" + str(i)), stmt)
                if self.unroll_type == 0:
                    body.append(util.replace_symbol(self.target_var, C.Add(C.SymbolRef(self.target_var), C.Constant(i)), stmt))
                elif self.unroll_type == 1 :
                    body.append(util.replace_symbol(self.target_var, C.Add(C.Mul(C.Constant(self.factor),C.SymbolRef(self.target_var)), C.Constant(i)), stmt))
                else:
                    assert(false)

            return body
         return node  


    def visit_AugAssign(self, node):
        check = [util.contains_symbol(node.value, var) for var in list(self.unrolled_vars) + [self.target_var]]
        if any(check):
            body = []
            if isinstance(node.target, C.SymbolRef):
                self.unrolled_vars.add(self._get_name(node.target.name))
                for i in range(self.factor):
                    stmt = deepcopy(node)
                    for var in self.unrolled_vars:
                        stmt = util.replace_symbol(var, C.SymbolRef(var + "_" + str(i)), stmt)
                    #body.append(util.replace_symbol(self.target_var, C.Add(C.SymbolRef(self.target_var), C.Constant(i)), stmt))
                    if self.unroll_type == 0:
                        body.append(util.replace_symbol(self.target_var, C.Add(C.SymbolRef(self.target_var), C.Constant(i)), stmt))
                    elif self.unroll_type == 1 :
                        body.append(util.replace_symbol(self.target_var, C.Add(C.Mul(C.Constant(self.factor),C.SymbolRef(self.target_var)), C.Constant(i)), stmt))
                    else:
                       assert(false)


                return body
            elif isinstance(node.target, C.BinaryOp) and isinstance(node.target.op, C.Op.ArrayRef):
                assert False
                for i in range(self.factor):
                    stmt = deepcopy(node)
                    for var in self.unrolled_vars:
                        stmt = util.replace_symbol(var, C.SymbolRef(var + "_" + str(i)), stmt)
                    #body.append(util.replace_symbol(self.target_var, C.Add(C.SymbolRef(self.target_var), C.Constant(i)), stmt))

                    if self.unroll_type == 0:
                        body.append(util.replace_symbol(self.target_var, C.Add(C.SymbolRef(self.target_var), C.Constant(i)), stmt))
                    elif self.unroll_type == 1 :
                        body.append(util.replace_symbol(self.target_var, C.Add(C.Mul(C.Constant(self.factor),C.SymbolRef(self.target_var)), C.Constant(i)), stmt))
                    else:
                       assert(false)

                return body
            else:
                raise NotImplementedError()
        return node

    def visit_FunctionCall(self, node):
        check = [util.contains_symbol(node, var) for var in list(self.unrolled_vars) + [self.target_var]]
        if "store" in node.func.name and "_mm" in node.func.name and any(check):
            body = []
            for i in range(self.factor):
                stmt = deepcopy(node)
                for var in self.unrolled_vars:
                    stmt = util.replace_symbol(var, C.SymbolRef(var + "_" + str(i)), stmt)
                #stmt = util.replace_symbol(self.target_var, C.Add(C.SymbolRef(self.target_var), C.Constant(i)), stmt)
                #body.append(stmt)
                if self.unroll_type == 0:
                    body.append(util.replace_symbol(self.target_var, C.Add(C.SymbolRef(self.target_var), C.Constant(i)), stmt))
                elif self.unroll_type == 1 :
                    body.append(util.replace_symbol(self.target_var, C.Add(C.Mul(C.Constant(self.factor),C.SymbolRef(self.target_var)), C.Constant(i)), stmt))
                else:
                    assert(false)

            return body
        return node

class LoopUnroller(ast.NodeTransformer):
    def __init__(self, target_var, factor, unroll_type):
        super().__init__()
        self.target_var = target_var
        self.unroll_type = unroll_type
        self.factor = factor
    if False:
        def visit(self, node):
            """
            Support replacing nodes with a list of nodes by flattening `body`
            fields.
            """
            node = super().visit(node)
            if hasattr(node, 'body'):
                node.body = util.flatten(node.body)
            return node

        def visit_For(self, node):
            node.body = [self.visit(s) for s in node.body]
            if node.init.left.name == self.target_var:
                node.incr = C.AddAssign(C.SymbolRef(self.target_var), C.Constant(self.factor))
                visitor = UnrollStatements(self.target_var, self.factor)
                node.body = util.flatten([visitor.visit(s) for s in node.body])
                if node.test.right.value == self.factor:
                    return [util.replace_symbol(node.init.left.name, C.Constant(0), s) for s in node.body]
            return node
    elif False:
        def visit_For(self, node):
            node.body = [self.visit(s) for s in node.body]
            if node.init.left.name == self.target_var:
                # node.pragma = "unroll_and_jam({})".format(self.factor)
                node.pragma = "unroll"
            return node
    else:
        def visit_For(self, node):
            node.body = [self.visit(s) for s in node.body]
            if node.init.left.name == self.target_var:
                if self.unroll_type == 0:
                    node.incr = C.AddAssign(C.SymbolRef(self.target_var), C.Constant(self.factor))
                    node.incr = C.AddAssign(C.SymbolRef(self.target_var), C.Constant(self.factor))
                elif self.unroll_type == 1:
                    assert(node.test.right.value%self.factor == 0)
                    node.test.right.value = node.test.right.value//self.factor
                else:
                    assert(0)   
                visitor = UnrollStatements(self.target_var, self.factor, self.unroll_type)
                node.body = util.flatten([visitor.visit(s) for s in node.body])
            return node


def unroll_loop(ast, target_var, factor, unroll_type=0):
    return LoopUnroller(target_var, factor, unroll_type).visit(ast)

class UnrollStatementsNoJam(ast.NodeTransformer):
    new_body={}
    def __init__(self, target_var, factor, unroll_type):
        super().__init__()
        self.target_var = target_var
        self.factor = factor
        self.unrolled_vars = set()
        self.unroll_type = unroll_type
        for i in range(1,self.factor):
          UnrollStatementsNoJam.new_body[i] = []
    def visit(self, node):
        """
        Support replacing nodes with a list of nodes by flattening `body`
        fields.
        """
        node = super().visit(node)
        if hasattr(node, 'body'):
            node.body = util.flatten(node.body)
        return node
    


    def visit_For(self, node):
 

            for j in range(1, self.factor):
                    UnrollStatementsNoJam.new_body[j]=[]
 
            # UnrollStatementsNoJam.new_body={}
            #for i in node.body:
                #new_body_cpy = deepcopy(UnrollStatementsNoJam.new_body)
            #node.body = [self.visit(s) for s in node.body]
           


            newbody=[]

            for s in node.body: 
                temp =  deepcopy(UnrollStatementsNoJam.new_body)

                t = self.visit(s)                   
                stmt2 = deepcopy(t)
                stmt = deepcopy(t)
                if self.unroll_type == 0:
                        s = util.replace_symbol(self.target_var, C.Add(C.SymbolRef(self.target_var), C.Constant(0)), stmt)
                else:
                        s  =  util.replace_symbol(self.target_var, C.Add(C.Mul(C.Constant(self.factor),C.SymbolRef(self.target_var)), C.Constant(0)), stmt)
 

                newbody.append(t)

                if not isinstance(t, C.For):   
                  for i in range(1, self.factor):
                    stmt = deepcopy(stmt2)
 
 
                    if self.unroll_type == 0:
                        if i in UnrollStatementsNoJam.new_body:
                            UnrollStatementsNoJam.new_body[i].append(util.replace_symbol(self.target_var, C.Add(C.SymbolRef(self.target_var), C.Constant(i)), stmt))
                        else:
                            UnrollStatementsNoJam.new_body[i] = [util.replace_symbol(self.target_var, C.Add(C.SymbolRef(self.target_var), C.Constant(i)), stmt)]
                    elif self.unroll_type == 1 :
                        if i in UnrollStatementsNoJam.new_body:
                            UnrollStatementsNoJam.new_body[i].append(util.replace_symbol(self.target_var, C.Add(C.Mul(C.Constant(self.factor),C.SymbolRef(self.target_var)), C.Constant(i)), stmt))
                        else:
                            UnrollStatementsNoJam.new_body[i] = [util.replace_symbol(self.target_var, C.Add(C.Mul(C.Constant(self.factor),C.SymbolRef(self.target_var)), C.Constant(i)), stmt)]
                    else:
                        assert(false)
 
                else:
                  var = t.init.left.name
                  
                  #if var != self.target_var:
                  for j in range(1,self.factor):
                    temp[j].append(C.For(
                        C.Assign(C.SymbolRef(var, ctypes.c_int()), C.Constant(0)),
                        C.Lt(C.SymbolRef(var), C.Constant(t.test.right.value)),
                        C.AddAssign(C.SymbolRef(var), C.Constant(t.incr.value.value)),
                        UnrollStatementsNoJam.new_body[j]))
                  
                  UnrollStatementsNoJam.new_body = deepcopy(temp)

            node.body = newbody 
            return node



class LoopUnrollerNoJam(ast.NodeTransformer):
    def __init__(self, unroll_var, unroll_factor, unroll_type):
        super().__init__()
        self.unroll_var = unroll_var
        self.unroll_factor = unroll_factor
        self.unroll_type = unroll_type
        self.newbody = {}

    def visit_For(self, node):
        node.body = [self.visit(s) for s in node.body]
        # node.body = util.flatten(node.body)
        if node.init.left.name == self.unroll_var:
            var = node.init.left.name  
            factor, unroll_type = self.unroll_factor, self.unroll_type
            if unroll_type == 0:
                node.incr = C.AddAssign(C.SymbolRef(var), C.Constant(factor))
                node.incr = C.AddAssign(C.SymbolRef(var), C.Constant(factor))
            elif unroll_type == 1:
                assert(node.test.right.value%factor == 0)
                node.test.right.value = node.test.right.value//factor
            else:
                assert(0)   
            
            '''
            UnrollStatementsNoJam.new_body={}
            
            visitor = UnrollStatementsNoJam(self.unroll_var, self.unroll_factor, self.unroll_type)
            
            node.body = util.flatten([visitor.visit(s) for s in node.body])
 

            '''
            #new_body = [] 
            #for i in range(1,factor):
            #    self.newbody[i] = []
            #for s in node.body:
            UnrollStatementsNoJam.new_body={}
            for i in range(1,factor):
                UnrollStatementsNoJam.new_body[i] = [] 

            visitor = UnrollStatementsNoJam(self.unroll_var, self.unroll_factor, self.unroll_type)
            
            node = visitor.visit(node)
            for i in range(1, factor):
               for j in range(len(UnrollStatementsNoJam.new_body[i])):
                    node.body.append(UnrollStatementsNoJam.new_body[i][j]);

            node.body = util.flatten(node.body)
            

            '''  
            if not isinstance(s, o.For):
                      
                      #visitor = UnrollStatementsNoJam(self.unroll_var, self.unroll_factor, self.unroll_type)
                      n = visitor.visit(s)
                      new_body.append(n)    
                      for j in range(1, factor):
                          for i in range(len(UnrollStatementsNoJam.new_body[j])):
                              self.newbody[j].append(util.flatten(UnrollStatementsNoJam.new_body[j][i]))
 
                else:
                    p = visitor.visit(s)
                    UnrollStatementsNoJam.new_body={}
                    n = [visitor.visit(t) for t in s.body]
                    new_body.append(p)            
                    for j in range(1, factor):
                          for i in range(len(UnrollStatementsNoJam.new_body[j])):
                              self.newbody[j].append(C.For(
                        C.Assign(C.SymbolRef(s.init.left.name, ctypes.c_int()), C.Constant(0)),
                        C.Lt(C.SymbolRef(s.init.left.name), C.Constant(s.test.right.value)),
                        C.AddAssign(C.SymbolRef(s.init.left.name), C.Constant(s.incr.value.value)),
                        util.flatten(UnrollStatementsNoJam.new_body[j][i])))
            for j in range(1, factor): 
                for i in range(len(self.newbody[j])):
                    new_body.append(self.newbody[j][i])


            node.body = util.flatten(new_body)
            #node.body = new_body
            '''
        
        return node
    
def unroll_no_jam_loop(ast, unroll_var, unroll_factor, unroll_type):
    return LoopUnrollerNoJam(unroll_var, unroll_factor, unroll_type).visit(ast)
