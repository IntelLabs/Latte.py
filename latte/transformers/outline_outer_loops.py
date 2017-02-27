import ast
import ctree.c.nodes as C
import astor
from copy import deepcopy
import numpy as np
from ctree.templates.nodes import StringTemplate
import latte.util as util
import ctree
 
class Outliner(ast.NodeTransformer):
    """
    Performs simple fusion of loops when to_source(node.iter) and to_source(node.target) are identical

    Does not perform dependence analysis
    """
    def __init__(self, buffers, name):
        self.seen = set()
        self.new_funcs =[]   
        self.func_headers =[]
        self.name = name
        self.buffers = buffers
        
    def visit_BinaryOp(self, node):
        
        #node_orig = super().visit(node)
        temp = node 
        if isinstance(temp.op, C.Op.ArrayRef):
            while not isinstance(temp, C.SymbolRef):
                temp = temp.left
            self.seen.add(temp.name)
        else:
            self.visit(temp.left)
            self.visit(temp.right)

        return node

 
    def visit_FunctionCall(self, node):
        node.args = [self.visit(a) for a in node.args]
        return node


    def visit_FunctionDecl(self, node):
        new_body = []
        pre=[]
        count = 0
        _id = 2
        for statement in node.defn:
            self.seen = set()
            #self.visit(statement)   
            

            if isinstance(statement, ast.For) or isinstance(statement, C.For):
              
                
               temp =[] 

               if hasattr(statement, 'pre_trans') and statement.pre_trans is not None:
                    #new_body.extend(stmt.pre_trans)
                    pre.extend(statement.pre_trans)   
                   # self.visit(temp)

               #self.seen = set();
               self.visit(statement)

               args =[]
               args2=[] 
               args3 =[]
               for var in self.seen: 
                    args.append(C.SymbolRef(var))
                    args2.append(var)
                    args3.append(C.SymbolRef("_"+var))
               # create function call
               
               func_name = self.name + str(_id)   
               _id = _id + 1 
                
 
               arg_bufs = [self.buffers[var] for var in args2]
               #arg_bufs.sort()
                 
               type_sig = [np.ctypeslib.ndpointer(buf.dtype, buf.ndim, buf.shape) for buf in arg_bufs]
               params   = [C.SymbolRef("_" + arg, typ()) for arg, typ in zip(args2, type_sig)]
               
               
                    
               outlined_func_call = C.FunctionCall(C.SymbolRef(func_name),args3)
               

               for arg in args2:
                     name = arg
                     buf = self.buffers[name]
 
               new_body2=[]
               for arg in args2:
                    name = arg
                    buf = self.buffers[name]
                    new_body2.insert(0, StringTemplate("__assume_aligned({}, 64);\n".format(name)))
                    util.insert_cast(new_body2, buf.shape[1:], name, buf.dtype)


               new_body2.append(statement) 
               func_decl =  C.FunctionDecl(None, C.SymbolRef(func_name), params, new_body2)
               

               if len(args2) > 0:
                    #shape_str = "{}* ".format(self.buffers[args2[0]].dtype) + args2[0].join(", {}* ".format(self.buffers[d].dtype) + "{}".format(d) for d in args2[1:])
                    shape_str = "{}* ".format(   ctree.types.codegen_type(ctree.types.get_c_type_from_numpy_dtype(self.buffers[args2[0]].dtype)())) + \
                            args2[0].join(", {}* ".format( ctree.types.codegen_type(ctree.types.get_c_type_from_numpy_dtype(self.buffers[d].dtype)())) + "{}".format(d) for d in args2[1:])

               else:
                    shape_str =""

               self.func_headers.append(StringTemplate("void  $func ($args);",
                    {"func":C.SymbolRef(func_name),
                     "args":C.SymbolRef(shape_str)  

                    }))    
               self.new_funcs.append(func_decl)


               new_body.append(outlined_func_call)



            else:
                new_body.append(statement)
        new_body = pre + new_body
        node.defn = new_body
        return node


  
