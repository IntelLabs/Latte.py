import ast
import ctree.c.nodes as C
import astor
#from copy import deepcopy
import ctypes

class Timer(ast.NodeTransformer):
    """
    Performs simple fusion of loops when to_source(node.iter) and to_source(node.target) are identical

    Does not perform dependence analysis
    """
   
    
    def visit_FunctionDecl(self, node):
        new_body = []
        count = 0
        #Declare array of times here


        #
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
        
        #arraydef = C.ArrayDef(C.SymbolRef('times'), ctypes.c_int(count))
        #arraydef = C.Array(C.SymbolRef('times'), ctypes.c_int(count))
        #new_body.insert(0, arraydef)       
        memset = C.Assign(C.SymbolRef('times'), C.FunctionCall(C.SymbolRef('doublecalloccast'),[C.Constant(count)]))
        new_body.insert(0,  memset)
        for i in range(0,count):
          print_stmt = C.FunctionCall(C.SymbolRef('printf'),[C.String("\ttimes[%d] = %g\\n"), C.Constant(i), C.ArrayRef(C.SymbolRef('times'), C.Constant(i))])
          new_body.append(print_stmt)
        node.defn = new_body     
        return node

def timer(ast):
    return Timer().visit(ast)
