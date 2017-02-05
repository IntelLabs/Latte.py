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
