import ast
import ctypes
import ctree
import copy
import ctree.np
import ctree.c.nodes as C

class TypeConverter(ast.NodeTransformer):

    def visit_For(self, node):
        node.init.left.type = ctypes.c_int()
        node.body = [self.visit(s) for s in node.body]

        return node

def loop_init_long_to_int(func_def):
    try:
        
        return TypeConverter().visit(func_def)
    except NotImplementedError as e:
        print("AST that caused exception during type inference")
        print(ast)
        raise e
