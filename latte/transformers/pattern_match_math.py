import ast
import ctypes
import ctree.c.nodes as C
import latte.config
from ctree.templates.nodes import StringTemplate
import random

class PatternMatchMath(ast.NodeTransformer):
    def visit_FunctionCall(self, node):
        if isinstance(node.func, C.SymbolRef):
            if node.func.name == "rand":
                if "OPENCL" in latte.config.parallel_strategy:
                    import struct
                    platform_c_maxint = 2 ** (struct.Struct('i').size * 8 - 1) - 1
                    return StringTemplate("((({} + get_global_id(0)) * 0x5DEECE66DL + 0xBL) & ((1L << 48) - 1)) >> 16".format(int(random.random() * platform_c_maxint)))
                return C.Div(node, C.Cast(ctypes.c_float(), C.SymbolRef("RAND_MAX")))
        #ANAND: 10/11/2016 Adding following uutility to convert python max to c max
        if isinstance(node.func, C.SymbolRef):
            if node.func.name == "max":
                return C.FunctionCall(C.SymbolRef("MAX"),
                [node.args[0],node.args[1]])
 




        return node
