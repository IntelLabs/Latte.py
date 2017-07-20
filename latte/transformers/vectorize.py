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

def set_zero_ps():
    return C.FunctionCall(C.SymbolRef({
        "AVX": "_mm256_setzero_ps",
        "AVX-2": "_mm256_setzero_ps",
        "AVX-512": "_mm512_setzero_ps"
    }[latte.config.vec_config]), [])

def store_ps(target, value):
    return C.FunctionCall(C.SymbolRef({
        "AVX": "_mm256_store_ps",
        "AVX-2": "_mm256_store_ps",
        "AVX-512": "_mm512_store_ps",
    }[latte.config.vec_config]), [target, value])

def load_ps(arg):
    return C.FunctionCall(C.SymbolRef({
        "AVX": "_mm256_load_ps",
        "AVX-2": "_mm256_load_ps",
        "AVX-512": "_mm512_load_ps",
    }[latte.config.vec_config]), [arg])

def store_epi32(target, value):
    return C.FunctionCall(C.SymbolRef({
        "AVX": "_mm256_store_epi32",
        "AVX-2": "_mm256_store_epi32",
        "AVX-512": "_mm512_store_epi32",
    }[latte.config.vec_config]), [target, value])


def broadcast_ss(arg):
    if latte.config.vec_config == "AVX-512":
        # AVX-512 doesn't support broadcast, use set1_ps and remove Ref node
        assert isinstance(arg, C.UnaryOp) and isinstance(arg.op, C.Op.Ref)
        arg = arg.arg
    return C.FunctionCall(C.SymbolRef({
        "AVX": "_mm256_broadcast_ss",
        "AVX-2": "_mm256_broadcast_ss",
        "AVX-512": "_mm512_set1_ps",
    }[latte.config.vec_config]), [arg])

def get_simd_type():
    return {
        "AVX": simd.types.m256,
        "AVX-2": simd.types.m256,
        "AVX-512": simd.types.m512,
    }[latte.config.vec_config]

def simd_fma(*args):
    assert len(args) == 3
    fma_func = {
        "AVX": "_mm256_fmadd_ps",
        "AVX-2": "_mm256_fmadd_ps",
        "AVX-512": "_mm512_fmadd_ps",
    }[latte.config.vec_config]
    return C.FunctionCall(C.SymbolRef(fma_func), list(args))

def simd_add(left, right):
    func = {
        "AVX": "_mm256_add_ps",
        "AVX-2": "_mm256_add_ps",
        "AVX-512": "_mm512_add_ps",
    }[latte.config.vec_config]
    return C.FunctionCall(C.SymbolRef(func), [left, right])


class RemoveIndexExprs(ast.NodeTransformer):
    def __init__(self, var):
        self.var = var

    def visit_SymbolRef(self, node):
        if node.name == self.var:
            return C.Constant(0)
        return node

class Vectorizer(ast.NodeTransformer):
    def __init__(self, loop_var):
        self.loop_var = loop_var
        self.transposed_buffers = {}
        self.symbol_table = {}
        self.seen = {}
 
    def visit_SymbolRef(self, node):
        if node.type is not None:
            self.seen[node.name] = node.type
        return node
 
    def get_type(self, node):
        if isinstance(node, C.SymbolRef):
            if node.name == "INFINITY":
                return ctypes.c_float()
            elif node.name in self.seen:
                return self.seen[node.name]
        elif isinstance(node, C.UnaryOp):
            return self.get_type(node.arg)
        elif isinstance(node, C.Constant):
            if isinstance(node.value, int):
                # promote all longs to int
                return ctypes.c_int()
            return ctree.types.get_ctype(node.value)
        elif isinstance(node, C.BinaryOp):
            if isinstance(node.op, C.Op.ArrayRef):
                while not isinstance(node, C.SymbolRef):
                    node = node.left
                pointer_type = self.get_type(node)
                if(pointer_type is None):
                   return None
                return ctree.types.get_c_type_from_numpy_dtype(pointer_type._dtype_)()
            else:
                left = self.get_type(node.left)
                right = self.get_type(node.right)
                return ctree.types.get_common_ctype([left, right])
        elif isinstance(node, C.FunctionCall):
            if node.func.name in ["MAX", "MIN", "max", "min", "floor"]:
                return ctree.types.get_common_ctype([self.get_type(a) for a in node.args])
        
        return None







    def visit(self, node):
        node = super().visit(node)
        if hasattr(node, "body"):
            node.body = util.flatten(node.body)
        return node

    def visit_For(self, node):
        node.body = [self.visit(s) for s in node.body]
        if node.init.left.name == self.loop_var:
            assert node.test.right.value == latte.config.SIMDWIDTH
            # index = C.Assign(
            #         C.SymbolRef(node.init.left.name, ctypes.c_int()),
            #         C.Constant(0)
            #     )
            return [RemoveIndexExprs(self.loop_var).visit(s) for s in node.body]
        return node

    def visit_AugAssign(self, node):
        node.value = self.visit(node.value)
        if util.contains_symbol(node.target, self.loop_var):
            if not util.contains_symbol(node.target.right, self.loop_var):
                target = self.visit(deepcopy(node.target))
                curr_node = node.target
                idx = 1
                while curr_node.left.right.name != self.loop_var:
                    curr_node = curr_node.left
                    idx += 1
                curr_node.left = curr_node.left.left
                node.target = C.ArrayRef(node.target, C.SymbolRef(self.loop_var))
                while not isinstance(curr_node, C.SymbolRef):
                    curr_node = curr_node.left
                if curr_node.name in self.transposed_buffers and self.transposed_buffers[curr_node.name] != idx:
                    raise NotImplementedError()
                self.transposed_buffers[curr_node.name] = idx
                curr_node.name += "_transposed"
                if isinstance(node.target.right, C.Constant) and node.target.value == 0.0:
                    return store_ps(
                        node.target.left,
                        C.BinaryOp(target, node.op, node.value)
                    )
                else:
                    return store_ps(
                        C.Ref(node.target),
                        C.BinaryOp(target, node.op, node.value)
                    )
            else:
                if isinstance(node.target.right, C.Constant) and node.target.value == 0.0:
                    return store_ps(
                        node.target.left,
                        C.BinaryOp(self.visit(node.target), node.op, node.value)
                    )
                else:
                    return store_ps(
                        C.Ref(node.target),
                        C.BinaryOp(self.visit(node.target), node.op, node.value)
                    )
        elif isinstance(node.op, C.Op.Add) and isinstance(node.value, C.FunctionCall):
            # TODO: Verfiy it's a vector intrinsic
            return C.Assign(node.target, C.FunctionCall(C.SymbolRef("_mm256_add_ps"), [node.value, node.target]))
        elif isinstance(node.target, C.BinaryOp) and isinstance(node.target.op, C.Op.ArrayRef):
            raise NotImplementedError(node)
        node.target = self.visit(node.target)
        return node

    def visit_BinaryOp(self, node):
        if isinstance(node.op, C.Op.ArrayRef):
            if util.contains_symbol(node, self.loop_var):
                if not util.contains_symbol(node.right, self.loop_var):
                    curr_node = node
                    idx = 1
                    while curr_node.left.right.name != self.loop_var:
                        curr_node = curr_node.left
                        idx += 1
                    curr_node.left = curr_node.left.left
                    node = C.ArrayRef(node, C.SymbolRef(self.loop_var))
                    while not isinstance(curr_node, C.SymbolRef):
                        curr_node = curr_node.left
                    if curr_node.name in self.transposed_buffers and self.transposed_buffers[curr_node.name] != idx:
                        raise NotImplementedError()
                    self.transposed_buffers[curr_node.name] = idx
                    curr_node.name += "_transposed"
                if isinstance(node.right, C.Constant) and node.target.value == 0.0:
                    return load_ps(node.left)
                else:
                    return load_ps(C.Ref(node))
            else:
                return broadcast_ss(C.Ref(node))
        elif isinstance(node.op, C.Op.Assign):
            node.right = self.visit(node.right)
            if isinstance(node.right, C.FunctionCall) and \
                    ("load_ps" in node.right.func.name or
                     "broadcast_ss" in node.right.func.name) and \
                    isinstance(node.left, C.SymbolRef) and node.left.type is not None:
                node.left.type = get_simd_type()()
                self.symbol_table[node.left.name] = node.left.type
                return node
            elif isinstance(node.left, C.BinaryOp) and util.contains_symbol(node.left, self.loop_var):
                if node.left.right.name != self.loop_var:
                    curr_node = node
                    idx = 1
                    while curr_node.left.right.name != self.loop_var:
                        curr_node = curr_node.left
                        idx += 1
                    curr_node.left = curr_node.left.left
                    node = C.ArrayRef(node, C.SymbolRef(self.loop_var))
                    while not isinstance(curr_node, C.SymbolRef):
                        curr_node = curr_node.left
                    if curr_node.name in self.transposed_buffers and self.transposed_buffers[curr_node.name] != idx:
                        raise NotImplementedError()
                    self.transposed_buffers[curr_node.name] = idx
                    curr_node.name += "_transposed"
                
                is_float = self.get_type(node.left)
        


                if isinstance(is_float, ctypes.c_float):  
                    if isinstance(node.left.right, C.Constant) and node.target.value == 0.0:
                        return store_ps(node.left.left, node.right)
                    else:
                        return store_ps(C.Ref(node.left), node.right)
                elif isinstance(is_float, ctypes.c_int):
                   if isinstance(node.left.right, C.Constant) and node.target.value == 0.0:
                        return store_epi32(node.left.left, node.right)
                   else:
                        return store_epi32(C.Ref(node.left), node.right)
                else:
                    if isinstance(node.left.right, C.Constant) and node.target.value == 0.0:
                        return store_ps(node.left.left, node.right)
                    else:
                        return store_ps(C.Ref(node.left), node.right)
 
    
            node.left = self.visit(node.left)
            return node
        node.left = self.visit(node.left)
        node.right = self.visit(node.right)
        return node

    def _is_vector_type(self, node):
        return node.name in self.symbol_table and isinstance(self.symbol_table[node.name], get_simd_type())

    def visit_FunctionCall(self, node):
        node.args = [self.visit(a) for a in node.args]
        if node.func.name in ["fmax", "fmin"]:
            node.func.name = {
                "AVX": "_mm256_{}_ps".format(node.func.name[1:]),
                "AVX-2": "_mm256_{}_ps".format(node.func.name[1:]),
                "AVX-512": "_mm512_{}_ps".format(node.func.name[1:]),
            }[latte.config.vec_config]
            args = []
            for arg in node.args:
                if isinstance(arg, C.Constant) and arg.value == 0:
                    args.append(set_zero_ps())
                else:
                    args.append(arg)
            node.args = args
        return node

def vectorize_loop(ast, loopvar):
    transformer = Vectorizer(loopvar)
    try:
        ast = transformer.visit(ast)
    except Exception as e:
        print("ERROR: Failed to vectorize loop with variable {}".format(loopvar))
        print("---------- BEGIN AST ----------")
        print(ast)
        print("---------- END AST   ----------")
        raise e
    return ast, transformer.transposed_buffers, transformer.symbol_table

class FMAReplacer(ast.NodeTransformer):
    def __init__(self):
        self.seen = {}

    def visit_SymbolRef(self, node):
        if node.type is not None:
            self.seen[node.name] = node.type
        return node

    def _get_type(self, node):
        if isinstance(node, C.SymbolRef):
            if node.name == "INFINITY":
                return ctypes.c_float()
            elif node.name in self.seen:
                return self.seen[node.name]
        elif isinstance(node, C.UnaryOp):
            return self._get_type(node.arg)
        elif isinstance(node, C.Constant):
            if isinstance(node.value, int):
                # promote all longs to int
                return ctypes.c_int()
            return ctree.types.get_ctype(node.value)
        elif isinstance(node, C.BinaryOp):
            if isinstance(node.op, C.Op.ArrayRef):
                while not isinstance(node, C.SymbolRef):
                    node = node.left
                pointer_type = self._get_type(node)
                return ctree.types.get_c_type_from_numpy_dtype(pointer_type._dtype_)()
            else:
                left = self._get_type(node.left)
                right = self._get_type(node.right)
                return ctree.types.get_common_ctype([left, right])
        elif isinstance(node, C.FunctionCall):
            if node.func.name in ["MAX", "MIN", "max", "min", "floor"]:
                return ctree.types.get_common_ctype([self._get_type(a) for a in node.args])
        raise NotImplementedError(ast.dump(node))
    def visit_BinaryOp(self, node):
        node.left = self.visit(node.left)
        node.right = self.visit(node.right)
        if isinstance(node.op, C.Op.Add) and isinstance(node.right, C.BinaryOp) and \
            isinstance(node.right.op, C.Op.Mul) and \
            isinstance(self._get_type(node.left), get_simd_type()) and \
            isinstance(self._get_type(node.right.left), get_simd_type()) and \
            isinstance(self._get_type(node.right.right), get_simd_type()):
                # FIXME: Check all are vector types
            return simd_fma(node.right.left, node.right.right, node.left)
        elif isinstance(node.op, C.Op.Add) and \
                isinstance(self._get_type(node.left), get_simd_type()) and \
                isinstance(self._get_type(node.right), get_simd_type()):
            return simd_add(node.left, node.right)
        return node

def fuse_multiply_adds(ast):
    return FMAReplacer().visit(ast)


class VectorLoadReplacer(ast.NodeTransformer):
    def __init__(self, load_stmt, new_stmt):
        self.load_stmt = load_stmt
        self.new_stmt = new_stmt

    def visit_FunctionCall(self, node):
        if node.codegen() == self.load_stmt:
            return self.new_stmt
        node.args = [self.visit(arg) for arg in node.args]
        return node


class VectorLoadCollector(ast.NodeVisitor):
    def __init__(self):
        super().__init__()
        self.loads = {}

    def visit(self, node):
        # Don't descend into nested expressions
        if hasattr(node, 'body'):
            return
        super().visit(node)

    def visit_FunctionCall(self, node):
        if "_mm" in node.func.name and ("_load_" in node.func.name or "_set1" in node.func.name or "_broadcast" in node.func.name):
            if node.codegen() not in self.loads:
                self.loads[node.args[0].codegen()] = [node.args[0], 0, node.func.name]
            self.loads[node.args[0].codegen()][1] += 1
        [self.visit(arg) for arg in node.args]

class VectorLoadStoresRegisterPromoter(ast.NodeTransformer):
    _tmp = -1

    def __init__(self, symbol_map):
        self.sym = deepcopy(symbol_map)

    def _gen_register(self):
        VectorLoadStoresRegisterPromoter._tmp += 1
        return "___x" + str(self._tmp)

    def visit(self, node):
        node = super().visit(node)
        if hasattr(node, 'body'):
            # [collector.visit(s) for s in node.body]
            new_body = []
            seen = {}
            stores = []
            collector = VectorLoadCollector()
            for s in node.body:
                collector.visit(s)
                for stmt in collector.loads.keys():
                    if stmt not in seen:
                        reg = self._gen_register()
                        load_node, number, func = collector.loads[stmt]
                        seen[stmt] = (reg, load_node, func)
                        self.sym[reg] = get_simd_type()()
                        new_body.append(C.Assign(C.SymbolRef(reg, get_simd_type()()),
                                                 C.FunctionCall(C.SymbolRef(func), [load_node])))
                if isinstance(s, C.FunctionCall) and "_mm" in s.func.name and "_store" in s.func.name:
                    if s.args[0].codegen() in seen:
                        stores.append((s.args[0], seen[s.args[0].codegen()][0], s.func.name))
                        s = C.Assign(C.SymbolRef(seen[s.args[0].codegen()][0]), s.args[1])
                for stmt in seen.keys():
                    reg, load_node, func = seen[stmt]
                    replacer = VectorLoadReplacer(
                            C.FunctionCall(C.SymbolRef(func), [load_node]).codegen(), 
                            C.SymbolRef(reg))
                    s = replacer.visit(s)
                new_body.append(s)
            for target, value ,name in stores:
                if "epi32" in name:
                    new_body.append(store_epi32(target, C.SymbolRef(value)))
                elif "ps" in name:
                    new_body.append(store_ps(target, C.SymbolRef(value)))
                else:
                    assert(false)
            node.body = util.flatten(new_body)
        return node

def register_promote_vector_loads_stores(ast, symbol_table):
    transformer = VectorLoadStoresRegisterPromoter(symbol_table)
    ast = transformer.visit(ast)

    return (ast, transformer.sym)
