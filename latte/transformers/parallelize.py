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
import ctree.c.nodes as C
from ctree.templates.nodes import StringTemplate
import latte.config
import latte.util as util
import ctypes
import logging
import numpy as np
from copy import deepcopy
logger = logging.getLogger("latte")

class LatteRuntimeLoopParallel(ast.NodeTransformer):
    def __init__(self, buffers, batch_size):
        self.buffers = buffers
        self.batch_size = batch_size

    def _gen_reduce_for_loop(self, loop, var, size):
        looplen1 = loop.test.right
        loopincr = loop.incr.value.value
        return StringTemplate("""
            //{
            //  ContinueNode *$reduce_node = new ContinueNode(&graph, [=]() {
              parallel_for(0,$size,
                [=](int low, int high) {
                  #pragma simd
                  for (int x = low; x < high; ++x) {
                    float sum = _$arr[x];
                    #pragma unroll
                    for (int i = 1; i < $batch_size; ++ i) {
                      sum += _$arr[i * $size + x];
                    }
                    _$arr[x] = sum;
                  }
                });
            //  });
            //  for (int i = 0; i < $looplen1; i+=$loopincr) {
            //    make_edge($node_list[i], $reduce_node);
            //  }
            //};
            """, {'size': C.Constant(size),
                  'batch_size': C.Constant(self.batch_size),
                  'arr': C.SymbolRef(var),
                  'node_list': C.SymbolRef("node_list_"),
                  'reduce_node': C.SymbolRef("reduce_node_"),
                  'looplen1': C.Constant(looplen1.value),  
                  'loopincr': C.Constant(loopincr)
                  })

    def visit_For(self, node):
        node.body = [self.visit(s) for s in node.body]
        if hasattr(node, 'parallel') and node.parallel:
            to_return = [StringTemplate("""
                parallel_for(0,$looplen / $loopincr,
                  [=](int low, int high) {
                    for (int tmp_$loopvar = low; tmp_$loopvar < high; tmp_$loopvar++) {
                      int $loopvar = tmp_$loopvar * $loopincr;
                      $body;
                    }
                  }
                );
                """, {
                    "looplen": node.test.right,
                    "loopvar": node.test.left,
                    "loopincr": node.incr.value,
                    "body": node.body
                })]
            if hasattr(node, 'reduce_vars') and len(node.reduce_vars) > 0:
                for var in node.reduce_vars:
                    size = np.prod(self.buffers[var].shape[1:])
                    to_return.append(self._gen_reduce_for_loop(node, var, size))
            return to_return
        return node

class LatteTBBLoopParallel(ast.NodeTransformer):
    def __init__(self, buffers, batch_size):
        self.buffers = buffers
        self.batch_size = batch_size

    def _gen_reduce_for_loop(self, loop, var, size):
        looplen1 = loop.test.right
        loopincr = loop.incr.value.value
        return StringTemplate("""
            //{
            //  ContinueNode *$reduce_node = new ContinueNode(&graph, [=]() {
              parallel_for(0,$size,
                [=](int low, int high) {
                  #pragma simd
                  for (int x = low; x < high; ++x) {
                    float sum = _$arr[x];
                    #pragma unroll
                    for (int i = 1; i < $batch_size; ++ i) {
                      sum += _$arr[i * $size + x];
                    }
                    _$arr[x] = sum;
                  }
                });
            //  });
            //  for (int i = 0; i < $looplen1; i+=$loopincr) {
            //    make_edge($node_list[i], $reduce_node);
            //  }
            //};
            """, {'size': C.Constant(size),
                  'batch_size': C.Constant(self.batch_size),
                  'arr': C.SymbolRef(var),
                  'node_list': C.SymbolRef("node_list_"),
                  'reduce_node': C.SymbolRef("reduce_node_"),
                  'looplen1': C.Constant(looplen1.value),  
                  'loopincr': C.Constant(loopincr)
                  })

    def visit_For(self, node):
        node.body = [self.visit(s) for s in node.body]
        if hasattr(node, 'parallel') and node.parallel:
            to_return = [StringTemplate("""
                parallel_for(0,$looplen / $loopincr,
                  [=](int low, int high) {
                    for (int tmp_$loopvar = low; tmp_$loopvar < high; tmp_$loopvar++) {
                      int $loopvar = tmp_$loopvar * $loopincr;
                      $body;
                    }
                  }
                );
                """, {
                    "looplen": node.test.right,
                    "loopvar": node.test.left,
                    "loopincr": node.incr.value,
                    "body": node.body
                })]
            if hasattr(node, 'reduce_vars') and len(node.reduce_vars) > 0:
                for var in node.reduce_vars:
                    size = np.prod(self.buffers[var].shape[1:])
                    to_return.append(self._gen_reduce_for_loop(node, var, size))
            return to_return
        return node

class LatteOpenMPParallel(ast.NodeTransformer):
    def __init__(self, buffers, batch_size):
        self.buffers = buffers
        self.batch_size = batch_size

    def _gen_reduce_for_loop(self, loop, var, size):
        looplen1 = loop.test.right
        loopincr = loop.incr.value.value
        return StringTemplate("""
              #pragma omp parallel for simd
              for (int x = 0; x < $size; ++x) {
                float sum = _$arr[x];
                #pragma unroll
                for (int i = 1; i < $batch_size; ++ i) {
                  sum += _$arr[i * $size + x];
                }
                _$arr[x] = sum;
              }
            """, {'size': C.Constant(size),
                  'batch_size': C.Constant(self.batch_size),
                  'arr': C.SymbolRef(var),
                  'node_list': C.SymbolRef("node_list_"),
                  'reduce_node': C.SymbolRef("reduce_node_"),
                  'looplen1': C.Constant(looplen1.value),  
                  'loopincr': C.Constant(loopincr)
                  })

    def visit_For(self, node):
        if hasattr(node, 'parallel') and node.parallel:
            to_return = []
            # Supports depth one nesting with collapse
            if all(isinstance(s, C.For) and hasattr(s, 'parallel') and s.parallel for s in node.body):
                for s in node.body:
                    to_return.append(
                        C.For(node.init, node.test, node.incr, [s])
                    )
                    to_return[-1].pragma = "omp parallel for collapse(2)"
            else:
                node.pragma = "omp parallel for"
                to_return = [node]
            if hasattr(node, 'reduce_vars') and len(node.reduce_vars) > 0:
                for var in node.reduce_vars:
                    size = np.prod(self.buffers[var].shape[1:])
                    to_return.append(self._gen_reduce_for_loop(node, var, size))
            return to_return
        return node

class CollectArrayReferences(ast.NodeVisitor):
    def __init__(self, seen):
        self.seen = seen

    def visit_BinaryOp(self, node):
        if isinstance(node.op, C.Op.ArrayRef):
            while not isinstance(node, C.SymbolRef):
                node = node.left
            self.seen.add(node.name)
        else:
            self.visit(node.left)
            self.visit(node.right)

if latte.config.parallel_strategy == "OPENCL_SIMPLE_LOOP":
    import pycl as cl

class LatteOpenCLSimpleLoopParallel(ast.NodeTransformer):
    kernel_id = -1

    def __init__(self, buffers, cl_buffers, kernels, batch_size):
        self.buffers = buffers
        self.cl_buffers = cl_buffers
        self.kernels = kernels
        self.batch_size = batch_size

    def _gen_unique_kernel_name(self):
        LatteOpenCLSimpleLoopParallel.kernel_id += 1
        return "latte_kernel_{}".format(LatteOpenCLSimpleLoopParallel.kernel_id)

    def _gen_reduce_for_loop(self, loop, var, size):
        looplen1 = loop.test.right
        loopincr = loop.incr.value.value
        kernel_name = self._gen_unique_kernel_name()
        kernel_src = StringTemplate("""
          __kernel void $kernel_name(__global float * $arr) {
            int x = get_global_id(0);
            float sum = $arr[x];
            #pragma unroll
            for (int i = 1; i < $batch_size; ++ i) {
              sum += $arr[i * $size + x];
            }
            $arr[x] = sum;
          }
        """, {'batch_size': C.Constant(self.batch_size),
              'arr': C.SymbolRef(var),
              'size': C.Constant(size),
              'kernel_name': C.SymbolRef(kernel_name)})
        program = cl.clCreateProgramWithSource(
            latte.config.cl_ctx, kernel_src.codegen()).build()
        kernel = program[kernel_name]
        self.kernels[kernel_name] = kernel
        kernel.setarg(0, self.cl_buffers[var], ctypes.sizeof(cl.cl_mem))
        return StringTemplate(
            """
            size_t global_size_{kernel_name}[1] = {{{looplen1}}};
            clEnqueueNDRangeKernel(queue, {kernel_name}, 1, NULL, global_size_{kernel_name}, NULL, 0, NULL, NULL);
            clFinish(queue);
            """.format(
                kernel_name=kernel_name, 
                looplen1=size)
        )

    def build_kernel(self, kernel_src, kernel_name, kernel_args):
        kernel_src = C.CFile('generated', [StringTemplate(
"""
#define MIN(x, y) (((x) < (y)) ? (x) : (y))
#define MAX(x, y) (((x) > (y)) ? (x) : (y))
"""
            ), kernel_src])
        try:
            program = cl.clCreateProgramWithSource(
                latte.config.cl_ctx, kernel_src.codegen()).build()
            kernel = program[kernel_name]
        except cl.BuildProgramFailureError as e:
            logger.error("Failed build program:\n %s", kernel_src.codegen())
            raise e
        self.kernels[kernel_name] = kernel
        for index, arg in enumerate(kernel_args):
            kernel.setarg(index, self.cl_buffers[arg], ctypes.sizeof(cl.cl_mem))
        logger.debug(kernel_src)

    def collect_args_and_insert_casts(self, kernel_args, body):
        [CollectArrayReferences(kernel_args).visit(s) for s in body]
        params = []
        for arg in kernel_args:
            buf = self.buffers[arg]
            typ = np.ctypeslib.ndpointer(buf.dtype, buf.ndim, buf.shape)
            params.append(C.SymbolRef("_" + arg, typ()))
            params[-1].set_global()
            util.insert_cast(body, buf.shape[1:], arg, buf.dtype, _global=True)
        return params

    def visit_For(self, node):
        if hasattr(node, 'parallel') and node.parallel:
            # Supports depth one nesting with collapse
            loopvar1 = node.init.left.name
            looplen1 = node.test.right
            to_return = []
            if all(isinstance(s, C.For) and hasattr(s, 'parallel') and s.parallel for s in node.body):
                for s in node.body:
                    body = s.body
                    kernel_args = set()
                    loopvar2 = s.init.left.name
                    looplen2 = s.test.right
                    kernel_name = self._gen_unique_kernel_name()
                    params = self.collect_args_and_insert_casts(kernel_args, body)
                    body.insert(0, C.Assign(
                        C.SymbolRef(loopvar1, ctypes.c_int()), 
                        C.FunctionCall(C.SymbolRef("get_global_id"), [C.Constant(0)])
                    ))
                    body.insert(0, C.Assign(
                        C.SymbolRef(loopvar2, ctypes.c_int()), 
                        C.FunctionCall(C.SymbolRef("get_global_id"), [C.Constant(1)])
                    ))
                    kernel_src = C.FunctionDecl(None, C.SymbolRef(kernel_name), params, body)
                    kernel_src.set_kernel()
                    self.build_kernel(kernel_src, kernel_name, kernel_args)
                    to_return.append(StringTemplate(
                        """
                        size_t global_size_{kernel_name}[2] = {{{looplen1}, {looplen2}}};
                        clEnqueueNDRangeKernel(queue, {kernel_name}, 2, NULL, global_size_{kernel_name}, NULL, 0, NULL, NULL);
                        clFinish(queue);
                        """.format(
                            kernel_name=kernel_name, 
                            looplen1=looplen1,
                            looplen2=looplen2)
                    ))
            else:
                kernel_args = set()
                body = node.body
                kernel_name = self._gen_unique_kernel_name()
                params = self.collect_args_and_insert_casts(kernel_args, body)
                body.insert(0, C.Assign(
                    C.SymbolRef(loopvar1, ctypes.c_int()), 
                    C.FunctionCall(C.SymbolRef("get_global_id"), [C.Constant(0)])
                ))
                kernel_src = C.FunctionDecl(None, C.SymbolRef(kernel_name), params, body)
                kernel_src.set_kernel()
                self.build_kernel(kernel_src, kernel_name, kernel_args)
                to_return.append(StringTemplate(
                    """
                    size_t global_size_{kernel_name}[1] = {{{looplen1}}};
                    clEnqueueNDRangeKernel(queue, {kernel_name}, 1, NULL, global_size_{kernel_name}, NULL, 0, NULL, NULL);
                    clFinish(queue);
                    """.format(
                        kernel_name=kernel_name, 
                        looplen1=looplen1)
                ))
            if hasattr(node, 'reduce_vars') and len(node.reduce_vars) > 0:
                for var in node.reduce_vars:
                    size = np.prod(self.buffers[var].shape[1:])
                    to_return.append(self._gen_reduce_for_loop(node, var, size))
            return to_return

        else:
            raise NotImplementedError(node)
        return node

def parallelize(tree, buffers, cl_buffers, kernels, batch_size):
    if latte.config.parallel_strategy == "SIMPLE_LOOP":
        return LatteTBBLoopParallel(buffers, batch_size).visit(tree)
    elif latte.config.parallel_strategy == "FLOWGRAPH_LOOP":
        raise NotImplementedError()
    elif latte.config.parallel_strategy == "OPENMP" or latte.config.parallel_strategy == "LIBXSMMOPENMP":
        return LatteOpenMPParallel(buffers, batch_size).visit(tree)
    elif latte.config.parallel_strategy == "OPENCL_SIMPLE_LOOP":
        return LatteOpenCLSimpleLoopParallel(buffers, cl_buffers, kernels, batch_size).visit(tree)
    else:
        raise NotImplementedError()

def tbb_parallelize(tree, buffers, cl_buffers, kernels, batch_size):
    return LatteTBBLoopParallel(buffers, batch_size).visit(tree)
    
