import ast
import ctree.c.nodes as C
from ctree.templates.nodes import StringTemplate
import latte.config
import latte.util as util
import ctypes
import logging
import numpy as np
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
                  for (int x = low; x < high; ++x) {
                    float sum = _$arr[x];
                    #pragma unroll($batch_size - 1)
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
    def visit_For(self, node):
        if hasattr(node, 'parallel') and node.parallel:
            node.pragma = "omp parallel for"
            # Supports depth one nesting with collapse
            if len(node.body) == 1 and hasattr(node.body[0], 'parallel') and \
                    node.body[0].parallel:
                node.pragma += " collapse(2)"
            elif all(isinstance(s, C.For) and hasattr(s, 'parallel') and s.parallel for s in node.body):
                # FIXME: Distribute the loops and use collapse
                raise NotImplementedError()
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

    def __init__(self, buffers, cl_buffers, kernels):
        self.buffers = buffers
        self.cl_buffers = cl_buffers
        self.kernels = kernels

    def _gen_unique_kernel_name(self):
        LatteOpenCLSimpleLoopParallel.kernel_id += 1
        return "latte_kernel_{}".format(LatteOpenCLSimpleLoopParallel.kernel_id)

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
                    [CollectArrayReferences(kernel_args).visit(s) for s in body]
                    loopvar2 = s.init.left.name
                    looplen2 = s.test.right
                    kernel_name = self._gen_unique_kernel_name()
                    params = [C.SymbolRef("_" + arg, ctypes.POINTER(ctypes.c_float)()) for arg in kernel_args]
                    [p.set_global() for p in params]
                    for arg in kernel_args:
                        buffer = self.buffers[arg]
                        util.insert_cast(body, buffer.shape[1:], arg, buffer.dtype, _global=True)
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
                return to_return
            else:
                raise NotImplementedError()
        else:
            raise NotImplementedError(node)
        return node

def parallelize(tree, buffers, cl_buffers, kernels, batch_size):
    if latte.config.parallel_strategy == "SIMPLE_LOOP":
        return LatteRuntimeLoopParallel(buffers, batch_size).visit(tree)
    elif latte.config.parallel_strategy == "FLOWGRAPH_LOOP":
        raise NotImplementedError()
    elif latte.config.parallel_strategy == "OPENMP":
        return LatteOpenMPParallel().visit(tree)
    elif latte.config.parallel_strategy == "OPENCL_SIMPLE_LOOP":
        return LatteOpenCLSimpleLoopParallel(buffers, cl_buffers, kernels).visit(tree)
    else:
        raise NotImplementedError()
